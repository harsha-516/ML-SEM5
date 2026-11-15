"""Telegram bot: risk replies, subscriptions and logged alerts. Run with python -m hazards.bot."""

from __future__ import annotations

import csv
import json
import sqlite3
import time
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path

import numpy as np
import requests
from dotenv import dotenv_values

from .common import RESULTS, ROOT, log

DB = ROOT / "data" / "bot.sqlite"
PLACES = Path(__file__).with_name("places.csv")
NEAR_KM = 30
COMPASS = ("north", "northeast", "east", "southeast", "south", "southwest", "west", "northwest")
STORMS = ROOT / "dashboard" / "public" / "data" / "storms.json"
HAZARDS = ("cyclone", "flood", "earthquake")
LEVELS = ("Low", "Medium", "High")
CHECK_S = 300
REPEAT_DAYS = 7
STORM_MAX_HOURS = 1  # the refresh loop writes storms every 5 minutes
MAP_MAX_HOURS = 48  # maps rescore once a UTC day
TYPHOON_10MIN_KT = 56  # 64 kt on the 1-minute scale
UNAVAILABLE = {"unavailable": True}
NORTH_INDIAN_EAST = 100  # IBTrACS is the only live source west of here
REGIONS = {
    "india": (6, 36, 68, 98),
    "bangladesh": (20, 27, 88, 93),
    "nepal": (26, 31, 80, 89),
    "pakistan": (23, 37, 60, 77),
    "srilanka": (5, 10, 79, 83),
    "myanmar": (9, 29, 92, 102),
    "thailand": (5, 21, 97, 106),
    "vietnam": (8, 24, 102, 110),
    "philippines": (4, 21, 116, 127),
    "indonesia": (-11, 6, 95, 141),
    "malaysia": (0, 8, 99, 120),
    "china": (18, 54, 73, 135),
    "japan": (24, 46, 122, 146),
    "taiwan": (21, 26, 119, 123),
    "bayofbengal": (5, 23, 80, 95),
    "westpacific": (0, 40, 120, 180),
    "asia": (-15, 60, 60, 180),
}
COUNTRIES = {
    "srilanka": "Sri Lanka",
    **{
        r: r.title()
        for r in (
            "india",
            "bangladesh",
            "nepal",
            "pakistan",
            "myanmar",
            "thailand",
            "vietnam",
            "philippines",
            "indonesia",
            "malaysia",
            "china",
            "japan",
            "taiwan",
        )
    },
}
NAMES = {"srilanka": "Sri Lanka", "bayofbengal": "Bay of Bengal", "westpacific": "West Pacific"}
SAFETY = {
    "cyclone": "Secure loose items and keep away from the coast if told to.",
    "flood": "Avoid low ground and river banks.",
    "earthquake": "Know your safe spots and keep an emergency kit ready.",
}
NOTICE = "This alert is model-based. Follow official agency evacuation instructions."
HELP = (
    "Commands\n/risk <hazard> <region>\n/subscribe <hazard> <region>\n/unsubscribe <hazard> <region>\n/help\n\n"
    f"Hazards: {', '.join(HAZARDS)}\nRegions: {', '.join(REGIONS)}\n\n"
    "Earthquake output is a risk level for a region and 30-day window, never a prediction of when an earthquake will occur. Risk levels are not comparable across hazards.\n\nPlace names: GeoNames, CC BY 4.0."
)


def fresh(stamp: str | None, hours: float, now: datetime) -> bool:
    """True when a timestamp is newer than the given number of hours."""
    try:
        t = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    except (AttributeError, ValueError):
        return False
    t = t if t.tzinfo else t.replace(tzinfo=UTC)
    return (now - t).total_seconds() <= hours * 3600


def level(p: float, cuts: dict) -> str:
    """Turn a chance into Low, Medium or High using the validation cut-offs."""
    return "High" if p >= cuts["high"] else "Medium" if p >= cuts["medium"] else "Low"


def inside(lat: float, lon: float, region: str) -> bool:
    """True when a point lies in the region box."""
    a, b, c, d = REGIONS[region]
    return a <= lat <= b and c <= lon <= d


@lru_cache(maxsize=1)
def load_places() -> tuple[list[dict], np.ndarray]:
    """Load the town table and its coordinates once."""
    with open(PLACES, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return rows, np.radians([[float(r["lat"]), float(r["lon"])] for r in rows])


def nearest(lat: float, lon: float) -> tuple[dict, float, np.ndarray]:
    """Nearest town, its distance in km and its position in radians."""
    rows, xy = load_places()
    a, b = np.radians(lat), np.radians(lon)
    h = np.sin((xy[:, 0] - a) / 2) ** 2 + np.cos(a) * np.cos(xy[:, 0]) * np.sin((xy[:, 1] - b) / 2) ** 2
    i = int(np.argmin(h))
    return rows[i], 2 * 6371 * np.arcsin(np.sqrt(h[i])), xy[i]


def in_country(lat: float, lon: float, region: str) -> bool:
    """Boxes cross borders, so a country region keeps points whose nearest town is in that country."""
    if region not in COUNTRIES:
        return True
    try:
        return nearest(lat, lon)[0]["country"] == COUNTRIES[region]
    except OSError:
        return True


def place(lat: float, lon: float) -> str:
    """Area name from the nearest town. Without the table the message keeps coordinates."""
    try:
        r, km, town = nearest(lat, lon)
    except OSError:
        return f"{lat}, {lon}"
    a, b = np.radians(lat), np.radians(lon)
    parts = list(dict.fromkeys(x for x in (r["name"], r["state"], r["country"]) if x))
    name = ", ".join(parts)
    if km <= NEAR_KM:
        return f"near {name}"
    dlat, dlon = town[0] - a, town[1] - b  # bearing runs from the town to the point
    bearing = np.degrees(np.arctan2(-dlon * np.cos(a), -dlat)) % 360
    return f"about {round(km, -1):.0f} km {COMPASS[int((bearing + 22.5) // 45) % 8]} of {name}"


def read_results(hazard: str, base: Path = RESULTS) -> dict:
    """Read one hazard results file."""
    return json.loads((base / f"{hazard}.json").read_text())


def map_status(hazard: str, region: str, r: dict) -> dict | None:
    """Highest chance in a region from a flood or earthquake map."""
    rows = [
        x for x in r["current_risk"] if inside(x["lat"], x["lon"], region) and in_country(x["lat"], x["lon"], region)
    ]
    if not rows:
        return None
    top = max(rows, key=lambda x: x["p"])
    return {
        "level": level(top["p"], r["risk_levels"]),
        "p": top["p"],
        "lat": top["lat"],
        "lon": top["lon"],
        "issued": r.get("scored_at", "")[:10],
    }


def storm_status(region: str, storms: list[dict], cuts: dict) -> dict | None:
    """Strongest active storm in a region. A storm at typhoon force is High, even if the model could not score it."""
    rows = [s for s in storms if inside(s["lat"], s["lon"], region)]
    if not rows:
        return None
    top = max(rows, key=lambda s: s["model"]["wind_now_1min"] if s.get("model") else s.get("wind_kt", 0) / 0.88)
    base = {"lat": top["lat"], "lon": top["lon"], "name": top["name"]}
    if not top.get("model"):
        high = top.get("wind_kt", 0) >= TYPHOON_10MIN_KT
        return {
            **base,
            "level": "High" if high else "Medium",
            "p": None,
            "issued": top.get("issued", ""),
            "scored": False,
            "wind": top.get("wind_kt", 0),
        }
    m = top["model"]
    p = m["reach_64_chance"]
    return {
        **base,
        "level": "High" if p is None else level(p, cuts),
        "p": p,
        "issued": m["from"],
        "scored": True,
        "wind": m["wind_now_1min"],
        "low": m["low"],
        "high": m["high"],
    }


def status(hazard: str, region: str, now: datetime | None = None) -> dict | None:
    """Current state of a region. Failed or old data returns UNAVAILABLE, never a quiet answer."""
    now = now or datetime.now(UTC)
    try:
        r = read_results(hazard)
        if hazard == "cyclone":
            feed = json.loads(STORMS.read_text())
            if feed.get("status") != "ok" or not fresh(feed.get("fetched_at"), STORM_MAX_HOURS, now):
                return UNAVAILABLE
            if feed.get("north_indian") != "ok" and REGIONS[region][2] < NORTH_INDIAN_EAST:
                return UNAVAILABLE
            return storm_status(region, feed["storms"], r["risk_levels"])
        if not fresh(r.get("scored_at"), MAP_MAX_HOURS, now):
            return UNAVAILABLE
        return map_status(hazard, region, r)
    except (OSError, ValueError, KeyError, TypeError):
        return UNAVAILABLE


def text_for(hazard: str, region: str, s: dict | None) -> str:
    """Write the message for one hazard and region, with the safety wording."""
    zone = NAMES.get(region, region.title())
    if s is not None and s.get("unavailable"):
        return f"{hazard.title()}, {zone}: data is unavailable or out of date, so risk is unknown. This does not mean there is no risk. Check official sources."
    if s is None:
        return (
            f"{hazard.title()}, {zone}: no active storm."
            if hazard == "cyclone"
            else f"{hazard.title()}, {zone}: this region has no scored location."
        )
    if hazard == "cyclone" and not s["scored"]:
        detail = f"Storm {s['name']}, {place(s['lat'], s['lon'])}, {s['wind']:.0f} kt from the agency feed. The model could not score it."
    elif hazard == "cyclone":
        chance = "already at typhoon force" if s["p"] is None else f"{s['p']:.0%} chance of reaching 64 kt in 24 hours"
        detail = f"Storm {s['name']}, {place(s['lat'], s['lon'])}, {s['wind']:.0f} kt now, {chance}. 24 hour wind range {s['low']:.0f} to {s['high']:.0f} kt."
    elif hazard == "flood":
        detail = (
            f"Highest area {place(s['lat'], s['lon'])}: {s['p']:.0%} chance of a flood report surge in the next 3 days."
        )
    else:
        detail = f"Highest area, a square roughly 100 km wide centred {place(s['lat'], s['lon'])}: {s['p']:.0%} chance of M5.0+ in the 30-day window. This is a risk level for a region and time window, never a predicted time."
    return f"{hazard.title()} risk, {zone}: {s['level']}\nPrediction time: {s['issued']}\n{detail}\n{SAFETY[hazard]}\n{NOTICE}"


def connect(path: Path = DB) -> sqlite3.Connection:
    """Open the bot database and create its tables if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.execute(
        "create table if not exists subscribers (chat_id integer, hazard text, region text, primary key (chat_id, hazard, region))"
    )
    con.execute(
        "create table if not exists alert_state (chat_id integer, hazard text, region text, level text, event text, primary key (chat_id, hazard, region))"
    )
    con.execute(
        "create table if not exists alerts (id integer primary key, chat_id integer, hazard text, region text, level text, issued text, message text, created text, sent integer default 0)"
    )
    return con


def should_send(prev: tuple | None, last_sent: str | None, lvl: str, event: str, now: datetime) -> bool:
    """Alert on entering High or on a different event at High. Repeat weekly only during one unbroken High."""
    if lvl != "High":
        return False
    if prev is None or prev[0] != "High" or prev[1] != event:
        return True
    return last_sent is None or (now - datetime.fromisoformat(last_sent)).days >= REPEAT_DAYS


def queue_alerts(con: sqlite3.Connection, now: datetime, look=status) -> int:
    """Store a new alert for each subscriber whose status needs one. Returns how many."""
    n = 0
    for chat, hazard, region in con.execute("select chat_id, hazard, region from subscribers").fetchall():
        s = look(hazard, region)
        key = (chat, hazard, region)
        if s is not None and s.get("unavailable"):  # unknown is not quiet, so keep the state as it was
            continue
        if s is None and hazard != "cyclone":
            continue
        lvl, event = ("Low", "") if s is None else (s["level"], s.get("name", ""))
        prev = con.execute(
            "select level, event from alert_state where chat_id = ? and hazard = ? and region = ?", key
        ).fetchone()
        last = con.execute(
            "select created from alerts where chat_id = ? and hazard = ? and region = ? order by id desc limit 1", key
        ).fetchone()
        if should_send(prev, last[0] if last else None, lvl, event, now):
            con.execute(
                "insert into alerts (chat_id, hazard, region, level, issued, message, created) values (?,?,?,?,?,?,?)",
                (chat, hazard, region, lvl, s["issued"], text_for(hazard, region, s), now.isoformat()),
            )
            n += 1
        con.execute("insert or replace into alert_state values (?,?,?,?,?)", (*key, lvl, event))
    con.commit()
    return n


def handle(con: sqlite3.Connection, chat: int, text: str, look=status) -> str:
    """Answer one incoming command and return the reply text."""
    parts = text.strip().split()
    cmd = parts[0].split("@")[0].lower() if parts else ""
    if cmd in ("/start", "/help"):
        return ("Tarang sends model-based hazard risk alerts for Asia.\n\n" if cmd == "/start" else "") + HELP
    if cmd not in ("/risk", "/subscribe", "/unsubscribe"):
        return WELCOME
    hazard = parts[1].lower() if len(parts) > 1 else ""
    region = "".join(parts[2:]).lower().replace("-", "")
    if hazard not in HAZARDS or region not in REGIONS:
        return f"Use: {cmd} <hazard> <region>. Send /help for the lists."
    if cmd == "/risk":
        return text_for(hazard, region, look(hazard, region))
    if cmd == "/subscribe":
        con.execute("insert or ignore into subscribers values (?,?,?)", (chat, hazard, region))
        con.commit()
        return f"Subscribed to {hazard}, {NAMES.get(region, region.title())}. You get a message when risk reaches High."
    con.execute("delete from subscribers where chat_id = ? and hazard = ? and region = ?", (chat, hazard, region))
    con.commit()
    return f"Unsubscribed from {hazard}, {NAMES.get(region, region.title())}."


WELCOME = "Hi! I am Tarang. I give model-based hazard risk levels for Asia. Pick an option below."
ACTIONS = {"risk": "Check risk", "sub": "Subscribe to alerts", "unsub": "Unsubscribe"}
ICONS = {"cyclone": "🌀", "flood": "🌊", "earthquake": "🫨"}


def btn(label: str, data: str) -> dict:
    return {"text": label, "callback_data": data}


def menu_markup() -> dict:
    rows = [[btn(f"🔎 {ACTIONS['risk']}", "m|risk")], [btn("🔔 Subscribe", "m|sub"), btn("🔕 Unsubscribe", "m|unsub")]]
    rows.append([btn("📋 My subscriptions", "mine"), btn("ℹ️ Help", "help")])
    return {"inline_keyboard": rows}


def hazard_markup(action: str) -> dict:
    rows = [[btn(f"{ICONS[h]} {h.title()}", f"h|{action}|{h}")] for h in HAZARDS]
    return {"inline_keyboard": rows + [[btn("⬅️ Menu", "menu")]]}


def region_markup(action: str, hazard: str) -> dict:
    cells = [btn(NAMES.get(r, r.title()), f"r|{action}|{hazard}|{r}") for r in REGIONS]
    rows = [cells[i : i + 3] for i in range(0, len(cells), 3)]
    return {"inline_keyboard": rows + [[btn("⬅️ Back", f"m|{action}"), btn("Menu", "menu")]]}


def subs_markup(con: sqlite3.Connection, chat: int) -> dict | None:
    subs = con.execute("select hazard, region from subscribers where chat_id = ?", (chat,)).fetchall()
    if not subs:
        return None
    rows = [[btn(f"❌ {ICONS.get(h, '')} {h.title()}, {NAMES.get(r, r.title())}", f"r|unsub|{h}|{r}")] for h, r in subs]
    return {"inline_keyboard": rows + [[btn("⬅️ Menu", "menu")]]}


def after(text: str) -> tuple[str, dict]:
    """Attach the main menu to a result so the user can keep going."""
    return text, menu_markup()


def press(con: sqlite3.Connection, chat: int, data: str, look=status) -> tuple[str, dict | None]:
    """Answer one button press with reply text and the next set of buttons."""
    p = data.split("|")
    if p[0] == "menu":
        return "What would you like to do?", menu_markup()
    if p[0] == "help":
        return after(HELP)
    if p[0] == "mine":
        subs = con.execute("select hazard, region from subscribers where chat_id = ?", (chat,)).fetchall()
        if not subs:
            return after("No subscriptions yet. Tap Subscribe to add one.")
        return after("Your subscriptions:\n" + "\n".join(f"- {h}, {NAMES.get(r, r.title())}" for h, r in subs))
    if p[0] == "m" and len(p) == 2 and p[1] in ACTIONS:
        if p[1] == "unsub":
            markup = subs_markup(con, chat)
            return ("Tap a subscription to remove it:", markup) if markup else after("You have no subscriptions.")
        return "Choose a hazard:", hazard_markup(p[1])
    if p[0] == "h" and len(p) == 3 and p[2] in HAZARDS:
        return "Choose a region:", region_markup(p[1], p[2])
    if p[0] == "r" and len(p) == 4 and p[1] in ACTIONS and p[2] in HAZARDS and p[3] in REGIONS:
        cmd = {"risk": "risk", "sub": "subscribe", "unsub": "unsubscribe"}[p[1]]
        return after(handle(con, chat, f"/{cmd} {p[2]} {p[3]}", look))
    return after(WELCOME)


class Telegram:
    def __init__(self, token: str):
        self.base = f"https://api.telegram.org/bot{token}"

    def send(self, chat: int, text: str, markup: dict | None = None) -> bool:
        body = {"chat_id": chat, "text": text}
        if markup:
            body["reply_markup"] = markup
        r = requests.post(f"{self.base}/sendMessage", json=body, timeout=30)
        return r.ok

    def answer(self, callback_id: str) -> None:
        requests.post(f"{self.base}/answerCallbackQuery", json={"callback_query_id": callback_id}, timeout=30)

    def set_commands(self) -> None:
        cmds = [{"command": "start", "description": "Open the menu"}, {"command": "help", "description": "Help"}]
        requests.post(f"{self.base}/setMyCommands", json={"commands": cmds}, timeout=30)

    def updates(self, offset: int) -> list[dict]:
        r = requests.get(f"{self.base}/getUpdates", params={"offset": offset, "timeout": 25}, timeout=40)
        r.raise_for_status()
        return r.json()["result"]


def send_pending(con: sqlite3.Connection, bot: Telegram) -> None:
    """Send stored alerts and mark each one as sent."""
    for i, chat, message in con.execute("select id, chat_id, message from alerts where sent = 0").fetchall():
        if bot.send(chat, message):
            con.execute("update alerts set sent = 1 where id = ?", (i,))
    con.commit()


def main() -> None:
    """Poll Telegram for commands and check for alerts on a schedule."""
    token = (dotenv_values(ROOT / ".env").get("TELEGRAM_BOT_TOKEN") or "").strip()
    bot, con, offset, checked = Telegram(token), connect(), 0, 0.0
    try:
        bot.set_commands()
    except requests.RequestException:
        pass
    log("Telegram bot start")
    while True:
        try:
            for u in bot.updates(offset):
                offset = u["update_id"] + 1
                cb = u.get("callback_query")
                if cb and cb.get("message"):
                    bot.answer(cb["id"])
                    text, markup = press(con, cb["message"]["chat"]["id"], cb.get("data", ""))
                    bot.send(cb["message"]["chat"]["id"], text, markup)
                    continue
                m = u.get("message") or {}
                if m.get("text"):
                    chat, raw = m["chat"]["id"], m["text"]
                    reply = handle(con, chat, raw)
                    words = raw.strip().split()
                    cmd = words[0].split("@")[0].lower() if words else ""
                    typed = cmd in ("/risk", "/subscribe", "/unsubscribe", "/help")
                    bot.send(chat, reply, None if typed and reply != WELCOME else menu_markup())
            if time.time() - checked > CHECK_S:
                checked = time.time()
                if queue_alerts(con, datetime.now(UTC)):
                    log("Telegram alerts queued")
                send_pending(con, bot)
        except (requests.RequestException, OSError, ValueError, KeyError) as exc:
            log(f"Telegram bot retry: {type(exc).__name__}")  # the URL holds the token, so never log the message
            time.sleep(30)


if __name__ == "__main__":
    main()
