"""Step 5: does a small CNN on Himawari storm pictures add skill? Writes results/exp_cnn.json.

The CNN score enters XGBoost as one feature. Training rows get scores from folds that left their storm out, so XGBoost never sees a score fitted on its own row.
Runs on CPU.
"""

import json

import numpy as np
import pandas as pd
import torch
from cyclone_experiments import compare, split_years
from sklearn.model_selection import GroupKFold
from torch import nn

from hazards import cyclone
from hazards import himawari as hw
from hazards.common import DATA, RESULTS, log
from hazards.typhoon import build_table

EPOCHS = 15
torch.set_num_threads(6)


def net() -> nn.Module:
    """Small CNN with one output."""

    def block(a, b):
        return nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(), nn.MaxPool2d(2))

    return nn.Sequential(
        block(1, 16),
        block(16, 32),
        block(32, 64),
        block(64, 64),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Dropout(0.3),
        nn.Linear(64, 1),
    )


def turn(x: torch.Tensor) -> torch.Tensor:
    """Storm pictures keep their meaning under quarter turns."""
    k = int(torch.randint(4, (1,)))
    return torch.rot90(x, k, (2, 3))


def fit_cnn(x: np.ndarray, y: np.ndarray, kind: str, seed: int) -> nn.Module:
    """Train one CNN."""
    torch.manual_seed(seed)
    m = net()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-3)
    X, Y = torch.tensor(x), torch.tensor(y, dtype=torch.float32)
    if kind == "class":
        loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(min(float((Y == 0).sum() / max(Y.sum(), 1)), 20.0)))
    else:
        loss = nn.L1Loss()
    for _ in range(EPOCHS):
        m.train()
        for i in torch.randperm(len(X)).split(64):
            opt.zero_grad()
            loss(m(turn(X[i])).squeeze(1), Y[i]).backward()
            opt.step()
    return m.eval()


@torch.no_grad()
def score(m: nn.Module, x: np.ndarray) -> np.ndarray:
    """CNN output for the pictures."""
    return np.concatenate([m(b).squeeze(1).numpy() for b in torch.tensor(x).split(256)])


def images(d: pd.DataFrame) -> np.ndarray:
    """Load the pictures for these rows."""
    names = [hw.crop_name(s, t) for s, t in zip(d.storm, d.time)]
    return np.stack([np.load(DATA / "himawari_crops" / f"{n}.npy") for n in names])[:, None].astype(np.float32) / 255


def with_scores(tr, va, te, target) -> tuple:
    """Add a CNN score column. Training rows get scores from folds that left out their storm."""
    kind = "class" if target != "wind" else "reg"
    ytr = tr[target].to_numpy(float) if kind == "class" else (tr.wind_24 - tr.wind).to_numpy(float) / 20
    a, b, c = images(tr), images(va), images(te)
    oof = np.zeros(len(tr))
    for i, j in GroupKFold(3).split(a, groups=tr.storm):
        oof[j] = score(fit_cnn(a[i], ytr[i], kind, 0), a[j])
    full = fit_cnn(a, ytr, kind, 0)
    log(f"    CNN fitted for {target}")
    return tr.assign(cnn=oof), va.assign(cnn=score(full, b)), te.assign(cnn=score(full, c))


def main() -> None:
    """Run the experiment and save the table."""
    log("CNN experiment start")
    tracks = cyclone.load_tracks([DATA / "ibtracs_NI.csv", DATA / "ibtracs_WP.csv"])
    d = build_table(tracks, first_year=1980, keep_strong=True)
    d = d[d.wind.notna() & d.wind_24.notna()].assign(ri=lambda x: (x.wind_24 - x.wind >= cyclone.RI_KT).astype(int))
    s = pd.read_parquet(DATA / "himawari_features.parquet")
    s["time"] = pd.to_datetime(s.time)
    d = d.merge(s, on=["storm", "time"])
    have = {p.stem for p in (DATA / "himawari_crops").glob("*.npy")}
    d = d[[hw.crop_name(a, b) in have for a, b in zip(d.storm, d.time)]]
    d = d[d[hw.NAMES].notna().all(axis=1)]
    a, b, c = split_years(sorted(d.storm_year.unique().tolist()))
    log(f"  Rows {len(d)}, train {a}, validation {b}, test {c}")
    rows = []
    for target, name in (("y", "Reach 64 kt"), ("wind", "24 h wind"), ("ri", "Rapid intensification")):
        part = d[d.wind < 64] if target == "y" else d
        tr, va, te = part[part.storm_year.isin(a)], part[part.storm_year.isin(b)], part[part.storm_year.isin(c)]
        tr, va, te = with_scores(tr, va, te, target)
        p, q, _ = compare(tr, va, te, hw.NAMES, target)
        base = cyclone.FEATURES
        cyclone.FEATURES = base + hw.NAMES
        try:
            _, r, win = compare(tr, va, te, ["cnn"], target)
        finally:
            cyclone.FEATURES = base
        show = (lambda v: f"{-v:.2f} kt") if target == "wind" else (lambda v: f"{v:.3f}")
        rows.append(
            {
                "target": name,
                "rows": len(tr) + len(va) + len(te),
                "plain_val": show(p[0]),
                "summary_val": show(q[0]),
                "cnn_val": show(r[0]),
                "plain_test": show(p[2]),
                "summary_test": show(q[2]),
                "cnn_test": show(r[2]),
                "kept": "Yes" if win else "No",
            }
        )
        log(f"  {rows[-1]}")
    out = {
        "table": rows,
        "years": {"train": a, "validation": b, "test": c},
        "note": "Plain model, then cloud summaries, then a small CNN score. The CNN is kept only if validation beats the summaries by more than 2 standard deviations over 3 seeds.",
    }
    (RESULTS / "exp_cnn.json").write_text(json.dumps(out, indent=1))
    log("CNN experiment finish")


if __name__ == "__main__":
    main()
