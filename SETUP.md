# Run Tarang on another computer

These steps are for Windows PowerShell. On macOS or Linux, use `.venv/bin/python` wherever you see `.venv\Scripts\python`.

## What you get at each stage

| After step | What works |
| --- | --- |
| 4 | Tests |
| 5 | Website on your computer, using the data saved in the repository |
| 6 and 7 | Telegram bot |
| 8 | Live updates and daily scoring |

You can stop at any stage. Showing the code, tests and website needs only steps 1 to 5.

## 1. Install the tools

Install these, keeping the default options:

- [Git](https://git-scm.com/downloads)
- [Python 3.12 or newer](https://www.python.org/downloads/). Tick "Add python.exe to PATH" in the installer.
- [Node.js, LTS version](https://nodejs.org/)

Close and reopen PowerShell, then check all three:

```powershell
git --version
python --version
node --version
```

## 2. Get access to the repository

The repository is private. The GitHub account you use must be `harsha-516` or an account added under Settings, Collaborators. If Git asks you to sign in during step 3, use the browser window it opens.

## 3. Clone the repository

```powershell
git clone https://github.com/harsha-516/ML-SEM5.git
cd ML-SEM5
```

Run all later commands from this `ML-SEM5` folder unless a step says otherwise.

## 4. Install Python packages and run the tests

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -e ".[dev]"
.venv\Scripts\python -m pytest -q
```

The install downloads a few GB, mostly PyTorch, and can take several minutes. The tests should end with all passed. They need no API keys and no `data` folder.

## 5. Run the website

```powershell
cd dashboard
npm ci
npm run dev
```

Open the address it prints, normally http://localhost:5173. Press Ctrl+C to stop it, then return to the main folder:

```powershell
cd ..
```

## 6. Add the private settings file

Keys are never stored in GitHub. Copy `.env` from the original computer into the `ML-SEM5` folder, using a pen drive or another private route. Do not send it through chat or email, and do not commit it.

If you are starting without that file, create `.env` with these lines:

```
TELEGRAM_BOT_TOKEN=
EARTHDATA_TOKEN=
CDSAPI_KEY=
YOUTUBE_API_KEY=
```

| Key | Needed for | Where to get it |
| --- | --- | --- |
| `TELEGRAM_BOT_TOKEN` | Telegram bot | Telegram, @BotFather |
| `EARTHDATA_TOKEN` | NASA IMERG and SMAP | NASA Earthdata profile |
| `CDSAPI_KEY` | ERA5 downloads | Copernicus Climate Data Store profile |
| `YOUTUBE_API_KEY` | Website videos | Google Cloud console |

## 7. Run the Telegram bot

Run the bot on one computer at a time. Stop it on the original computer first, because Telegram allows only one copy to read a bot's messages.

```powershell
.venv\Scripts\python -m hazards.bot
```

Leave this window open. In Telegram, send `/help` to the bot.

The bot reads `results/` and the website data files, and those are only as fresh as the last push unless step 8 is running.

## 8. Copy the data folder and run live updates

Saved models, raw data and the forecast log are not in GitHub because of their size. Copy the whole `data` folder (about 1.4 GB) from the original computer into `ML-SEM5`. It also holds `bot.sqlite`, which is the list of subscribers.

Then open a second PowerShell window in `ML-SEM5` and run:

```powershell
.venv\Scripts\python -m hazards.refresh --loop 5
```

This refreshes data every 5 minutes. The computer must stay awake. Run it on one computer only, so that all forecasts land in one log.

The hosted website changes only when this data is pushed to GitHub.

## 9. Push changes to GitHub

Do this once on each new computer:

```powershell
git config user.name "harsha-516"
git config user.email "<your GitHub noreply email>"
```

The noreply email is under GitHub Settings, Emails. Then, for each change:

```powershell
git pull
.venv\Scripts\python -m pytest -q
git add -A
git commit -m "Describe the change"
git push
```

A push to `main` updates the Vercel website automatically. `.env` and `data/` are ignored, so `git add -A` will not include them.

## Common problems

| Problem | Fix |
| --- | --- |
| `python` is not recognized | Reinstall Python with "Add python.exe to PATH" ticked, then reopen PowerShell |
| `repository not found` on clone | The signed-in GitHub account has no access; see step 2 |
| `No module named hazards` | Run step 4 from the `ML-SEM5` folder |
| Bot says `TELEGRAM_BOT_TOKEN` is missing | Put `.env` in the `ML-SEM5` folder; see step 6 |
| Bot stops with a Telegram conflict | Another copy is running; stop it |
| Refresh fails on a missing model or data file | Copy the `data` folder; see step 8 |
| Commit is rejected for a missing name or email | Run the two `git config` commands in step 9 |
