# Tarang

Tarang scores flood, earthquake and cyclone risk across South and East Asia from public data. It shows the results on a website and sends them through a Telegram bot.

- Website: https://tarang-sepia.vercel.app
- Starting proposal: `Multi_Hazard_ML_Project_Proposal.pdf`

Tarang raises awareness and does not replace official warnings. Earthquake output is a risk level for a region and a 30-day window, never a predicted time.

## Results on held-out test data

PR-AUC runs from 0 to 1, and higher is better. It measures how well a model ranks real events above quiet periods. The last column shows what random guessing scores.

| Hazard | Question | Test period | Best model | PR-AUC | Baseline | Random |
| --- | --- | --- | --- | --- | --- | --- |
| Cyclone | Reaches 64 knots within 24 hours? | 2022 onward | Stacking | 0.744 | 0.458 | 0.113 |
| Flood | Surge in flood reports in the next 3 days? | 2024 to January 2026 | Stacking | 0.380 | 0.200 | 0.066 |
| Earthquake | Magnitude 5.0 or above in a 1 degree cell in the next 30 days? | 2020 onward | XGBoost blended with ETAS | 0.138 | 0.129 | 0.014 |

The 24-hour cyclone wind forecast has a mean error of 10.8 knots, against 14.8 knots for assuming no change. Its 80% range holds 79.6% of outcomes. Exact figures are in `results/`.

These are test scores, not live performance. Every live forecast is logged in `data/forecasts.sqlite` and scored once its window closes.

## Project layout

| Path | Contents |
| --- | --- |
| `src/hazards/` | Python package: models, live scoring, bot and data readers |
| `scripts/retrain.py` | Retrain one hazard and keep the result only if validation improves |
| `scripts/export_dashboard.py` | Refresh every data file the website reads |
| `scripts/data/` | Download and prepare data |
| `scripts/experiments/` | Experiments behind each model decision, and the Models page tables |
| `tests/` | pytest tests, including leakage checks |
| `results/` | Scores and experiment results as JSON |
| `dashboard/` | Website, built with React, TypeScript and MapLibre |

Main modules in `src/hazards/`:

| Module | Purpose |
| --- | --- |
| `common.py` | Shared evaluation, model comparison and risk cut-offs |
| `cyclone.py`, `flood.py`, `earthquake.py` | One model pipeline per hazard |
| `typhoon.py` | Older JMA-only model, and the storm table the cyclone model uses |
| `etas.py` | ETAS earthquake model, the serious baseline |
| `recal.py` | Rolling recalibration |
| `lab.py` | Regularisation, tuning and class imbalance experiments |
| `live.py` | Saved models, daily scoring and outcome scoring |
| `refresh.py` | News, events, source checks and website data |
| `bot.py` | Telegram bot and alerts |
| `events.py` | Replays of real past events |
| `era5.py`, `himawari.py`, `satellite.py`, `basins.py` | External data readers |

## Setup

Full steps for a new computer are in [SETUP.md](SETUP.md). In short, with Python 3.12 or newer and Node.js:

```
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"
cd dashboard
npm install
```

Raw data, saved models and logs are not in the repository because of their size. `data/` is rebuilt by the scripts. API keys and the bot token go in `.env`, which is never committed.

## Commands

Run from the project root.

```
.venv/Scripts/python -m pytest -q                      # tests
.venv/Scripts/python -m hazards.cyclone                # train one hazard; also hazards.flood, hazards.earthquake
.venv/Scripts/python -m hazards.lab cyclone flood earthquake
.venv/Scripts/python scripts/retrain.py flood
.venv/Scripts/python -m hazards.refresh --loop 5       # keep live data updated every 5 minutes
.venv/Scripts/python -m hazards.bot                    # Telegram bot; needs TELEGRAM_BOT_TOKEN in .env
```

Website, from `dashboard/`:

```
npm run dev
npm run build
```

## Validation rules

- No random splits: older years train, later years validate and test.
- All records of one storm stay in one set.
- Cross-validation folds are time-ordered.
- SMOTE touches training folds only.
- Stacking uses out-of-fold predictions.
- Models, features and alert thresholds are chosen on validation data. Test data is only reported.
- A new data source is kept only if it beats the same model without it by more than the noise between runs.

## Data sources

JMA best track and storm feed, IBTrACS, USGS, Groundsource, GDACS, Open-Meteo, NASA IMERG and SMAP, Himawari on NOAA open data, ERA5 and GeoNames (CC BY 4.0).
