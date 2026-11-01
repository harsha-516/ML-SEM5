# Morning report, 2026-11-01

All plan tasks ran. Flood improved most. Cyclone and earthquake scores did not move. The risk maps now update by themselves. ERA5 and Himawari were tested but are not in live forecasts. Details are in `docs/overnight-log.md`. Nothing was committed, pushed or hosted.

## Before and after, test set

| Measure | Start of night | Now | Baseline |
|---|---|---|---|
| Flood PR-AUC, Stacking | 0.279 | 0.385 | 0.200 |
| Flood replays warned | 0 of 4 | 2 of 4 | |
| Cyclone reach 64 kt PR-AUC, Stacking | 0.743 | 0.744 | 0.458 |
| Cyclone 24 h wind MAE | | 10.8 kt | 14.8 kt persistence |
| Cyclone 80% range coverage | 71.1% | 77.8% | |
| Cyclone rapid intensification PR-AUC | | 0.383 | 7.6% base rate |
| Earthquake PR-AUC, Stacking | 0.135 | 0.132 | 0.129 fitted ETAS |

The flood test is now 44,460 rows using forecasts as issued. Actual later rain would give 0.428, but it cannot exist live. The earthquake baseline is new: the old rule scored 0.125, and fitted ETAS is a harder target.

## What changed

- Flood: real archived forecasts, ratio features and IMERG Late rainfall. Validation 0.373.
- Earthquake: ETAS fitted by maximum likelihood and pyCSEP tests.
- Cyclone: tuned settings and a conformal margin of 1.7 kt on the range.
- Live: saved models in `data/models/`. Earthquake and flood rescore daily. Active storms score each refresh, with IBTrACS as a second source for the North Indian basin. Outcomes go to `data/forecasts.sqlite`, and `scripts/retrain.py` is a manual promote-or-rollback script.
- Site: experiment tables, live scorecard, active storm panel, last scored times, figures read from results.

## Experiments

| Experiment | Validation without, with | Test without, with | Result |
|---|---|---|---|
| IMERG, flood | 0.359, 0.365 | 0.373, 0.381 | In the model |
| SMAP, flood | 0.359, 0.347 | 0.373, 0.366 | Rejected |
| ERA5, reach 64 kt | 0.627, 0.675 | 0.695, 0.722 | Passed, not live |
| ERA5, 24 h wind | 11.08, 10.71 kt | 11.82, 11.71 kt | Passed, not live |
| ERA5, rapid intensification | 0.254, 0.312 | 0.418, 0.388 | Passed validation, worse on test, not live |
| Himawari, reach 64 kt | 0.614, 0.645 | 0.605, 0.677 | Below the noise rule |
| Himawari, 24 h wind | 11.79, 11.03 kt | 15.39, 15.00 kt | Passed, not in production |
| Himawari, rapid intensification | 0.187, 0.223 | 0.183, 0.222 | Below the noise rule |

## What failed or fell short

- Corrected later on 2026-11-01: this report first said the ML model fails the pyCSEP number test and ETAS passes. That test was unfair to yes or no windows. Counted as a sum of chances, ML is well calibrated and ETAS under-forecasts, 2,725 against 2,845 observed. See the overnight log, improvement step 1.
- 2 of 4 flood replays still give no warning, and Nilambur peaked at only 7.6%.
- Cyclone range coverage is 77.8%, short of 80%.
- ERA5 reached 2022 to 2026, not 2000, because each file takes about an hour.
- Himawari reached August 2024, not 2022, before the 3 hour limit.
- MODIS label check was inconclusive: 27% of 26 clear labelled surges against 32% of 57 controls. GDACS agreed more often, 26.5% against 14.8%, with regional bias.

## Unverified

- ERA5 and Himawari results rest on one test season and few rows. Himawari trained on a single season.
- Himawari cloud was read at best-track positions, corrected after each storm. Live positions would be rougher, so that gain is not proven real-time.
- The live scorecard has no outcomes yet: 0 of 5,183 forecasts have closed and been scored, with 2 storm forecasts logged.
- First flood outcomes use GDACS, not the training label.
- Murasu had no track at the last check. It appears once feeds carry it and `hazards.cyclone` reruns. Its North Indian path has been tested only with made-up data.

## Decisions, settled after the run

1. Himawari continues back to 2022, with an 8 hour limit. Rerun `scripts/cyclone_experiments.py himawari` when it stops.
2. No live ERA5 substitute yet. First rerun on the longer ERA5 record; the current test is one season and rapid intensification got worse.
3. Superseded by improvement step 1: a blend of ML and ETAS is now the earthquake model, test PR-AUC 0.138.

## State left running

- Refresh loop, 14 of 14 sources on its last cycle.
- Preview at http://localhost:4173.
- `scripts/fetch_era5.py 2000`, still fetching older years. Stop it any time; it resumes from cached files.

55 tests pass. `npx tsc -b`, `npm run build` and 20 page checks at 1440 and 390 are clean.
