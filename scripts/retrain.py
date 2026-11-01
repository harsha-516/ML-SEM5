"""Retrain one hazard, compare with the current version on validation, and promote only if better.

Usage: python scripts/retrain.py cyclone|flood|earthquake
Run by hand. A worse or equal model is rolled back, including its saved model file.
"""

import json
import shutil
import sys
import tempfile
from pathlib import Path

from hazards import cyclone, earthquake, flood
from hazards.common import RESULTS, log
from hazards.live import MODELS

RUN = {"cyclone": cyclone.run, "flood": flood.run, "earthquake": earthquake.run}


def best(path: Path) -> float:
    """Score of the winning model in a results file."""
    r = json.loads(path.read_text())
    return r["models"][r["winner"]]["validation"]["pr_auc"]


def main(hazard: str) -> None:
    """Retrain one hazard and keep the old model if the new one is worse."""
    result, model = RESULTS / f"{hazard}.json", MODELS / f"{hazard}.joblib"
    keep = Path(tempfile.mkdtemp())
    shutil.copy(result, keep / result.name)
    shutil.copy(model, keep / model.name)
    old = best(result)
    log(f"Retrain {hazard} start: current validation PR-AUC {old:.4f}")
    RUN[hazard]()
    new = best(result)
    if new > old:
        log(f"Retrain {hazard} promoted: {old:.4f} to {new:.4f}")
    else:
        shutil.copy(keep / result.name, result)
        shutil.copy(keep / model.name, model)
        log(f"Retrain {hazard} rolled back: {new:.4f} does not beat {old:.4f}")
    shutil.rmtree(keep)


if __name__ == "__main__":
    main(sys.argv[1])
