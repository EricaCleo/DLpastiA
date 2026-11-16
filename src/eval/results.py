"""Each method notebook saves its metrics to outputs/results/<method>__<mask>.json;
07_results.ipynb loads them all into one comparison table."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# RESULTS_DIR=outputs/results_final keeps the final runs apart from the earlier small runs
DIR = Path(os.environ["RESULTS_DIR"]) if os.environ.get("RESULTS_DIR") else ROOT / "outputs" / "results"
if not DIR.is_absolute():
    DIR = ROOT / DIR


def save_result(method, mask_type, res, extra=None):
    DIR.mkdir(parents=True, exist_ok=True)
    (DIR / f"{method}__{mask_type}.json").write_text(
        json.dumps({"method": method, "mask": mask_type, **res, **(extra or {})}, indent=2))


def load_all():
    return [json.loads(p.read_text()) for p in sorted(DIR.glob("*.json"))]


def table(mask_type=None):
    import pandas as pd
    df = pd.DataFrame(load_all())
    if mask_type and len(df):
        df = df[df["mask"] == mask_type]
    return df.set_index(["method", "mask"]) if len(df) else df
