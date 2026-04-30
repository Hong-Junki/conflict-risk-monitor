import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import TEST_PATH


GUIDE_BASELINE = {
    "model": "LightGBM + SE",
    "pr_auc": 0.1307,
    "p_at_5pct": 0.190,
    "r_at_p_ge_010": 0.558,
}


def precision_at_top_fraction(y_true, y_prob, frac):
    k = int(np.ceil(len(y_true) * frac))
    top_idx = np.argsort(-np.asarray(y_prob))[:k]
    return float(np.asarray(y_true)[top_idx].mean())


def recall_at_precision(y_true, y_prob, threshold):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    mask = precision >= threshold
    return float(recall[mask].max()) if mask.any() else 0.0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True, help="Submission csv with date,country,y_prob.")
    return parser.parse_args()


def main():
    args = parse_args()
    pred = pd.read_csv(args.pred)
    test = pd.read_parquet(TEST_PATH)[["date", "country", "y_escalation"]].copy()
    test["date"] = test["date"].dt.strftime("%Y-%m-%d")

    merged = test.merge(pred, on=["date", "country"], how="inner", validate="one_to_one")
    if len(merged) != len(test):
        raise ValueError(f"Submission row mismatch: expected {len(test)}, matched {len(merged)}")

    y_true = merged["y_escalation"].to_numpy(dtype=int)
    y_prob = merged["y_prob"].to_numpy(dtype=float)

    metrics = {
        "rows": len(merged),
        "positive_rate": float(y_true.mean()),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "p_at_5pct": precision_at_top_fraction(y_true, y_prob, 0.05),
        "r_at_p_ge_010": recall_at_precision(y_true, y_prob, 0.10),
    }

    print("submission_metrics")
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}" if isinstance(value, float) else f"{key}: {value}")

    print("\nvs_guide_baseline")
    print(f"baseline_model: {GUIDE_BASELINE['model']}")
    for key in ["pr_auc", "p_at_5pct", "r_at_p_ge_010"]:
        delta = metrics[key] - GUIDE_BASELINE[key]
        print(f"{key}: ours={metrics[key]:.6f}, baseline={GUIDE_BASELINE[key]:.6f}, delta={delta:.6f}")


if __name__ == "__main__":
    main()
