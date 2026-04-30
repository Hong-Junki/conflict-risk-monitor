import math

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

from src.config import TARGET_COL


def pr_auc(y_true, y_prob) -> float:
    return float(average_precision_score(y_true, y_prob))


def precision_at_top_fraction(y_true, y_prob, frac: float) -> float:
    n = len(y_true)
    k = max(1, int(math.ceil(n * frac)))
    order = np.argsort(-np.asarray(y_prob))
    top_idx = order[:k]
    return float(np.asarray(y_true)[top_idx].mean())


def recall_at_precision_threshold(y_true, y_prob, threshold: float) -> float:
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    mask = precision >= threshold
    if not np.any(mask):
        return 0.0
    return float(np.max(recall[mask]))


def expected_calibration_error(y_true, y_prob, bins: int = 10) -> float:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        if right == 1.0:
            mask = (y_prob >= left) & (y_prob <= right)
        else:
            mask = (y_prob >= left) & (y_prob < right)
        if not np.any(mask):
            continue
        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += np.abs(acc - conf) * mask.mean()
    return float(ece)


def persistence_baseline(eval_df: pd.DataFrame, history_df: pd.DataFrame) -> np.ndarray:
    cols = ["date", "country", TARGET_COL]
    joined = pd.concat([history_df[cols], eval_df[cols]], axis=0, ignore_index=True)
    joined = joined.sort_values(["country", "date"]).reset_index(drop=True)
    joined["persistence_pred"] = joined.groupby("country")[TARGET_COL].shift(1).fillna(0)

    eval_keys = set(zip(eval_df["date"], eval_df["country"]))
    preds = joined[joined.apply(lambda row: (row["date"], row["country"]) in eval_keys, axis=1)].copy()
    preds = preds.sort_values(["date", "country"]).reset_index(drop=True)

    eval_sorted = eval_df.sort_values(["date", "country"]).reset_index(drop=True)
    merged = eval_sorted.merge(
        preds[["date", "country", "persistence_pred"]],
        on=["date", "country"],
        how="left",
        validate="one_to_one",
    )
    return merged["persistence_pred"].to_numpy(dtype=float)


def evaluate_predictions(eval_df: pd.DataFrame, y_prob, history_df: pd.DataFrame) -> dict:
    eval_sorted = eval_df.sort_values(["date", "country"]).reset_index(drop=True)
    y_true = eval_sorted[TARGET_COL].to_numpy(dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    persistence_prob = persistence_baseline(eval_sorted, history_df)
    persistence_score = pr_auc(y_true, persistence_prob)
    model_score = pr_auc(y_true, y_prob)

    return {
        "pr_auc": model_score,
        "persistence_pr_auc": persistence_score,
        "persistence_gain": model_score - persistence_score,
        "p_at_1pct": precision_at_top_fraction(y_true, y_prob, 0.01),
        "p_at_5pct": precision_at_top_fraction(y_true, y_prob, 0.05),
        "p_at_10pct": precision_at_top_fraction(y_true, y_prob, 0.10),
        "r_at_p_10": recall_at_precision_threshold(y_true, y_prob, 0.10),
        "r_at_p_20": recall_at_precision_threshold(y_true, y_prob, 0.20),
        "r_at_p_30": recall_at_precision_threshold(y_true, y_prob, 0.30),
        "ece_10bin": expected_calibration_error(y_true, y_prob, bins=10),
        "positive_rate": float(y_true.mean()),
    }
