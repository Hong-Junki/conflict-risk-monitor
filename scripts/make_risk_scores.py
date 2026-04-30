import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import ARTIFACT_DIR, PROJECT_ROOT
from src.data_utils import load_dataset_bundle, make_model_input, split_xy


BASELINE_SCORE_PATH = PROJECT_ROOT / "processed" / "features" / "baseline_scores.parquet"
DEFAULT_CONFIG_PATH = ARTIFACT_DIR / "best_model_config__se.json"
DEFAULT_MODEL_PATH = ARTIFACT_DIR / "best_val_model__se.joblib"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True, help="Prediction csv with date,country,y_prob.")
    parser.add_argument("--out", default=None, help="Output csv path.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument(
        "--adjacency",
        default=None,
        help="Optional csv with country,neighbor columns for hotspot score.",
    )
    return parser.parse_args()


def logistic_0_100(z):
    z = np.clip(z, -6, 6)
    return 100.0 / (1.0 + np.exp(-z))


def rolling_z_score(frame, raw_col, out_col):
    pieces = []
    for _, g in frame.sort_values(["country", "date"]).groupby("country", sort=False):
        s = g[raw_col].astype(float)
        mean = s.shift(1).rolling(90, min_periods=14).mean()
        std = s.shift(1).rolling(90, min_periods=14).std().replace(0, np.nan)
        z = ((s - mean) / std).fillna(0.0)
        g[out_col] = logistic_0_100(z)
        pieces.append(g)
    return pd.concat(pieces, axis=0).sort_index()


def build_c_state(full_df):
    df = full_df[["date", "country"]].copy()

    df["U_raw"] = (
        full_df["acled_ratio_vac"].fillna(0) * 2.0
        + np.log1p(full_df["acled_fatalities_30d"].fillna(0))
        + np.log1p(full_df["acled_event_count_30d"].fillna(0)) * 0.25
    )
    df["C_raw"] = (
        full_df["acled_ratio_battles"].fillna(0)
        + full_df["acled_ratio_explosions"].fillna(0)
        + np.log1p(full_df["acled_event_count_14d"].fillna(0)) * 0.5
    )
    df["S_raw"] = (
        full_df["gdelt_quadclass_3_ratio"].fillna(0)
        + full_df["gdelt_quadclass_4_ratio"].fillna(0)
        + np.maximum(-full_df["gdelt_tone_mean_14d"].fillna(0), 0) * 0.1
    )
    df["I_raw"] = (
        np.log1p(full_df["gdelt_mentions_sum_14d"].fillna(0))
        + np.log1p(full_df["gdelt_event_count_14d"].fillna(0)) * 0.5
        + full_df["gdelt_goldstein_std_14d"].fillna(0) * 0.1
    )

    for raw_col, score_col in [
        ("U_raw", "U_score"),
        ("C_raw", "C_score"),
        ("S_raw", "S_score"),
        ("I_raw", "I_score"),
    ]:
        df = rolling_z_score(df, raw_col, score_col)

    df["C_state"] = df[["U_score", "C_score", "S_score", "I_score"]].mean(axis=1)
    return df[["date", "country", "U_score", "C_score", "S_score", "I_score", "C_state"]]


def validation_f_scale(config, model_path):
    train_df, val_df, _ = load_dataset_bundle(config["dataset"])
    pipeline = joblib.load(model_path)
    x_val, _ = split_xy(val_df, config["feature_columns"])
    x_val = make_model_input(x_val, config["feature_columns"], date_mode=config.get("date_mode", "ordinal"))
    val_prob = pipeline.predict_proba(x_val)[:, 1]
    val_p99 = float(np.quantile(val_prob, 0.99))
    return max(val_p99, 1e-8)


def build_hotspot_score(scored, adjacency_path):
    if adjacency_path is None:
        scored["hotspot_score"] = 0.0
        return scored

    adjacency = pd.read_csv(adjacency_path)
    required = {"country", "neighbor"}
    missing = required - set(adjacency.columns)
    if missing:
        raise ValueError(f"Adjacency file is missing columns: {sorted(missing)}")

    c_lookup = scored[["date", "country", "C_score"]].rename(
        columns={"country": "neighbor", "C_score": "neighbor_C_score"}
    )
    hotspot = scored[["date", "country"]].merge(adjacency, on="country", how="left")
    hotspot = hotspot.merge(c_lookup, on=["date", "neighbor"], how="left")
    hotspot["is_hot_neighbor"] = (hotspot["neighbor_C_score"] >= 50).astype(int)
    hotspot = hotspot.groupby(["date", "country"], as_index=False)["is_hot_neighbor"].sum()
    hotspot["hotspot_score"] = np.minimum(hotspot["is_hot_neighbor"] * 1.5, 5.0)

    return scored.merge(hotspot[["date", "country", "hotspot_score"]], on=["date", "country"], how="left").fillna(
        {"hotspot_score": 0.0}
    )


def main():
    args = parse_args()
    pred = pd.read_csv(args.pred)
    pred["date"] = pd.to_datetime(pred["date"], utc=True)

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    train_df, val_df, test_df = load_dataset_bundle(config["dataset"])
    full_df = pd.concat([train_df, val_df, test_df], axis=0, ignore_index=True).sort_values(["country", "date"])

    c_state = build_c_state(full_df)
    baseline = pd.read_parquet(BASELINE_SCORE_PATH)[["iso3", "B_score"]].rename(columns={"iso3": "country"})
    f_scale = validation_f_scale(config, args.model)

    scored = pred.merge(test_df[["date", "country"]], on=["date", "country"], how="inner", validate="one_to_one")
    scored = scored.merge(baseline, on="country", how="left", validate="many_to_one")
    scored = scored.merge(c_state, on=["date", "country"], how="left", validate="one_to_one")
    scored = build_hotspot_score(scored, args.adjacency)

    scored["F_score"] = np.minimum(scored["y_prob"] / f_scale * 100.0, 100.0)
    scored["risk_score"] = (
        0.2 * scored["B_score"]
        + 0.4 * scored["C_state"]
        + 0.4 * scored["F_score"]
        + scored["hotspot_score"]
    ).clip(0, 100)

    out = args.out
    if out is None:
        pred_path = Path(args.pred)
        out = pred_path.with_name(pred_path.stem.replace("predictions", "risk_scores") + ".csv")

    cols = [
        "date",
        "country",
        "y_prob",
        "risk_score",
        "B_score",
        "C_state",
        "F_score",
        "hotspot_score",
        "U_score",
        "C_score",
        "S_score",
        "I_score",
    ]
    scored["date"] = scored["date"].dt.strftime("%Y-%m-%d")
    scored[cols].to_csv(out, index=False, encoding="utf-8")

    print(f"saved: {out}")
    print(f"rows: {len(scored)}")
    print(f"validation_p99_for_F_score: {f_scale:.6f}")
    print(f"risk_score range: {scored['risk_score'].min():.3f} -> {scored['risk_score'].max():.3f}")
    print(scored[cols].sort_values("risk_score", ascending=False).head(10).to_string(index=False))


if __name__ == "__main__":
    main()
