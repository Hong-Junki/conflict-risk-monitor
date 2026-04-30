import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import FULL_PATH, TARGET_COL
from src.data_utils import get_feature_columns, load_dataset_bundle, load_parquet


def summarize(name, df):
    feature_columns = get_feature_columns(df)
    print(f"[{name}]")
    print(f"shape: {df.shape}")
    print(f"date range: {df['date'].min()} -> {df['date'].max()}")
    print(f"countries: {df['country'].nunique()}")
    print(f"{TARGET_COL} positive rate: {df[TARGET_COL].mean():.4f}")
    print(f"feature count before filtering: {len(feature_columns)}")
    print()


def main():
    full_df = load_parquet(FULL_PATH)
    train_df, val_df, test_df = load_dataset_bundle("base")
    train_se_df, val_se_df, test_se_df = load_dataset_bundle("se")

    summarize("full", full_df)
    summarize("train", train_df)
    summarize("val", val_df)
    summarize("test", test_df)
    summarize("train_se", train_se_df)
    summarize("val_se", val_se_df)
    summarize("test_se", test_se_df)

    country_rate = (
        full_df.groupby("country")[TARGET_COL]
        .agg(["mean", "count"])
        .sort_values("mean", ascending=False)
        .head(15)
        .reset_index()
    )
    print("[top countries by escalation rate]")
    print(country_rate.to_string(index=False))
    print()

    payload = {
        "full_shape": list(full_df.shape),
        "feature_columns": get_feature_columns(full_df),
        "top_country_rates": country_rate.to_dict(orient="records"),
    }
    print("[json summary]")
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
