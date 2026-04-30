import json
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    FULL_SE_PATH,
    ID_COLS,
    LABEL_META_COLS,
    TARGET_COL,
    TEST_PATH,
    TRAIN_END,
    TRAIN_PATH,
    VAL_END,
    VAL_PATH,
)


def load_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def load_dataset_bundle(dataset_name: str = "base") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if dataset_name == "base":
        return load_parquet(TRAIN_PATH), load_parquet(VAL_PATH), load_parquet(TEST_PATH)
    if dataset_name == "se":
        df = load_parquet(FULL_SE_PATH).sort_values(["date", "country"]).reset_index(drop=True)
        train_df = df[df["date"] <= pd.Timestamp(TRAIN_END, tz="UTC")].copy()
        val_df = df[(df["date"] > pd.Timestamp(TRAIN_END, tz="UTC")) & (df["date"] <= pd.Timestamp(VAL_END, tz="UTC"))].copy()
        test_df = df[df["date"] > pd.Timestamp(VAL_END, tz="UTC")].copy()
        return train_df, val_df, test_df
    raise ValueError(f"Unsupported dataset_name: {dataset_name}")


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if col not in LABEL_META_COLS]


def split_xy(df: pd.DataFrame, feature_columns: list[str]):
    x = df[feature_columns].copy()
    y = df[TARGET_COL].astype(int).copy()
    return x, y


def build_preprocessor(x: pd.DataFrame, model_name: str) -> ColumnTransformer:
    numeric_cols = [c for c in x.columns if c != "country" and c != "date"]
    categorical_cols = ["country"] if "country" in x.columns else []

    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if model_name == "logreg":
        numeric_steps.append(("scaler", StandardScaler()))

    categorical_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)

    transformers = []
    if numeric_cols:
        transformers.append(("num", Pipeline(numeric_steps), numeric_cols))
    if categorical_cols:
        transformers.append(
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", categorical_encoder)]), categorical_cols)
        )

    return ColumnTransformer(transformers=transformers, remainder="drop")


def drop_constant_feature_columns(df: pd.DataFrame, feature_columns: list[str]) -> list[str]:
    keep = []
    for col in feature_columns:
        if col in ID_COLS:
            keep.append(col)
            continue
        if df[col].nunique(dropna=False) > 1:
            keep.append(col)
    return keep


def make_model_input(df: pd.DataFrame, feature_columns: list[str], date_mode: str = "ordinal") -> pd.DataFrame:
    x = df[feature_columns].copy()
    if "date" in x.columns:
        if date_mode == "ordinal":
            x["date_ordinal"] = x["date"].astype("int64") // 10**9
        elif date_mode == "parts":
            x["year"] = x["date"].dt.year
            x["month"] = x["date"].dt.month
            x["dayofyear"] = x["date"].dt.dayofyear
        elif date_mode != "none":
            raise ValueError(f"Unsupported date_mode: {date_mode}")
        x = x.drop(columns=["date"])
    return x


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(payload: dict, path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
