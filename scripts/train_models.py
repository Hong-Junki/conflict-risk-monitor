import json
import sys
from pathlib import Path
import argparse

sys.path.append(str(Path(__file__).resolve().parents[1]))

import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.config import ARTIFACT_DIR, SEED
from src.data_utils import (
    build_preprocessor,
    drop_constant_feature_columns,
    ensure_dir,
    get_feature_columns,
    load_dataset_bundle,
    make_model_input,
    save_json,
    split_xy,
)
from src.metrics import evaluate_predictions


def build_model(model_name: str):
    if model_name == "logreg":
        return LogisticRegression(
            random_state=SEED,
            max_iter=2000,
            class_weight="balanced",
            solver="liblinear",
        )
    if model_name == "lightgbm":
        return LGBMClassifier(
            random_state=SEED,
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            class_weight="balanced",
            n_jobs=-1,
            verbosity=-1,
        )
    if model_name == "xgboost":
        return XGBClassifier(
            random_state=SEED,
            n_estimators=500,
            learning_rate=0.04,
            max_depth=4,
            min_child_weight=8,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=3.0,
            eval_metric="aucpr",
            tree_method="hist",
            n_jobs=-1,
        )
    raise ValueError(f"Unsupported model: {model_name}")


def fit_and_score(model_name, train_df, val_df, feature_columns, date_mode):
    x_train, y_train = split_xy(train_df, feature_columns)
    x_val, _ = split_xy(val_df, feature_columns)

    x_train = make_model_input(x_train, feature_columns, date_mode=date_mode)
    x_val = make_model_input(x_val, feature_columns, date_mode=date_mode)

    preprocessor = build_preprocessor(x_train, model_name=model_name)
    estimator = build_model(model_name)
    pipeline = Pipeline([("prep", preprocessor), ("model", estimator)])
    pipeline.fit(x_train, y_train)

    val_prob = pipeline.predict_proba(x_val)[:, 1]
    metrics = evaluate_predictions(val_df, val_prob, history_df=train_df)
    metrics["model_name"] = model_name
    return pipeline, metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["base", "se"], default="base")
    parser.add_argument("--date-mode", choices=["ordinal", "parts", "none"], default="ordinal")
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_dir(ARTIFACT_DIR)

    train_df, val_df, _ = load_dataset_bundle(args.dataset)

    feature_columns = get_feature_columns(train_df)
    feature_columns = drop_constant_feature_columns(train_df, feature_columns)

    results = []
    best_pipeline = None
    best_metrics = None

    for model_name in ["logreg", "lightgbm", "xgboost"]:
        pipeline, metrics = fit_and_score(model_name, train_df, val_df, feature_columns, args.date_mode)
        metrics["dataset"] = args.dataset
        metrics["date_mode"] = args.date_mode
        results.append(metrics)
        if best_metrics is None or metrics["pr_auc"] > best_metrics["pr_auc"]:
            best_metrics = metrics
            best_pipeline = pipeline

    result_df = pd.DataFrame(results).sort_values("pr_auc", ascending=False).reset_index(drop=True)
    result_path = ARTIFACT_DIR / f"model_comparison__{args.dataset}.csv"
    result_df.to_csv(result_path, index=False, encoding="utf-8")

    best_path = ARTIFACT_DIR / f"best_val_model__{args.dataset}.joblib"
    joblib.dump(best_pipeline, best_path)

    config_payload = {
        "dataset": args.dataset,
        "date_mode": args.date_mode,
        "selected_model": best_metrics["model_name"],
        "selection_metric": "pr_auc",
        "feature_columns": feature_columns,
        "validation_metrics": best_metrics,
        "artifacts": {
            "comparison_csv": str(result_path),
            "best_model_joblib": str(best_path),
        },
    }
    save_json(config_payload, ARTIFACT_DIR / f"best_model_config__{args.dataset}.json")

    print(result_df.to_string(index=False))
    print()
    print(json.dumps(config_payload, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
