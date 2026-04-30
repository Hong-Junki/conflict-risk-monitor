import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import ARTIFACT_DIR, SEED, SUBMISSION_DIR
from src.data_utils import build_preprocessor, ensure_dir, load_dataset_bundle, make_model_input, split_xy


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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="solo_run", help="Used in the output file name.")
    parser.add_argument("--dataset", choices=["base", "se"], default="base")
    parser.add_argument(
        "--config-path",
        default=None,
        help="Path to the best model config json.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_dir(SUBMISSION_DIR)

    config_path = args.config_path or str(ARTIFACT_DIR / f"best_model_config__{args.dataset}.json")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    model_name = config["selected_model"]
    date_mode = config.get("date_mode", "ordinal")
    feature_columns = config["feature_columns"]

    train_df, val_df, test_df = load_dataset_bundle(config["dataset"])

    train_val_df = pd.concat([train_df, val_df], axis=0, ignore_index=True)

    x_train, y_train = split_xy(train_val_df, feature_columns)
    x_test, _ = split_xy(test_df, feature_columns)

    x_train = make_model_input(x_train, feature_columns, date_mode=date_mode)
    x_test = make_model_input(x_test, feature_columns, date_mode=date_mode)

    preprocessor = build_preprocessor(x_train, model_name=model_name)
    estimator = build_model(model_name)
    pipeline = Pipeline([("prep", preprocessor), ("model", estimator)])
    pipeline.fit(x_train, y_train)

    y_prob = pipeline.predict_proba(x_test)[:, 1]

    submission = test_df[["date", "country"]].copy()
    submission["date"] = submission["date"].dt.strftime("%Y-%m-%d")
    submission["y_prob"] = y_prob

    output_path = SUBMISSION_DIR / f"predictions__{model_name}__{args.run_name}.csv"
    submission.to_csv(output_path, index=False, encoding="utf-8")

    print(f"saved: {output_path}")
    print(submission.head(10).to_string(index=False))
    print(f"rows: {len(submission)}")
    print(f"y_prob range: {submission['y_prob'].min():.6f} -> {submission['y_prob'].max():.6f}")


if __name__ == "__main__":
    main()
