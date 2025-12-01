"""Gradient boosting baseline for AML transaction detection.

Usage:
    python tabular_baseline.py --dataset dataset/HI-Small_Trans.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gradient boosting baseline")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("dataset/HI-Small_Trans.csv"),
        help="Path to the transactions CSV",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test split proportion",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--target-fpr",
        type=float,
        default=None,
        help="If provided, select the highest threshold with FPR <= target",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional path to dump metrics as JSON",
    )
    return parser.parse_args()


def load_transactions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(
        columns={
            "From Bank": "from_bank",
            "To Bank": "to_bank",
            "Amount Received": "amount_received",
            "Receiving Currency": "receiving_currency",
            "Amount Paid": "amount_paid",
            "Payment Currency": "payment_currency",
            "Payment Format": "payment_format",
            "Is Laundering": "is_laundering",
        }
    )
    if "Account" in df.columns:
        df = df.rename(columns={"Account": "from_account"})
    if "Account.1" in df.columns:
        df = df.rename(columns={"Account.1": "to_account"})
    return df


def engineer_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    work = df.copy()
    work["timestamp"] = pd.to_datetime(work["Timestamp"], errors="coerce")
    work["hour"] = work["timestamp"].dt.hour.fillna(-1).astype(int)
    work["dayofweek"] = work["timestamp"].dt.dayofweek.fillna(-1).astype(int)
    work["month"] = work["timestamp"].dt.month.fillna(-1).astype(int)
    work["is_weekend"] = (work["dayofweek"] >= 5).astype(int)

    work["same_bank"] = (work["from_bank"] == work["to_bank"]).astype(int)
    if "from_account" in work.columns and "to_account" in work.columns:
        work["same_account"] = (work["from_account"] == work["to_account"]).astype(int)
    else:
        work["same_account"] = 0

    work["amount_diff"] = work["amount_received"] - work["amount_paid"]
    work["amount_ratio"] = np.divide(
        work["amount_received"],
        work["amount_paid"],
        out=np.full(work["amount_received"].shape, np.nan, dtype=float),
        where=work["amount_paid"].abs() > 0,
    )
    work["amount_ratio"] = np.where(
        np.isfinite(work["amount_ratio"]), work["amount_ratio"], np.nan
    )
    work["is_round_amount"] = ((work["amount_paid"] % 100) == 0).astype(int)

    numeric_cols = [
        "amount_received",
        "amount_paid",
        "amount_diff",
        "amount_ratio",
        "hour",
        "dayofweek",
        "month",
        "is_weekend",
        "same_bank",
        "same_account",
        "is_round_amount",
    ]
    categorical_cols = [
        "from_bank",
        "to_bank",
        "receiving_currency",
        "payment_currency",
        "payment_format",
    ]

    feature_frame = work[numeric_cols + categorical_cols]
    return feature_frame, work["is_laundering"].astype(int)


def build_pipeline(
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> Pipeline:
    numeric_processor = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
        ]
    )
    categorical_processor = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "encode",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_processor, numeric_cols),
            ("cat", categorical_processor, categorical_cols),
        ]
    )
    categorical_indices = list(
        range(len(numeric_cols), len(numeric_cols) + len(categorical_cols))
    )
    model = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.1,
        max_depth=None,
        max_leaf_nodes=63,
        class_weight="balanced",
        random_state=42,
        categorical_features=categorical_indices,
    )
    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])


def threshold_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> Dict[str, Any]:
    preds = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    tnr = tn / (tn + fp) if tn + fp else 0.0
    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": (2 * precision * recall / (precision + recall)) if precision + recall else 0.0,
        "fpr": fpr,
        "tnr": tnr,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def select_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_fpr: float | None,
) -> Dict[str, Dict[str, Any]]:
    results: Dict[str, Dict[str, Any]] = {}
    results["default"] = threshold_metrics(y_true, y_prob, 0.5)

    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    thresholds_extended = np.append(thresholds, 1.0)
    denom = precision + recall
    f1_scores = np.divide(
        2 * precision * recall,
        denom,
        out=np.zeros_like(denom),
        where=denom > 0,
    )
    best_idx = int(f1_scores.argmax())
    best_threshold = float(thresholds_extended[best_idx])
    results["best_f1"] = threshold_metrics(y_true, y_prob, best_threshold)

    if target_fpr is not None:
        grid = np.linspace(0.0, 1.0, num=501)
        viable = []
        for cand in grid:
            metrics = threshold_metrics(y_true, y_prob, float(cand))
            if metrics["fpr"] <= target_fpr:
                viable.append(metrics)
        if viable:
            results["target_fpr"] = max(viable, key=lambda item: item["recall"])
    return results


def evaluate_model(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    target_fpr: float | None,
) -> Dict[str, Any]:
    y_prob = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_prob)
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall, precision)
    thresholds = select_thresholds(y_test.to_numpy(), y_prob, target_fpr)

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "thresholds": thresholds,
        "positive_rate": float(y_test.mean()),
    }


def main() -> None:
    args = parse_args()
    df = load_transactions(args.dataset)
    X, y = engineer_features(df)
    numeric_cols = [
        "amount_received",
        "amount_paid",
        "amount_diff",
        "amount_ratio",
        "hour",
        "dayofweek",
        "month",
        "is_weekend",
        "same_bank",
        "same_account",
        "is_round_amount",
    ]
    categorical_cols = [
        "from_bank",
        "to_bank",
        "receiving_currency",
        "payment_currency",
        "payment_format",
    ]

    low_cardinality = [
        col for col in categorical_cols if X[col].nunique(dropna=True) <= 255
    ]
    high_cardinality = [
        col for col in categorical_cols if col not in low_cardinality
    ]
    for col in high_cardinality:
        codes, _ = pd.factorize(X[col], sort=False)
        codes = codes.astype(np.float32)
        codes[codes < 0] = np.nan
        X[f"{col}_code"] = codes
        numeric_cols.append(f"{col}_code")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        stratify=y,
        random_state=args.random_state,
    )

    pipeline = build_pipeline(numeric_cols, low_cardinality)
    pipeline.fit(X_train, y_train)
    metrics = evaluate_model(pipeline, X_test, y_test, args.target_fpr)

    print("=== Gradient Boosting Baseline ===")
    print(f"ROC AUC: {metrics['roc_auc']:.4f}")
    print(f"PR AUC : {metrics['pr_auc']:.4f}")
    print(f"Test positive prevalence: {metrics['positive_rate']:.4%}")
    for name, detail in metrics["thresholds"].items():
        print(f"\nThreshold strategy: {name}")
        for key, value in detail.items():
            if key in {"tp", "fp", "fn", "tn"}:
                print(f"  {key.upper():<3}: {value}")
            else:
                print(f"  {key:<10}: {value:.4f}")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nMetrics saved to {args.report}")


if __name__ == "__main__":
    main()
