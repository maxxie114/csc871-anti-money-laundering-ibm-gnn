"""PyTorch MLP baseline for AML transaction detection."""
from __future__ import annotations

import argparse
import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tabular MLP baseline")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("dataset/HI-Small_Trans.csv"),
        help="Path to the transactions CSV",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8192,
        help="Mini-batch size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Adam learning rate",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-5,
        help="Adam weight decay",
    )
    parser.add_argument(
        "--hidden-dims",
        type=str,
        default="256,128,64",
        help="Comma-separated hidden layer sizes",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout probability inside the MLP",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test split proportion",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.1,
        help="Validation split proportion (of the whole dataset)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--base-embed-dim",
        type=int,
        default=8,
        help="Base embedding dimension scaling factor",
    )
    parser.add_argument(
        "--max-embed-dim",
        type=int,
        default=64,
        help="Maximum embedding dimension",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Training device (auto|cpu|cuda|mps)",
    )
    parser.add_argument(
        "--target-fpr",
        type=float,
        default=None,
        help="If provided, report highest-recall threshold with FPR <= target",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional JSON output path",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on number of rows for quick experiments",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker count",
    )
    return parser.parse_args()


class TabularDataset(Dataset):
    def __init__(
        self,
        numeric: np.ndarray,
        categorical: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        self.numeric = torch.tensor(numeric, dtype=torch.float32)
        self.categorical = torch.tensor(categorical, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return self.labels.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.numeric[idx], self.categorical[idx], self.labels[idx]


class TabularMLP(nn.Module):
    def __init__(
        self,
        num_numeric: int,
        categorical_cardinalities: List[int],
        hidden_dims: List[int],
        dropout: float,
        base_embed_dim: int,
        max_embed_dim: int,
    ) -> None:
        super().__init__()
        embeddings: List[nn.Module] = []
        embed_total = 0
        for cardinality in categorical_cardinalities:
            if cardinality <= 1:
                embeddings.append(None)  # type: ignore[arg-type]
                continue
            dim = min(max_embed_dim, max(4, int(math.ceil(cardinality ** 0.25 * base_embed_dim))))
            emb = nn.Embedding(cardinality, dim, padding_idx=0)
            nn.init.xavier_uniform_(emb.weight)
            embeddings.append(emb)
            embed_total += dim
        self.embeddings = nn.ModuleList([emb for emb in embeddings if emb is not None])
        self.embedding_indices = [idx for idx, emb in enumerate(embeddings) if emb is not None]

        input_dim = num_numeric + embed_total
        layers: List[nn.Module] = []
        prev_dim = input_dim
        for width in hidden_dims:
            layers.append(nn.Linear(prev_dim, width))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(width))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = width
        self.mlp = nn.Sequential(*layers) if layers else nn.Identity()
        self.head = nn.Linear(prev_dim, 1)

    def forward(self, numeric: torch.Tensor, categorical: torch.Tensor) -> torch.Tensor:
        pieces: List[torch.Tensor] = []
        if numeric.shape[1] > 0:
            pieces.append(numeric)
        if self.embeddings:
            embeds: List[torch.Tensor] = []
            for module, idx in zip(self.embeddings, self.embedding_indices):
                embeds.append(module(categorical[:, idx]))
            pieces.append(torch.cat(embeds, dim=1))
        features = pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=1)
        hidden = self.mlp(features) if isinstance(self.mlp, nn.Sequential) else self.mlp(features)
        return self.head(hidden).squeeze(dim=1)


def load_transactions(path: Path, max_samples: int | None) -> pd.DataFrame:
    df = pd.read_csv(path)
    if max_samples is not None and len(df) > max_samples:
        df = df.sample(max_samples, random_state=42).reset_index(drop=True)
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
        out=np.full(work.shape[0], np.nan, dtype=float),
        where=work["amount_paid"].abs() > 0,
    )
    work["amount_ratio"] = np.where(np.isfinite(work["amount_ratio"]), work["amount_ratio"], np.nan)
    work["is_round_amount"] = ((work["amount_paid"] % 100) == 0).astype(int)

    feature_cols = [
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
        "from_bank",
        "to_bank",
        "receiving_currency",
        "payment_currency",
        "payment_format",
    ]
    return work[feature_cols], work["is_laundering"].astype(int)


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float,
    val_size: float,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )
    relative_val = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp,
        y_tmp,
        test_size=relative_val,
        stratify=y_tmp,
        random_state=random_state + 1,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def prepare_categorical(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    categorical_cols: List[str],
) -> Tuple[Dict[str, Dict[str, int]], Dict[str, int], Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    mappings: Dict[str, Dict[str, int]] = {}
    cardinalities: Dict[str, int] = {}
    train_encoded: Dict[str, np.ndarray] = {}
    val_encoded: Dict[str, np.ndarray] = {}
    test_encoded: Dict[str, np.ndarray] = {}

    for col in categorical_cols:
        uniques = X_train[col].dropna().unique().tolist()
        mapping = {value: idx + 1 for idx, value in enumerate(uniques)}
        mappings[col] = mapping
        cardinalities[col] = len(mapping) + 1

        def encode(series: pd.Series) -> np.ndarray:
            coded = series.map(mapping).fillna(0).astype(np.int64)
            return coded.to_numpy()

        train_encoded[col] = encode(X_train[col])
        val_encoded[col] = encode(X_val[col])
        test_encoded[col] = encode(X_test[col])

    return mappings, cardinalities, train_encoded, val_encoded, test_encoded


def prepare_numeric(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    numeric_cols: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_numeric = X_train[numeric_cols].copy()
    medians = train_numeric.median()
    train_numeric = train_numeric.fillna(medians)
    means = train_numeric.mean()
    stds = train_numeric.std().replace(0, 1.0)

    def transform(df: pd.DataFrame) -> np.ndarray:
        filled = df[numeric_cols].fillna(medians)
        normalized = (filled - means) / stds
        return normalized.to_numpy(dtype=np.float32)

    train_array = transform(X_train)
    val_array = transform(X_val)
    test_array = transform(X_test)
    return train_array, val_array, test_array


def build_datasets(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
    numeric_cols: List[str],
    categorical_cols: List[str],
) -> Tuple[
    TabularDataset,
    TabularDataset,
    TabularDataset,
    List[int],
]:
    _, cardinalities, train_cats, val_cats, test_cats = prepare_categorical(
        X_train, X_val, X_test, categorical_cols
    )
    train_numeric, val_numeric, test_numeric = prepare_numeric(
        X_train, X_val, X_test, numeric_cols
    )

    def stack_cats(encoded: Dict[str, np.ndarray]) -> np.ndarray:
        if not encoded:
            return np.zeros((len(y_train), 0), dtype=np.int64)
        ordered = [encoded[col] for col in categorical_cols]
        return np.stack(ordered, axis=1)

    train_cat = stack_cats(train_cats)
    val_cat = stack_cats(val_cats)
    test_cat = stack_cats(test_cats)

    train_ds = TabularDataset(train_numeric, train_cat, y_train.to_numpy())
    val_ds = TabularDataset(val_numeric, val_cat, y_val.to_numpy())
    test_ds = TabularDataset(test_numeric, test_cat, y_test.to_numpy())

    cardinality_list = [cardinalities[col] for col in categorical_cols]
    return train_ds, val_ds, test_ds, cardinality_list


def determine_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def create_model(
    train_dataset: TabularDataset,
    cardinalities: List[int],
    hidden_dims: List[int],
    dropout: float,
    base_embed_dim: int,
    max_embed_dim: int,
) -> TabularMLP:
    num_numeric = train_dataset.numeric.shape[1]
    model = TabularMLP(
        num_numeric=num_numeric,
        categorical_cardinalities=cardinalities,
        hidden_dims=hidden_dims,
        dropout=dropout,
        base_embed_dim=base_embed_dim,
        max_embed_dim=max_embed_dim,
    )
    return model


def threshold_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> Dict[str, Any]:
    preds = (y_prob >= threshold).astype(int)
    tp = int(np.sum((preds == 1) & (y_true == 1)))
    fp = int(np.sum((preds == 1) & (y_true == 0)))
    fn = int(np.sum((preds == 0) & (y_true == 1)))
    tn = int(np.sum((preds == 0) & (y_true == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    tnr = tn / (tn + fp) if tn + fp else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "tnr": tnr,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
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
        viable: List[Dict[str, Any]] = []
        for candidate in grid:
            metrics = threshold_metrics(y_true, y_prob, float(candidate))
            if metrics["fpr"] <= target_fpr:
                viable.append(metrics)
        if viable:
            results["target_fpr"] = max(viable, key=lambda item: item["recall"])
    return results


def run_epoch(
    model: nn.Module,
    loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    criterion: nn.Module,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0
    for numeric, categorical, labels in loader:
        numeric = numeric.to(device)
        categorical = categorical.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(numeric, categorical)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.detach().item() * batch_size
        total_samples += batch_size
    return total_loss / max(total_samples, 1)


def predict_proba(
    model: nn.Module,
    loader: Iterable,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_probs: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    with torch.no_grad():
        for numeric, categorical, labels in loader:
            numeric = numeric.to(device)
            categorical = categorical.to(device)
            logits = model(numeric, categorical)
            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)


def evaluate(
    model: nn.Module,
    loader: Iterable,
    device: torch.device,
    target_fpr: float | None,
) -> Dict[str, Any]:
    probs, labels = predict_proba(model, loader, device)
    roc_auc = roc_auc_score(labels, probs)
    precision, recall, _ = precision_recall_curve(labels, probs)
    pr_auc = auc(recall, precision)
    thresholds = select_thresholds(labels, probs, target_fpr)
    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "thresholds": thresholds,
        "positive_rate": float(labels.mean()),
    }


def format_hidden_dims(spec: str) -> List[int]:
    if not spec:
        return []
    return [int(part.strip()) for part in spec.split(",") if part.strip()]


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.random_state)
    np.random.seed(args.random_state)

    df = load_transactions(args.dataset, args.max_samples)
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

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X,
        y,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.random_state,
    )

    train_ds, val_ds, test_ds, cardinalities = build_datasets(
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        numeric_cols,
        categorical_cols,
    )

    hidden_dims = format_hidden_dims(args.hidden_dims)
    model = create_model(
        train_ds,
        cardinalities,
        hidden_dims,
        args.dropout,
        args.base_embed_dim,
        args.max_embed_dim,
    )

    device = determine_device(args.device)
    model.to(device)

    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin,
    )

    positives = float(y_train.sum())
    negatives = float(len(y_train) - positives)
    pos_weight_value = negatives / max(positives, 1.0)
    pos_weight = torch.tensor(pos_weight_value, dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    best_state = deepcopy(model.state_dict())
    best_metric = -np.inf

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, device, criterion)
        val_metrics = evaluate(model, val_loader, device, args.target_fpr)
        if val_metrics["pr_auc"] > best_metric:
            best_metric = val_metrics["pr_auc"]
            best_state = deepcopy(model.state_dict())
        print(
            f"Epoch {epoch:03d} | Loss {train_loss:.4f} | Val ROC AUC {val_metrics['roc_auc']:.4f} | "
            f"Val PR AUC {val_metrics['pr_auc']:.4f}"
        )

    model.load_state_dict(best_state)
    val_metrics = evaluate(model, val_loader, device, args.target_fpr)
    test_metrics = evaluate(model, test_loader, device, args.target_fpr)

    print("\n=== Validation Metrics ===")
    print(f"ROC AUC: {val_metrics['roc_auc']:.4f}")
    print(f"PR AUC : {val_metrics['pr_auc']:.4f}")
    print(f"Positive prevalence: {val_metrics['positive_rate']:.4%}")
    for name, detail in val_metrics["thresholds"].items():
        print(f"\nThreshold strategy: {name}")
        for key, value in detail.items():
            if key in {"tp", "fp", "fn", "tn"}:
                print(f"  {key.upper():<3}: {value}")
            else:
                print(f"  {key:<10}: {value:.4f}")

    print("\n=== Test Metrics ===")
    print(f"ROC AUC: {test_metrics['roc_auc']:.4f}")
    print(f"PR AUC : {test_metrics['pr_auc']:.4f}")
    print(f"Positive prevalence: {test_metrics['positive_rate']:.4%}")
    for name, detail in test_metrics["thresholds"].items():
        print(f"\nThreshold strategy: {name}")
        for key, value in detail.items():
            if key in {"tp", "fp", "fn", "tn"}:
                print(f"  {key.upper():<3}: {value}")
            else:
                print(f"  {key:<10}: {value:.4f}")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "validation": val_metrics,
            "test": test_metrics,
        }
        with args.report.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"\nMetrics saved to {args.report}")


if __name__ == "__main__":
    main()
