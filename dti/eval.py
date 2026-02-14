from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold


@dataclass(frozen=True)
class FoldMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float


def compute_metrics(y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5) -> FoldMetrics:
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba).astype(float)
    y_pred = (y_proba >= threshold).astype(int)

    # Use zero_division=0 to keep reports stable when a class is never predicted.
    return FoldMetrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        roc_auc=float(roc_auc_score(y_true, y_proba)),
    )


def _predict_proba_positive(estimator, X) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        proba = estimator.predict_proba(X)
        return np.asarray(proba)[:, 1]
    if hasattr(estimator, "decision_function"):
        # Convert scores to [0,1] via sigmoid for ROC-AUC compatibility.
        scores = np.asarray(estimator.decision_function(X))
        return 1.0 / (1.0 + np.exp(-scores))
    raise ValueError("Estimator must support predict_proba or decision_function.")


def cross_validate_binary_classifier(
    pipeline,
    X,
    y: np.ndarray,
    cv_folds: int,
    seed: int,
    sample_weight: Optional[np.ndarray] = None,
) -> Tuple[List[FoldMetrics], Dict[str, Any]]:
    """Stratified K-fold CV with leakage-safe sklearn pipeline."""
    y = np.asarray(y).astype(int)
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    fold_metrics: List[FoldMetrics] = []
    for train_idx, test_idx in skf.split(np.zeros_like(y), y):
        est = clone(pipeline)
        fit_kwargs = {}
        if sample_weight is not None:
            fit_kwargs["clf__sample_weight"] = sample_weight[train_idx]

        est.fit(_subset(X, train_idx), y[train_idx], **fit_kwargs)
        y_proba = _predict_proba_positive(est, _subset(X, test_idx))
        fold_metrics.append(compute_metrics(y[test_idx], y_proba))

    summary = summarize_folds(fold_metrics)
    return fold_metrics, summary


def out_of_fold_pred_proba(
    pipeline,
    X,
    y: np.ndarray,
    cv_folds: int,
    seed: int,
    sample_weight: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute out-of-fold predicted probabilities for ROC (leakage-safe).

    Returns a vector `y_proba_oof` of length n_samples where each entry is predicted
    by a model that did not train on that sample.
    """
    y = np.asarray(y).astype(int)
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    y_proba_oof = np.zeros((len(y),), dtype=float)

    for train_idx, test_idx in skf.split(np.zeros_like(y), y):
        est = clone(pipeline)
        fit_kwargs = {}
        if sample_weight is not None:
            fit_kwargs["clf__sample_weight"] = sample_weight[train_idx]
        est.fit(_subset(X, train_idx), y[train_idx], **fit_kwargs)
        y_proba_oof[test_idx] = _predict_proba_positive(est, _subset(X, test_idx))

    return y_proba_oof


def summarize_folds(folds: List[FoldMetrics]) -> Dict[str, Any]:
    rows = [asdict(m) for m in folds]
    df = pd.DataFrame(rows)
    out: Dict[str, Any] = {"per_fold": rows}
    for col in df.columns:
        out[f"{col}_mean"] = float(df[col].mean())
        out[f"{col}_std"] = float(df[col].std(ddof=1)) if len(df) > 1 else 0.0
    return out


def extract_feature_importance(pipeline) -> Optional[pd.DataFrame]:
    """Extract a feature-importance table if the estimator supports it.

    Supported:
    - LogisticRegression: |coef|
    - RandomForest / XGBoost: feature_importances_

    Returns None if not available (e.g., MLP without permutation importance).
    """
    if not hasattr(pipeline, "named_steps"):
        return None
    steps = pipeline.named_steps
    if "featurize" not in steps or "clf" not in steps:
        return None

    featurizer = steps["featurize"]
    clf = steps["clf"]

    try:
        names = featurizer.get_feature_names_out()
    except Exception:
        return None

    importances = None
    method = None

    if hasattr(clf, "coef_"):
        coef = np.asarray(clf.coef_)
        # binary case: shape (1, n_features)
        if coef.ndim == 2 and coef.shape[0] == 1:
            coef = coef[0]
        importances = np.abs(coef).astype(float)
        method = "abs_coef"
    elif hasattr(clf, "feature_importances_"):
        importances = np.asarray(clf.feature_importances_).astype(float)
        method = "feature_importances_"

    if importances is None or importances.shape[0] != len(names):
        return None

    df = pd.DataFrame(
        {
            "feature": names,
            "importance": importances,
            "method": method,
        }
    ).sort_values("importance", ascending=False, kind="mergesort")
    return df


def save_json(obj: Dict[str, Any], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def save_roc_curve_plot(y_true: np.ndarray, y_proba: np.ndarray, path: str) -> Dict[str, Any]:
    """Save ROC curve plot and return curve stats (AUC + points)."""
    # Import matplotlib lazily (keeps the core import path lightweight).
    import matplotlib.pyplot as plt

    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba).astype(float)

    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    auc = float(roc_auc_score(y_true, y_proba))

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (out-of-fold)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

    return {
        "auc": auc,
        "n_points": int(len(fpr)),
    }


def _subset(X, idx: np.ndarray):
    # Keep pandas DataFrame support (preferred for column-based featurizer input).
    if hasattr(X, "iloc"):
        return X.iloc[idx]
    arr = np.asarray(X)
    return arr[idx]

