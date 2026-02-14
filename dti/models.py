from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_sample_weight


@dataclass(frozen=True)
class ModelConfig:
    model_name: str  # "logreg" | "rf" | "xgb" | "mlp"
    seed: int = 42


def compute_imbalance_params(y: np.ndarray) -> Dict[str, Any]:
    """Return parameters helpful for class imbalance.

    - scale_pos_weight for XGBoost: n_negative / n_positive
    """
    y = np.asarray(y).astype(int)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    scale_pos_weight = float(n_neg) / float(n_pos) if n_pos > 0 else 1.0
    return {"n_pos": n_pos, "n_neg": n_neg, "scale_pos_weight": scale_pos_weight}


def make_logreg(seed: int) -> LogisticRegression:
    # SAGA handles larger feature spaces well; class_weight handles imbalance.
    return LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="saga",
        max_iter=2000,
        class_weight="balanced",
        n_jobs=-1,
        random_state=seed,
    )


def make_rf(seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=600,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=seed,
    )


def make_xgb(seed: int, scale_pos_weight: float):
    # Import lazily so the project still runs if xgboost is not installed.
    try:
        from xgboost import XGBClassifier
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "xgboost is not installed. Install it (pip) or use --model mlp."
        ) from e

    return XGBClassifier(
        n_estimators=800,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        min_child_weight=1.0,
        gamma=0.0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=seed,
        scale_pos_weight=scale_pos_weight,
    )


def make_mlp(seed: int) -> MLPClassifier:
    # A simple non-linear baseline. Early stopping prevents overfitting and long runtimes.
    return MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        alpha=1e-4,
        batch_size=256,
        learning_rate="adaptive",
        learning_rate_init=1e-3,
        max_iter=200,
        early_stopping=True,
        n_iter_no_change=15,
        random_state=seed,
        verbose=False,
    )


def make_estimator(model_name: str, seed: int, y_for_imbalance: Optional[np.ndarray] = None):
    name = model_name.lower().strip()
    if name == "logreg":
        return make_logreg(seed)
    if name == "rf":
        return make_rf(seed)
    if name == "mlp":
        return make_mlp(seed)
    if name == "xgb":
        if y_for_imbalance is None:
            scale_pos_weight = 1.0
        else:
            scale_pos_weight = compute_imbalance_params(y_for_imbalance)["scale_pos_weight"]
        return make_xgb(seed, scale_pos_weight=scale_pos_weight)
    raise ValueError(f"Unknown model_name={model_name!r}. Choose from: logreg, rf, xgb, mlp.")


def maybe_sample_weight_for_estimator(estimator, y: np.ndarray) -> Optional[np.ndarray]:
    """Return sample weights when needed/beneficial.

    - LogReg/RF: class_weight is already used; sample weights are optional.
    - MLP: no class_weight in sklearn MLPClassifier -> use sample weights.
    - XGBoost: scale_pos_weight is used at estimator construction; sample weights optional.
    """
    if isinstance(estimator, MLPClassifier):
        return compute_sample_weight(class_weight="balanced", y=y)
    return None

