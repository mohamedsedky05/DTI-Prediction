from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class PreprocessConfig:
    """Preprocessing configuration.

    We intentionally do NOT scale binary fingerprint bits (0/1).
    We DO scale continuous-valued features (descriptors, AAC, hashed k-mers).
    """

    scale_continuous: bool = True


class SelectiveStandardScaler(BaseEstimator, TransformerMixin):
    """Standardize only selected columns, keeping original column order.

    This is important for:
    - interpretability/feature naming
    - keeping fingerprint bits as 0/1
    - avoiding ColumnTransformer reordering surprises
    """

    def __init__(
        self,
        continuous_mask: np.ndarray,
        with_mean: bool = True,
        with_std: bool = True,
    ) -> None:
        self.continuous_mask = np.asarray(continuous_mask, dtype=bool)
        self.with_mean = with_mean
        self.with_std = with_std
        self._scaler: Optional[StandardScaler] = None

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError("SelectiveStandardScaler expects a 2D array.")
        if self.continuous_mask.shape[0] != X.shape[1]:
            raise ValueError(
                f"continuous_mask length {self.continuous_mask.shape[0]} "
                f"does not match n_features {X.shape[1]}"
            )
        cols = np.where(self.continuous_mask)[0]
        self._scaler = StandardScaler(with_mean=self.with_mean, with_std=self.with_std)
        self._scaler.fit(X[:, cols])
        return self

    def transform(self, X) -> np.ndarray:
        if self._scaler is None:
            raise RuntimeError("SelectiveStandardScaler is not fitted.")
        X = np.asarray(X, dtype=np.float32)
        out = X.copy()
        cols = np.where(self.continuous_mask)[0]
        out[:, cols] = self._scaler.transform(X[:, cols])
        return out


def continuous_mask_excluding_fingerprint(
    n_features: int,
    fingerprint_slice: slice,
) -> np.ndarray:
    """Create a boolean mask where fingerprint columns are False and all others True."""
    mask = np.ones((n_features,), dtype=bool)
    mask[fingerprint_slice] = False
    return mask

