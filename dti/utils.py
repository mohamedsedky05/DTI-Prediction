from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np


def set_global_seed(seed: int) -> None:
    """Best-effort reproducibility across numpy/python hash/random.

    Notes:
    - Full determinism is not guaranteed for every algorithm/backend, but this is a good baseline.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


@dataclass(frozen=True)
class DatasetStats:
    n_rows_raw: int
    n_rows_valid: int
    n_invalid_smiles: int
    n_invalid_sequences: int
    n_pos: int
    n_neg: int


def safe_div(numer: float, denom: float) -> float:
    return float(numer) / float(denom) if denom else 0.0

