from __future__ import annotations

from dataclasses import asdict
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .features.drug import is_valid_smiles
from .features.protein import is_valid_protein_sequence
from .utils import DatasetStats


REQUIRED_COLUMNS = ("compound_iso_smiles", "target_sequence", "label")


def load_dti_csv(
    path: str,
    drop_invalid: bool = True,
    label_column: str = "label",
) -> Tuple[pd.DataFrame, DatasetStats]:
    """Load a DTI dataset CSV and validate raw inputs.

    Parameters
    ----------
    path:
        Path to CSV with required columns:
        - compound_iso_smiles
        - target_sequence
        - label (0/1)
    drop_invalid:
        If True, rows with invalid SMILES or protein sequences are dropped.
        If False, rows are kept but invalid inputs will be featurized as all-zeros.
    label_column:
        Column name to use as the binary label.
    """
    df = pd.read_csv(path)
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col!r}. Found columns: {list(df.columns)}")
    if label_column != "label" and label_column not in df.columns:
        raise ValueError(f"label_column={label_column!r} not found in CSV.")

    # Clean types / missingness (keep it simple + explicit for explainability).
    df = df.copy()
    df["compound_iso_smiles"] = df["compound_iso_smiles"].astype(str)
    df["target_sequence"] = df["target_sequence"].astype(str)

    # Normalize labels to {0,1}
    y = df[label_column].values
    try:
        y = y.astype(int)
    except Exception as e:
        raise ValueError("Labels must be convertible to int (0/1).") from e

    unique = set(np.unique(y).tolist())
    if not unique.issubset({0, 1}):
        raise ValueError(f"Label column must be binary (0/1). Found unique labels: {sorted(unique)}")

    df[label_column] = y

    valid_smiles_mask = df["compound_iso_smiles"].map(is_valid_smiles).astype(bool).values
    valid_seq_mask = df["target_sequence"].map(is_valid_protein_sequence).astype(bool).values

    invalid_smiles = int((~valid_smiles_mask).sum())
    invalid_seq = int((~valid_seq_mask).sum())

    if drop_invalid:
        keep = valid_smiles_mask & valid_seq_mask
        df_valid = df.loc[keep].reset_index(drop=True)
    else:
        df_valid = df.reset_index(drop=True)

    n_pos = int((df_valid[label_column].values == 1).sum())
    n_neg = int((df_valid[label_column].values == 0).sum())

    stats = DatasetStats(
        n_rows_raw=int(len(df)),
        n_rows_valid=int(len(df_valid)),
        n_invalid_smiles=invalid_smiles,
        n_invalid_sequences=invalid_seq,
        n_pos=n_pos,
        n_neg=n_neg,
    )
    return df_valid, stats


def dataset_stats_to_dict(stats: DatasetStats) -> dict:
    return asdict(stats)

