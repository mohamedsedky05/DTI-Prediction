from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from .drug import (
    DrugFeaturizerConfig,
    featurize_smiles,
    get_drug_descriptor_names,
    get_morgan_feature_names,
)
from .protein import ProteinFeaturizerConfig, featurize_protein, get_protein_feature_names


@dataclass(frozen=True)
class DTIFeaturizerConfig:
    drug: DrugFeaturizerConfig = DrugFeaturizerConfig()
    protein: ProteinFeaturizerConfig = ProteinFeaturizerConfig()


class DTIFeaturizer(BaseEstimator, TransformerMixin):
    """Featurize raw (SMILES, protein sequence) pairs into a single numeric vector.

    Expected input format:
    - pandas DataFrame with columns: `compound_iso_smiles`, `target_sequence`, OR
    - numpy array-like with 2 columns: [smiles, sequence]
    """

    def __init__(
        self,
        cfg: DTIFeaturizerConfig = DTIFeaturizerConfig(),
        smiles_col: str = "compound_iso_smiles",
        seq_col: str = "target_sequence",
    ) -> None:
        self.cfg = cfg
        self.smiles_col = smiles_col
        self.seq_col = seq_col

        self._n_desc = len(get_drug_descriptor_names())
        self._n_fp = int(cfg.drug.n_bits)
        self._n_prot = 20 + int(cfg.protein.kmer_dim)

    @property
    def n_features_(self) -> int:  # sklearn-like API
        return self._n_desc + self._n_fp + self._n_prot

    @property
    def fingerprint_slice_(self) -> slice:
        """Slice covering Morgan fingerprint bits in the final feature vector."""
        start = self._n_desc
        end = self._n_desc + self._n_fp
        return slice(start, end)

    def get_feature_names_out(self) -> np.ndarray:
        names: List[str] = []
        names.extend([f"drug_desc_{n}" for n in get_drug_descriptor_names()])
        names.extend(get_morgan_feature_names(self.cfg.drug.n_bits))
        names.extend(get_protein_feature_names(self.cfg.protein.kmer_dim))
        return np.asarray(names, dtype=object)

    def fit(self, X, y=None):
        return self

    def _extract_smiles_and_seq(self, X) -> Tuple[List[str], List[str]]:
        if isinstance(X, pd.DataFrame):
            smiles = X[self.smiles_col].astype(str).tolist()
            seqs = X[self.seq_col].astype(str).tolist()
            return smiles, seqs
        # numpy array-like
        arr = np.asarray(X)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(
                "DTIFeaturizer expects a DataFrame with SMILES/sequence columns "
                "or a 2D array with exactly 2 columns: [smiles, sequence]."
            )
        return arr[:, 0].astype(str).tolist(), arr[:, 1].astype(str).tolist()

    def transform(self, X) -> np.ndarray:
        smiles_list, seq_list = self._extract_smiles_and_seq(X)
        n = len(smiles_list)
        out = np.zeros((n, self.n_features_), dtype=np.float32)

        for i, (smiles, seq) in enumerate(zip(smiles_list, seq_list)):
            desc, fp, _ok_d = featurize_smiles(smiles, self.cfg.drug)
            prot, _ok_p = featurize_protein(seq, self.cfg.protein)

            # Assemble: [drug_desc | drug_fp | protein]
            out[i, 0 : self._n_desc] = desc
            out[i, self._n_desc : self._n_desc + self._n_fp] = fp.astype(np.float32, copy=False)
            out[i, self._n_desc + self._n_fp :] = prot

        return out

