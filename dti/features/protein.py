from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


# 20 canonical amino acids used for AAC (amino-acid composition).
CANONICAL_AA = "ACDEFGHIKLMNPQRSTVWY"
CANONICAL_SET = set(CANONICAL_AA)


def normalize_protein_sequence(seq: str) -> str:
    """Normalize protein sequence by stripping whitespace and uppercasing."""
    if seq is None:
        return ""
    return "".join(str(seq).split()).upper()


def is_valid_protein_sequence(seq: str, allow_x: bool = True) -> bool:
    """Basic validation for protein sequences.

    Rules (simple + explainable):
    - sequence must be non-empty after normalization
    - must contain only canonical amino acids (plus optional 'X' unknown)
    """
    s = normalize_protein_sequence(seq)
    if not s or s.lower() in {"nan", "none"}:
        return False
    allowed = CANONICAL_SET | ({"X"} if allow_x else set())
    return all((c in allowed) for c in s)


@dataclass(frozen=True)
class ProteinFeaturizerConfig:
    kmer_k: int = 3
    kmer_dim: int = 1024
    allow_x: bool = True


def amino_acid_composition(seq: str, allow_x: bool = True) -> Tuple[np.ndarray, bool]:
    """Return (AAC vector of shape (20,), is_valid)."""
    s = normalize_protein_sequence(seq)
    if not is_valid_protein_sequence(s, allow_x=allow_x):
        return np.zeros((20,), dtype=np.float32), False

    # Count only canonical residues; ignore 'X' if present.
    counts = np.zeros((20,), dtype=np.float32)
    length = 0
    for c in s:
        if c in CANONICAL_SET:
            counts[CANONICAL_AA.index(c)] += 1.0
            length += 1
    if length == 0:
        return np.zeros((20,), dtype=np.float32), False
    return counts / float(length), True


def _stable_hash_to_uint64(text: str) -> int:
    """Stable hashing for reproducible feature hashing across runs/platforms."""
    h = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, byteorder="little", signed=False)


def hashed_kmer_counts(seq: str, k: int, dim: int, allow_x: bool = True) -> Tuple[np.ndarray, bool]:
    """Return (hashed k-mer vector of shape (dim,), is_valid).

    Implementation notes (from scratch, explainable):
    - We slide a window of length k across the sequence.
    - Each k-mer is mapped to an index via a stable hash mod `dim`.
    - We use *signed hashing* (+1 / -1) to reduce collision bias.
    - We length-normalize by number of k-mers so values are comparable across proteins.
    """
    s = normalize_protein_sequence(seq)
    if not is_valid_protein_sequence(s, allow_x=allow_x):
        return np.zeros((dim,), dtype=np.float32), False
    if k <= 0:
        raise ValueError("k must be >= 1")
    if dim <= 0:
        raise ValueError("dim must be >= 1")
    if len(s) < k:
        return np.zeros((dim,), dtype=np.float32), False

    vec = np.zeros((dim,), dtype=np.float32)
    n_kmers = 0
    for i in range(len(s) - k + 1):
        kmer = s[i : i + k]
        h = _stable_hash_to_uint64(kmer)
        idx = int(h % dim)
        sign = 1.0 if ((h >> 63) & 1) == 1 else -1.0
        vec[idx] += sign
        n_kmers += 1

    if n_kmers > 0:
        vec /= float(n_kmers)
    return vec, True


def get_protein_feature_names(kmer_dim: int) -> List[str]:
    aac = [f"prot_aac_{aa}" for aa in CANONICAL_AA]
    kmers = [f"prot_kmerhash_{i:04d}" for i in range(kmer_dim)]
    return aac + kmers


def featurize_protein(
    seq: str,
    cfg: ProteinFeaturizerConfig,
) -> Tuple[np.ndarray, bool]:
    """Return (protein_features, is_valid). Invalid -> zeros + False."""
    aac, ok_aac = amino_acid_composition(seq, allow_x=cfg.allow_x)
    kmer, ok_kmer = hashed_kmer_counts(
        seq, k=cfg.kmer_k, dim=cfg.kmer_dim, allow_x=cfg.allow_x
    )
    ok = bool(ok_aac and ok_kmer)
    feats = np.concatenate([aac, kmer]).astype(np.float32, copy=False)
    return feats, ok

