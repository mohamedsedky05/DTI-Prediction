from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

# RDKit is a required dependency for drug featurization.
# On Windows, it is usually easiest to install via conda-forge.
try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
except Exception as e:  # pragma: no cover
    Chem = None  # type: ignore[assignment]
    DataStructs = None  # type: ignore[assignment]
    AllChem = None  # type: ignore[assignment]
    Descriptors = None  # type: ignore[assignment]
    rdMolDescriptors = None  # type: ignore[assignment]
    _RDKit_IMPORT_ERROR = e
else:
    _RDKit_IMPORT_ERROR = None


def _require_rdkit() -> None:
    if _RDKit_IMPORT_ERROR is not None:
        raise ImportError(
            "RDKit is required for SMILES featurization but could not be imported.\n"
            "Recommended (Windows): `conda env create -f environment.yml` then `conda activate dti`.\n"
            f"Original error: {_RDKit_IMPORT_ERROR}"
        )


def is_valid_smiles(smiles: str) -> bool:
    """Return True if RDKit can parse the SMILES string."""
    _require_rdkit()
    if smiles is None:
        return False
    s = str(smiles).strip()
    if not s or s.lower() in {"nan", "none"}:
        return False
    mol = Chem.MolFromSmiles(s)
    return mol is not None


@dataclass(frozen=True)
class DrugFeaturizerConfig:
    radius: int = 2
    n_bits: int = 2048
    use_chirality: bool = True


def get_drug_descriptor_names() -> List[str]:
    """A compact, explainable set of RDKit descriptors (global molecular properties)."""
    # Keep this list stable for reproducibility and feature-name reporting.
    return [
        "MolWt",
        "MolLogP",
        "MolMR",
        "TPSA",
        "NumHDonors",
        "NumHAcceptors",
        "NumRotatableBonds",
        "NumRings",
        "NumAromaticRings",
        "NumAliphaticRings",
        "HeavyAtomCount",
        "FractionCSP3",
        "BertzCT",
        "LabuteASA",
    ]


def compute_drug_descriptors(mol) -> np.ndarray:
    """Compute the descriptor vector for a parsed RDKit Mol."""
    _require_rdkit()
    # Descriptor implementations are standard RDKit formulas, but computed here from raw SMILES->Mol.
    vals = [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.MolMR(mol),
        rdMolDescriptors.CalcTPSA(mol),
        rdMolDescriptors.CalcNumHBD(mol),
        rdMolDescriptors.CalcNumHBA(mol),
        rdMolDescriptors.CalcNumRotatableBonds(mol),
        rdMolDescriptors.CalcNumRings(mol),
        rdMolDescriptors.CalcNumAromaticRings(mol),
        rdMolDescriptors.CalcNumAliphaticRings(mol),
        float(mol.GetNumHeavyAtoms()),
        rdMolDescriptors.CalcFractionCSP3(mol),
        Descriptors.BertzCT(mol),
        Descriptors.LabuteASA(mol),
    ]
    return np.asarray(vals, dtype=np.float32)


def morgan_fingerprint_bits(
    mol,
    radius: int,
    n_bits: int,
    use_chirality: bool,
) -> np.ndarray:
    """Compute a binary Morgan fingerprint as a numpy array of shape (n_bits,)."""
    _require_rdkit()
    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol,
        radius=radius,
        nBits=n_bits,
        useChirality=use_chirality,
    )
    arr = np.zeros((n_bits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def featurize_smiles(
    smiles: str,
    cfg: DrugFeaturizerConfig,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Return (descriptors, fingerprint_bits, is_valid). Invalid SMILES -> zeros + False."""
    _require_rdkit()
    n_desc = len(get_drug_descriptor_names())
    desc_zeros = np.zeros((n_desc,), dtype=np.float32)
    fp_zeros = np.zeros((cfg.n_bits,), dtype=np.int8)

    if smiles is None:
        return desc_zeros, fp_zeros, False

    s = str(smiles).strip()
    if not s or s.lower() in {"nan", "none"}:
        return desc_zeros, fp_zeros, False

    mol = Chem.MolFromSmiles(s)
    if mol is None:
        return desc_zeros, fp_zeros, False

    desc = compute_drug_descriptors(mol)
    fp = morgan_fingerprint_bits(mol, cfg.radius, cfg.n_bits, cfg.use_chirality)
    return desc, fp, True


def get_morgan_feature_names(n_bits: int) -> List[str]:
    # We cannot name hashed/circular substructures without additional bit->fragment analysis,
    # so we report bit indices for transparent interpretability.
    return [f"drug_morgan_bit_{i:04d}" for i in range(n_bits)]

