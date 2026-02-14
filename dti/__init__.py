"""
DTI (Drug–Target Interaction) prediction package.

All feature extraction is performed from raw inputs:
- Drug: SMILES string
- Protein: amino-acid sequence
"""

from .utils import set_global_seed

__all__ = ["set_global_seed"]

