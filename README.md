# Drug–Target Interaction (DTI) Prediction (raw SMILES + raw protein sequence)

This project trains explainable baseline and advanced ML models to predict whether a **drug molecule** interacts with a **protein target** using only:

- `compound_iso_smiles` (SMILES string)
- `target_sequence` (amino-acid sequence)
- `label` (0/1 interaction)

No precomputed/pre-engineered features are assumed: **all featurization is implemented here**.

---

## Why these representations?

### Drug (from SMILES via RDKit)

- **RDKit molecular descriptors (interpretable, global properties)**  
  Examples: molecular weight, LogP, H-bond donors/acceptors, TPSA, rotatable bonds, ring count, fraction Csp3.  
  These are low-dimensional, human-interpretable signals that often correlate with binding likelihood.

- **Morgan fingerprint (ECFP-like, local substructure signal)**  
  We compute a fixed-length binary vector (default 2048 bits) with radius 2.  
  This captures local chemical environments/substructures and is a standard baseline in cheminformatics.

### Protein (from amino-acid sequence)

- **Amino-acid composition (AAC, interpretable global signal)**  
  A 20-dim vector: fraction of each amino acid in the sequence.  
  This captures coarse properties (e.g., enrichment of hydrophobic/polar residues).

- **Hashed k-mer counts (captures local motifs, controlled dimensionality)**  
  We count k-mers (default k=3) and map them into a fixed-dimensional vector (default 1024) using a hashing trick.  
  This approximates a bag-of-k-mers representation without exploding dimensionality (since \(20^k\) grows fast).

---

## Project structure

- `dti/`
  - `io.py`: dataset loading + validation
  - `features/`
    - `drug.py`: RDKit descriptors + Morgan fingerprints
    - `protein.py`: AAC + hashed k-mer encoding
    - `dti.py`: concatenation + feature-name management
  - `preprocess.py`: scaling only continuous features (keeps fingerprints as 0/1)
  - `models.py`: model factory (LogReg, RF, XGBoost or MLP) + class-imbalance handling
  - `eval.py`: metrics + stratified cross-validation + reporting
- `train.py`: CLI for training/evaluating and saving artifacts

Artifacts are saved under `--outdir`:

- `model.joblib`: full pipeline (featurization + preprocessing + classifier)
- `report.json`: metrics summary (CV means/std, dataset stats)
- `feature_importance.csv`: top features (if applicable)

---

## Setup

### Option A (recommended on Windows): Conda + RDKit

Create the environment from `environment.yml`:

```bash
conda env create -f environment.yml
conda activate dti
```

### Option B: pip (may fail for RDKit on Windows)

```bash
pip install -r requirements.txt
```

---

## Data format

Input CSV must contain:

- `compound_iso_smiles`
- `target_sequence`
- `label` (0/1)

Optional columns are allowed (e.g., `affinity`).

---

## Train + evaluate

```bash
python train.py --data path\to\data.csv --outdir runs\baseline_logreg --model logreg
python train.py --data path\to\data.csv --outdir runs\baseline_rf --model rf
python train.py --data path\to\data.csv --outdir runs\advanced_xgb --model xgb
```

Key options:

- `--cv 5`: stratified CV folds
- `--seed 42`: reproducibility
- `--drop_invalid`: drop rows with invalid SMILES/sequence (recommended for correctness)
- `--kmer_k 3 --kmer_dim 1024`: protein k-mer hashing settings

---

## Notes on scientific correctness

- **Invalid inputs**: invalid SMILES or sequences are detected and (by default) removed, and counts are reported.
- **Leakage-safe scaling**: scalers are fit **inside each CV fold** via scikit-learn pipelines.
- **Class imbalance**: handled via `class_weight` (LogReg/RF), `scale_pos_weight` (XGBoost), or sample weights (MLP).

