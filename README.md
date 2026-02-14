# Drug–Target Interaction (DTI) Prediction System
### Graduation Project – Faculty of Computers & Information

An end-to-end machine learning system for predicting **drug–protein interactions** directly from raw molecular SMILES and protein amino-acid sequences.

---

## 🚀 Overview

This project trains explainable baseline and advanced machine learning models to predict whether a **drug molecule interacts with a protein target** using only:

- `compound_iso_smiles` → molecular structure  
- `target_sequence` → amino-acid sequence  
- `label` → interaction (0 / 1)

No precomputed features are assumed — **all feature engineering is implemented within this repository**.

The system is designed for:

✔ scientific correctness  
✔ reproducibility  
✔ scalability  
✔ deployment readiness  

---

## 🎓 Project Context

This system was developed as a **graduation project** focusing on:

- AI in drug discovery & bioinformatics  
- feature extraction from biological data  
- machine learning pipeline engineering  
- model evaluation & deployment  
- building a reproducible ML workflow  

---

## 🧬 Why These Representations?

### 🧪 Drug Representation (from SMILES via RDKit)

#### ✔ RDKit molecular descriptors (interpretable, global properties)
Examples:

- Molecular weight  
- LogP  
- H-bond donors/acceptors  
- TPSA  
- Rotatable bonds  
- Ring count  
- Fraction Csp3  

These low-dimensional descriptors provide human-interpretable signals often correlated with binding likelihood.

#### ✔ Morgan fingerprint (ECFP-like, local substructure signal)

- Radius = 2  
- Default length = 2048 bits  
- Captures local chemical environments & substructures  
- Standard baseline in cheminformatics  

---

### 🧫 Protein Representation (from amino-acid sequence)

#### ✔ Amino-acid composition (AAC, interpretable global signal)

A 20-dimensional vector representing the fraction of each amino acid.  
Captures coarse biochemical properties such as hydrophobicity and polarity.

#### ✔ Hashed k-mer counts (local motif representation)

- Default k = 3  
- Fixed dimensionality (default = 1024)  
- Captures local sequence motifs  
- Uses hashing to avoid exponential growth of \(20^k\)

---

## ⚙️ Final Feature Vector

[drug_descriptors | drug_fingerprint | protein_AAC | protein_kmer_hash]


Feature names are preserved for interpretability and importance analysis.

---

## 🧠 Machine Learning Models

Implemented models:

- Logistic Regression (baseline)
- Random Forest
- XGBoost
- Multi-Layer Perceptron (MLP)

### Handling Class Imbalance

- `class_weight` (LogReg & RF)  
- `scale_pos_weight` (XGBoost)  
- sample weighting (MLP)

---

## 📊 Scientific & Engineering Best Practices

✔ Invalid SMILES & sequences detected and reported  
✔ Leakage-safe scaling using sklearn pipelines  
✔ Stratified cross-validation  
✔ Reproducible training with fixed seeds  
✔ Feature importance reporting  
✔ Modular & reusable pipeline design  

---

## 📁 Project Structure

dti/
├── io.py # dataset loading & validation
├── utils.py
├── preprocess.py # scaling logic
├── models.py # model factory & imbalance handling
├── eval.py # metrics & cross-validation
└── features/
├── drug.py # RDKit descriptors & fingerprints
├── protein.py # AAC & k-mer encoding
└── dti.py # feature concatenation

train.py # CLI training & evaluation
api.py # FastAPI inference service
examples/ # tiny dataset & smoke test


---

## 📊 Results (KIBA Dataset)

- ROC-AUC ≈ **0.96**  
- Accuracy ≈ **89%**  
- Balanced classification performance  

---

## 🌐 API Inference

Start the API:

```bash
uvicorn api:app --reload
Example Request
POST /predict

{
  "compound_iso_smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
  "target_sequence": "MVKVYAPASSANMSVGFDVLGAAVTPVDGALLGDVVTVEAAETFSLNNLGQKLTKELGADVVV"
}
Example Response
{
  "probability": 0.91,
  "label": 1
}
⚙️ Setup
✅ Recommended (Windows): Conda + RDKit
conda env create -f environment.yml
conda activate dti
Alternative (pip)
pip install -r requirements.txt
RDKit installation via pip may fail on Windows.

📄 Data Format
Input CSV must contain:

compound_iso_smiles

target_sequence

label (0/1)

Optional columns (e.g., affinity) are supported.

🏋️ Train & Evaluate
python train.py --data path\to\data.csv --outdir runs\logreg --model logreg
python train.py --data path\to\data.csv --outdir runs\rf --model rf
python train.py --data path\to\data.csv --outdir runs\xgb --model xgb
Useful Options
--cv 5 → cross validation folds

--seed 42 → reproducibility

--drop_invalid → remove invalid inputs

--kmer_k 3 --kmer_dim 1024 → protein encoding settings

📦 Model Artifacts
Artifacts saved to --outdir:

model.joblib → full pipeline

report.json → metrics & dataset stats

feature_importance.csv → top features

roc_curve_oof.png → ROC curve

Model artifacts are not included in this repository due to size.

🔁 Reproducibility
To reproduce results:

python train.py --data your_dataset.csv --model rf --cv 5
🧪 Example Dataset
A small dataset for testing:

examples/tiny_dti.csv
