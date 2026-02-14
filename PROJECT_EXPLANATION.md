# Complete Educational Explanation: Drug–Target Interaction Prediction Project

## Table of Contents
1. [Big Picture Overview](#1-big-picture-overview)
2. [Dataset Explanation](#2-dataset-explanation)
3. [Feature Engineering (Most Important Part)](#3-feature-engineering-most-important-part)
4. [Project Structure](#4-project-structure-folder-by-folder-file-by-file)
5. [Machine Learning Pipeline](#5-machine-learning-pipeline)
6. [Models Used](#6-models-used)
7. [Evaluation and Results Interpretation](#7-evaluation-and-results-interpretation)
8. [Prediction Capability](#8-prediction-capability)
9. [Limitations and Assumptions](#9-limitations-and-assumptions)
10. [Final Summary](#10-final-summary)

---

## 1. Big Picture Overview

### What Problem Does This Project Solve?

**The Core Question:** Given a drug molecule (represented as a SMILES string) and a protein target (represented as an amino acid sequence), can we predict whether they will interact?

This is a fundamental problem in **computational drug discovery** and **bioinformatics**. In pharmaceutical research, discovering that a drug binds to a specific protein target is expensive and time-consuming. If we can predict interactions computationally before wet-lab experiments, we can:

- **Accelerate drug discovery**: Screen millions of drug–target pairs computationally
- **Reduce costs**: Avoid expensive experimental tests for unlikely interactions
- **Understand mechanisms**: Learn what molecular properties drive binding

### Classification vs. Regression

This project solves a **binary classification problem**:

- **Class 0 (Negative)**: No interaction between drug and target
- **Class 1 (Positive)**: Interaction exists between drug and target

**Why classification instead of regression?**

The original dataset (like KIBA) often contains **affinity values** (e.g., IC50, Ki, Kd) that measure binding strength. However, converting to binary classification is common in DTI prediction because:

1. **Practical utility**: In early drug discovery, you often need a yes/no answer: "Should we test this pair experimentally?"
2. **Data availability**: Many datasets have missing affinity values but have interaction labels
3. **Interpretability**: Binary predictions are easier to explain to domain experts
4. **Baseline comparison**: Classification metrics (ROC-AUC, F1) are standard in the field

**Note**: The codebase is designed so that future regression experiments (predicting actual affinity values) can be added easily by swapping the classifier for a regressor.

### Relationship to KIBA Dataset

The **KIBA dataset** (Kinase Inhibitor Bioactivity) is a benchmark dataset in DTI prediction. It contains:
- Drug molecules (as SMILES)
- Protein targets (kinases, as sequences)
- Binding affinity scores (KIBA scores, which integrate multiple affinity measurements)

**How this project relates:**

- **Input format**: Our project expects the same raw inputs (SMILES + sequences) as KIBA
- **Label creation**: If you have KIBA affinity values, you typically convert them to binary labels using a threshold (e.g., median affinity: above = interaction, below = no interaction)
- **Evaluation**: We use the same metrics (ROC-AUC, F1) that are standard in KIBA benchmark papers

**The key difference**: This project implements **feature extraction from scratch**. Many papers use pre-computed features or deep learning embeddings. Here, we explicitly compute molecular descriptors and fingerprints ourselves, making the pipeline transparent and reproducible.

### What Question Does the Model Answer?

The trained model answers: **"Given a drug molecule and a protein target, what is the probability that they interact?"**

The output is a probability score between 0 and 1:
- **Close to 1.0**: High confidence that interaction exists
- **Close to 0.0**: High confidence that no interaction exists
- **Around 0.5**: Uncertain prediction

This probability can be thresholded (default: 0.5) to make a binary decision, but keeping the probability allows for ranking predictions by confidence.

---

## 2. Dataset Explanation

### What Does Each Row Represent?

Each row in your CSV dataset represents **one drug–target pair** and whether they interact.

**Example row:**
```
compound_iso_smiles: "CC(=O)OC1=CC=CC=C1C(=O)O"
target_sequence: "MVKVYAPASSANMSVGFDVLGAAVTPVDGALLGDVVTVEAAETFSLNNLGQKLTKELGADVVV"
label: 1
```

This means: "The drug aspirin (represented by that SMILES string) interacts with the protein (represented by that amino acid sequence)."

### Understanding Each Column

#### `compound_iso_smiles`

**What it is**: SMILES (Simplified Molecular Input Line Entry System) is a text representation of a molecule's structure.

**Example**: `"CC(=O)OC1=CC=CC=C1C(=O)O"` represents aspirin.

**How to read it**:
- `C` = carbon atom
- `(=O)` = double-bonded oxygen
- `1` = ring closure (connects atoms in a ring)
- Numbers indicate ring connections

**Why SMILES?**
- Standard format in cheminformatics
- Compact (one line per molecule)
- Can be parsed by tools like RDKit to reconstruct the 3D structure
- Contains all information needed to compute molecular properties

**The "iso" prefix**: Indicates this is an "isomeric" SMILES, meaning it includes stereochemistry information (which way atoms are oriented in 3D space). This is important for drug activity.

#### `target_sequence`

**What it is**: A string of letters representing the amino acid sequence of a protein.

**Example**: `"MVKVYAPASSANMSVGFDVLGAAVTPVDGALLGDVVTVEAAETFSLNNLGQKLTKELGADVVV"`

**How to read it**:
- Each letter represents one amino acid
- `M` = Methionine, `V` = Valine, `K` = Lysine, etc.
- There are 20 canonical amino acids: `ACDEFGHIKLMNPQRSTVWY`
- The sequence order matters: it determines the protein's 3D structure and function

**Why sequences?**
- Proteins are too complex to represent as 3D structures computationally
- The sequence determines everything: structure, function, binding sites
- Sequences are easy to work with computationally (just strings)
- Many databases (UniProt) provide sequences for millions of proteins

#### `affinity` (Optional Column)

**What it is**: A numerical value measuring how strongly the drug binds to the target.

**Common units**:
- **IC50**: Concentration needed to inhibit 50% of activity (lower = stronger binding)
- **Ki**: Inhibition constant (lower = stronger binding)
- **Kd**: Dissociation constant (lower = stronger binding)
- **KIBA score**: Normalized score combining multiple affinity measurements

**Example**: `affinity = 7.2` might mean IC50 = 10^7.2 nM (very weak binding) or KIBA score = 7.2 (moderate binding).

**Why it's optional**: This project focuses on binary classification. If you have affinity values, you can use them to create labels, but the model doesn't need them during training.

#### `label` (Required)

**What it is**: Binary indicator (0 or 1) indicating whether an interaction exists.

- **1**: Interaction exists (positive class)
- **0**: No interaction (negative class)

### Converting Affinity to Binary Labels

**The Process** (typically done before training):

1. **Load affinity values** from your dataset
2. **Choose a threshold**: Often the median affinity value
   ```python
   threshold = df['affinity'].median()  # Example: 7.2
   ```
3. **Create labels**:
   ```python
   df['label'] = (df['affinity'] >= threshold).astype(int)
   # Above threshold = interaction (1)
   # Below threshold = no interaction (0)
   ```

**Why median threshold?**

- **Balanced classes**: Median ensures roughly equal numbers of positives and negatives
- **Standard practice**: Common in DTI literature
- **Interpretability**: "Above average binding strength = interaction"

**Implications**:

- **What "interaction" means**: In this project, "interaction" means "binding strength above the dataset median." This is a **relative definition**, not an absolute biological truth.
- **Dataset-dependent**: A drug–target pair labeled "1" in one dataset might be "0" in another if the median differs
- **Future work**: Regression models can predict actual affinity values, avoiding this thresholding step

---

## 3. Feature Engineering (Most Important Part)

This is the **core innovation** of your project: converting raw text (SMILES strings and protein sequences) into numerical features that machine learning models can understand.

### The Challenge

Machine learning models need **numbers**, not text. We must convert:
- `"CC(=O)OC1=CC=CC=C1C(=O)O"` → array of numbers
- `"MVKVYAPASSANMSVGFDVLGAAVTPVDGALLGDVVTVEAAETFSLNNLGQKLTKELGADVVV"` → array of numbers

### How Drugs Are Represented

We use **two complementary representations** that capture different aspects of molecular structure:

#### A. Molecular Descriptors (14 features)

**What they are**: Computed properties that describe the **entire molecule** as a whole.

**Implementation**: In `dti/features/drug.py`, we compute 14 descriptors using RDKit:

1. **MolWt** (Molecular Weight): Total mass of the molecule
   - **Why it matters**: Larger molecules may have different binding properties
   - **Example**: Aspirin ≈ 180 g/mol, large drug ≈ 500 g/mol

2. **MolLogP** (Logarithm of Partition Coefficient): Measures lipophilicity (how well the molecule dissolves in fat vs. water)
   - **Why it matters**: Determines if a drug can cross cell membranes
   - **Example**: LogP > 0 = hydrophobic (fat-loving), LogP < 0 = hydrophilic (water-loving)

3. **MolMR** (Molar Refractivity): Related to molecular volume and polarizability
   - **Why it matters**: Affects how molecules interact with proteins

4. **TPSA** (Topological Polar Surface Area): Sum of polar atoms' surface areas
   - **Why it matters**: Predicts membrane permeability (important for oral drugs)
   - **Rule of thumb**: TPSA < 140 Å² = good oral bioavailability

5. **NumHDonors** (Number of Hydrogen Bond Donors): Count of N-H and O-H groups
   - **Why it matters**: Hydrogen bonds are crucial for drug–protein binding

6. **NumHAcceptors** (Number of Hydrogen Bond Acceptors): Count of N and O atoms that can accept H-bonds
   - **Why it matters**: More acceptors = more potential binding interactions

7. **NumRotatableBonds** (Rotatable Bonds): Bonds that can rotate freely
   - **Why it matters**: Flexibility affects how well a drug fits into a protein pocket

8. **NumRings** (Total Ring Count): Number of cyclic structures
   - **Why it matters**: Rings provide structural rigidity

9. **NumAromaticRings**: Count of aromatic (benzene-like) rings
   - **Why it matters**: Aromatic rings often participate in π-π stacking interactions with proteins

10. **NumAliphaticRings**: Non-aromatic rings
    - **Why it matters**: Different binding properties than aromatic rings

11. **HeavyAtomCount**: Atoms other than hydrogen
    - **Why it matters**: Size indicator

12. **FractionCSP3**: Fraction of carbon atoms that are sp³ hybridized (tetrahedral)
    - **Why it matters**: Measures "flatness" vs. "3D-ness" of the molecule

13. **BertzCT** (Bertz Complexity Index): Measures molecular complexity
    - **Why it matters**: More complex molecules may have more specific interactions

14. **LabuteASA** (Labute Approximate Surface Area): Estimated molecular surface area
    - **Why it matters**: Surface area affects binding

**Why descriptors?**

- **Interpretable**: Each number has a clear chemical meaning
- **Low-dimensional**: Only 14 features (easy to visualize and understand)
- **Domain knowledge**: These properties are known to correlate with drug activity
- **Fast to compute**: RDKit computes them instantly

**Limitation**: Descriptors capture **global properties** but miss **local substructures** (specific chemical groups that might be important for binding).

#### B. Morgan Fingerprints (2048 binary features)

**What they are**: Binary vectors encoding **local chemical environments** (substructures) present in the molecule.

**Conceptual explanation**:

Imagine you're describing a molecule by listing all the **small chemical patterns** it contains:

- "Has a benzene ring"
- "Has a carboxyl group (-COOH)"
- "Has a methyl group (-CH3) attached to an oxygen"

A Morgan fingerprint does this systematically by:

1. **Starting at each atom** in the molecule
2. **Looking in a radius** (default: radius=2, meaning 2 bonds away)
3. **Encoding the chemical environment** around that atom as a bit
4. **Hashing** the environment to a fixed position in a 2048-bit vector

**Example**:

For aspirin `CC(=O)OC1=CC=CC=C1C(=O)O`:
- Bit 42 might be set to 1 if there's a "carboxyl group" pattern
- Bit 156 might be set to 1 if there's a "benzene ring" pattern
- Bit 892 might be set to 1 if there's an "ester group" pattern
- Most bits remain 0 (the molecule doesn't contain those patterns)

**Why fingerprints?**

- **Captures local patterns**: Identifies specific chemical groups that might bind to proteins
- **Standard in cheminformatics**: ECFP (Extended Connectivity Fingerprints) are the gold standard
- **Fixed length**: Always 2048 bits, regardless of molecule size (good for ML)
- **Binary**: Each bit is 0 or 1 (simple, fast)

**Limitation**: Fingerprints are **not interpretable**—you can't easily say "bit 42 means carboxyl group" without additional analysis.

**Why both descriptors AND fingerprints?**

- **Descriptors**: Global, interpretable properties (e.g., "this molecule is hydrophobic")
- **Fingerprints**: Local, pattern-based features (e.g., "this molecule contains a specific binding motif")

Together, they provide **complementary information** that helps the model learn both global and local patterns.

### How Proteins Are Represented

We use **two complementary representations** for proteins:

#### A. Amino Acid Composition (AAC) (20 features)

**What it is**: A 20-dimensional vector where each element is the **fraction** of that amino acid in the sequence.

**Example**:

Sequence: `"ACDEFGHIKLMNPQRSTVWY"` (contains all 20 amino acids once, length=20)

AAC vector:
```
prot_aac_A: 0.05  (1/20 = 5% Alanine)
prot_aac_C: 0.05  (1/20 = 5% Cysteine)
prot_aac_D: 0.05  (1/20 = 5% Aspartic acid)
...
prot_aac_Y: 0.05  (1/20 = 5% Tyrosine)
```

**Why AAC?**

- **Interpretable**: "This protein is 30% hydrophobic amino acids" is meaningful
- **Global property**: Captures overall composition (e.g., hydrophobic vs. hydrophilic)
- **Fast to compute**: Just count and divide
- **Low-dimensional**: Only 20 features

**Limitation**: AAC **ignores order**. The sequences `"AAA"` and `"AA"` have different AAC, but `"AAA"` and `"AAA"` (same sequence) have identical AAC. More importantly, `"ACDEFG"` and `"GFEDCA"` (reversed) have identical AAC but might fold differently.

#### B. Hashed k-mer Counts (1024 features)

**What k-mers are**: Overlapping subsequences of length k.

**Example** (k=3, called "3-mers" or "trigrams"):

Sequence: `"ACDEFG"`

3-mers:
- `"ACD"` (positions 0-2)
- `"CDE"` (positions 1-3)
- `"DEF"` (positions 2-4)
- `"EFG"` (positions 3-5)

**Why k-mers?**

- **Captures local motifs**: Specific amino acid patterns (e.g., `"GXXG"` is a common binding motif)
- **Order-sensitive**: Unlike AAC, k-mers preserve sequence order
- **Biological relevance**: Many protein functions depend on short sequence motifs

**The Problem**: There are 20^k possible k-mers. For k=3, that's 20³ = 8,000 possible 3-mers. For k=4, that's 160,000. This **explodes** with k.

**Solution: Feature Hashing**

Instead of creating a feature for each possible k-mer, we use a **hashing trick**:

1. **Slide a window** of length k across the sequence
2. **Hash each k-mer** to an index between 0 and 1023 (using a stable hash function)
3. **Count** how many times each index is hit (with signed hashing to reduce collision bias)
4. **Normalize** by sequence length

**Example**:

Sequence: `"ACDEFG"`, k=3, dim=1024

1. Extract 3-mers: `["ACD", "CDE", "DEF", "EFG"]`
2. Hash each:
   - `"ACD"` → hash → index 42 → increment vector[42]
   - `"CDE"` → hash → index 156 → increment vector[156]
   - `"DEF"` → hash → index 892 → increment vector[892]
   - `"EFG"` → hash → index 42 → increment vector[42] (collision!)
3. Normalize: Divide by 4 (number of k-mers)

**Why hashing?**

- **Fixed dimensionality**: Always 1024 features, regardless of sequence length or k
- **Memory efficient**: Don't need to store 20^k features
- **Fast**: Hashing is O(1) per k-mer
- **Collision handling**: Signed hashing (+1/-1) reduces bias from collisions

**Limitation**: Hash collisions mean different k-mers map to the same feature. This is acceptable because:
- We use a large dimension (1024) to minimize collisions
- Signed hashing cancels out some collision bias
- The model can still learn patterns despite collisions

**Why k=3?**

- **Balance**: k=2 (400 possible) is too small (loses information), k=4 (160,000 possible) is too large (sparse)
- **Standard**: k=3 is common in bioinformatics (captures most short motifs)
- **Computational**: 1024 dimensions is manageable for ML models

### Concatenating Drug and Protein Features

**The Final Feature Vector**:

After featurization, each drug–target pair becomes a single vector:

```
[drug_descriptors (14) | drug_fingerprint (2048) | protein_aac (20) | protein_kmerhash (1024)]
```

**Total**: 14 + 2048 + 20 + 1024 = **3,106 features**

**Why concatenate?**

- **Single vector per pair**: ML models expect one feature vector per sample
- **Preserves all information**: Both drug and protein features are available to the model
- **Model learns interactions**: The model can learn that "high LogP + high hydrophobic AAC = likely interaction"

**Biological Interpretation**:

The final feature vector represents:
- **Drug properties**: Size, polarity, flexibility, chemical patterns
- **Protein properties**: Composition, local sequence motifs
- **Potential interactions**: The model learns which drug properties match which protein properties

**Computational Interpretation**:

- **High-dimensional**: 3,106 features is large but manageable for modern ML
- **Mixed types**: Continuous (descriptors, AAC) and binary (fingerprints)
- **Sparse**: Fingerprints are mostly zeros (only ~10-50 bits set per molecule)

---

## 4. Project Structure (Folder-by-Folder, File-by-File)

### Root Directory: `dti_project/`

**Purpose**: Contains the entire project, including code, configuration, examples, and documentation.

#### `train.py` (Main Entry Point)

**Responsibility**: Command-line interface (CLI) that orchestrates the entire pipeline.

**What it does**:
1. **Parses command-line arguments** (data path, model type, CV folds, etc.)
2. **Loads the dataset** via `load_dti_csv()`
3. **Builds the sklearn Pipeline** (featurization → scaling → classifier)
4. **Runs cross-validation** (if requested)
5. **Trains final model** on all data
6. **Saves artifacts**: `model.joblib`, `report.json`, `feature_importance.csv`, `roc_curve_oof.png`

**Why it's separate**: 
- **Separation of concerns**: Orchestration logic is separate from feature extraction logic
- **CLI usability**: Easy to run from command line: `python train.py --data X --model Y`
- **Reproducibility**: All hyperparameters are command-line arguments (can be logged/versioned)

**Key functions**:
- `build_pipeline()`: Creates the sklearn Pipeline with featurizer, scaler, and classifier
- `replace_classifier()`: Swaps in the chosen model (LogReg/RF/XGB/MLP)
- `main()`: Entry point that runs everything

#### `README.md`

**Responsibility**: Documentation explaining how to set up and run the project.

**Why it's important**: 
- **Onboarding**: New users (or you in 6 months) can understand the project quickly
- **Reproducibility**: Documents dependencies and setup steps
- **Scientific communication**: Explains design choices (why these features, why these models)

#### `requirements.txt` and `environment.yml`

**Responsibility**: Dependency management.

- `requirements.txt`: pip dependencies (may fail for RDKit on Windows)
- `environment.yml`: Conda environment (recommended, includes RDKit from conda-forge)

**Why both**: 
- **Flexibility**: Users can choose pip or conda
- **Windows compatibility**: Conda is more reliable for RDKit on Windows

### `dti/` Package (Core Implementation)

**Purpose**: Python package containing all the ML logic, organized into modules.

#### `dti/__init__.py`

**Responsibility**: Package initialization, exports public API.

**Why it exists**: Makes `dti` a proper Python package (can be imported as `from dti.io import ...`).

#### `dti/utils.py`

**Responsibility**: Utility functions used across the project.

**Key functions**:
- `set_global_seed()`: Sets random seeds for reproducibility (numpy, Python random, hash seed)
- `DatasetStats`: Dataclass storing dataset statistics (n_rows, n_invalid, etc.)
- `safe_div()`: Safe division (handles divide-by-zero)

**Why separate**: 
- **Reusability**: Used by multiple modules
- **Organization**: Keeps utility code separate from domain logic

#### `dti/io.py` (Data Loading)

**Responsibility**: Loading and validating CSV datasets.

**Key functions**:
- `load_dti_csv()`: Main function that:
  1. Reads CSV file
  2. Validates required columns exist
  3. Checks SMILES validity (via `is_valid_smiles()`)
  4. Checks protein sequence validity (via `is_valid_protein_sequence()`)
  5. Optionally drops invalid rows (`drop_invalid=True`)
  6. Returns cleaned DataFrame and statistics

**Why this module**:
- **Single responsibility**: All I/O logic in one place
- **Validation**: Ensures data quality before featurization
- **Error handling**: Clear error messages if CSV is malformed
- **Statistics**: Reports how many rows were dropped (important for scientific reporting)

**Design choice**: Returns both DataFrame and `DatasetStats` so the caller knows what happened (e.g., "1000 rows loaded, 50 invalid SMILES dropped").

#### `dti/features/` Package (Feature Extraction)

**Purpose**: Contains all feature extraction logic, organized by domain (drug vs. protein).

##### `dti/features/drug.py`

**Responsibility**: Featurizing drug molecules from SMILES strings.

**Key functions**:
- `is_valid_smiles()`: Validates SMILES can be parsed by RDKit
- `compute_drug_descriptors()`: Computes 14 molecular descriptors
- `morgan_fingerprint_bits()`: Computes Morgan fingerprint (2048 bits)
- `featurize_smiles()`: Main function that returns (descriptors, fingerprint, is_valid)
- `get_drug_descriptor_names()`: Returns list of descriptor names (for feature importance)

**Why separate module**:
- **Domain separation**: Drug featurization is complex and self-contained
- **RDKit dependency**: Isolates RDKit import (can fail gracefully if not installed)
- **Reusability**: Can be used independently (e.g., "just featurize this SMILES")

**Design choices**:
- **Returns validity flag**: Allows caller to handle invalid SMILES gracefully
- **Feature names**: Exported so feature importance can be interpreted
- **Configurable**: `DrugFeaturizerConfig` allows tuning (radius, n_bits, chirality)

##### `dti/features/protein.py`

**Responsibility**: Featurizing protein sequences.

**Key functions**:
- `is_valid_protein_sequence()`: Validates sequence contains only canonical amino acids
- `amino_acid_composition()`: Computes 20-dim AAC vector
- `hashed_kmer_counts()`: Computes hashed k-mer vector (1024 dim)
- `featurize_protein()`: Main function that returns (features, is_valid)
- `get_protein_feature_names()`: Returns feature names (AAC + k-mer hash indices)

**Why separate module**:
- **Domain separation**: Protein featurization is independent of drug featurization
- **No external dependencies**: Pure Python (no RDKit needed)
- **Reusability**: Can featurize proteins independently

**Design choices**:
- **Stable hashing**: Uses `blake2b` for reproducible hashing (same k-mer → same index across runs)
- **Signed hashing**: Reduces collision bias (some k-mers increment, some decrement)
- **Length normalization**: Divides by number of k-mers so values are comparable across proteins of different lengths

##### `dti/features/dti.py` (Feature Concatenation)

**Responsibility**: Combines drug and protein features into a single vector.

**Key class**: `DTIFeaturizer` (sklearn Transformer)

**What it does**:
1. Takes DataFrame with `compound_iso_smiles` and `target_sequence` columns
2. Calls `featurize_smiles()` for each drug
3. Calls `featurize_protein()` for each protein
4. Concatenates: `[drug_desc | drug_fp | protein_aac | protein_kmerhash]`
5. Returns numpy array of shape `(n_samples, 3106)`

**Why sklearn Transformer**:
- **Pipeline compatibility**: Can be used in sklearn `Pipeline`
- **Fit/transform API**: Standard interface (even though `fit()` does nothing)
- **Feature names**: Implements `get_feature_names_out()` for interpretability

**Design choices**:
- **Preserves feature order**: Always `[drug | protein]` (important for scaling mask)
- **Exposes slices**: `fingerprint_slice_` property tells scaler which columns are binary
- **Handles invalid inputs**: Invalid SMILES/sequences become all-zeros (graceful degradation)

#### `dti/preprocess.py` (Scaling)

**Responsibility**: Scaling continuous features while leaving binary features untouched.

**Key class**: `SelectiveStandardScaler` (sklearn Transformer)

**What it does**:
1. Takes a boolean mask indicating which columns are continuous
2. Fits `StandardScaler` only on continuous columns
3. Transforms: continuous columns are standardized (mean=0, std=1), binary columns unchanged

**Why this is critical**:

**Problem**: If you scale binary fingerprint bits (0/1), they become non-binary (e.g., 0.3, -0.7), which:
- **Loses interpretability**: "Bit is set" no longer means "value = 1"
- **Changes model behavior**: Some models (like tree-based) work better with binary features
- **Unnecessary**: Binary features don't need scaling (they're already normalized)

**Solution**: Only scale continuous features (descriptors, AAC, k-mers).

**Design choices**:
- **Custom transformer**: sklearn's `StandardScaler` scales everything; we need selective scaling
- **Preserves column order**: Doesn't reorder features (important for feature names)
- **Leakage-safe**: Fits inside CV folds (via Pipeline), so scaler statistics come only from training data

**Function**: `continuous_mask_excluding_fingerprint()`: Creates mask where fingerprint bits are `False` (don't scale) and everything else is `True` (scale).

#### `dti/models.py` (Model Factory)

**Responsibility**: Creates ML models with appropriate hyperparameters and class imbalance handling.

**Key functions**:
- `make_logreg()`: Creates Logistic Regression with `class_weight="balanced"`
- `make_rf()`: Creates Random Forest with `class_weight="balanced_subsample"`
- `make_xgb()`: Creates XGBoost with `scale_pos_weight` (computed from label imbalance)
- `make_mlp()`: Creates MLP with sample weights (computed separately)
- `make_estimator()`: Factory function that creates the chosen model
- `maybe_sample_weight_for_estimator()`: Returns sample weights for models that need them (MLP)

**Why this module**:
- **Centralized model creation**: All model hyperparameters in one place
- **Class imbalance handling**: Each model uses appropriate technique:
  - LogReg/RF: `class_weight` parameter
  - XGBoost: `scale_pos_weight` parameter
  - MLP: `sample_weight` argument (passed during `fit()`)
- **Reproducibility**: Fixed random seeds, documented hyperparameters

**Design choices**:
- **SAGA solver for LogReg**: Handles large feature spaces well (3,106 features)
- **Balanced class weights**: Prevents models from always predicting the majority class
- **XGBoost lazy import**: Only imports if needed (allows project to run without XGBoost)

#### `dti/eval.py` (Evaluation)

**Responsibility**: Cross-validation, metrics computation, ROC curves, feature importance extraction.

**Key functions**:
- `compute_metrics()`: Computes Accuracy, Precision, Recall, F1, ROC-AUC for a set of predictions
- `cross_validate_binary_classifier()`: Runs stratified K-fold CV, returns metrics per fold
- `out_of_fold_pred_proba()`: Computes out-of-fold predictions for ROC curve (leakage-safe)
- `save_roc_curve_plot()`: Saves ROC curve as PNG image
- `extract_feature_importance()`: Extracts feature importance from trained model (if available)
- `summarize_folds()`: Aggregates fold metrics (mean ± std)

**Why this module**:
- **Evaluation logic**: All metrics and CV code in one place
- **Leakage prevention**: CV functions clone pipelines to ensure no data leakage
- **Reporting**: Generates JSON reports and plots for scientific communication

**Design choices**:
- **Stratified CV**: Ensures each fold has same class distribution as full dataset
- **Out-of-fold predictions**: ROC curve uses predictions from models that didn't train on those samples (prevents overfitting in visualization)
- **Feature importance**: Extracts from models that support it (LogReg: |coefficients|, RF/XGB: `feature_importances_`)

### `examples/` Directory

**Purpose**: Example data and scripts for testing.

#### `examples/tiny_dti.csv`

**Responsibility**: Small example dataset (4 rows) for smoke testing.

**Why it exists**: 
- **Quick testing**: Verify the pipeline works without downloading large datasets
- **Documentation**: Shows expected CSV format
- **CI/CD**: Can be used in automated tests

#### `examples/smoke_test.py`

**Responsibility**: Script that runs the pipeline on `tiny_dti.csv`.

**Why it exists**: 
- **Verification**: Ensures everything works end-to-end
- **Onboarding**: New users can run this to verify setup

### `runs/` Directory (Generated)

**Purpose**: Output directory where trained models and reports are saved.

**Structure** (created by `train.py`):
```
runs/
  baseline_logreg/
    model.joblib          # Trained pipeline
    report.json           # Metrics and statistics
    feature_importance.csv  # Top features (if available)
    roc_curve_oof.png     # ROC plot (if --save_roc)
  baseline_rf/
    ...
  advanced_xgb/
    ...
```

**Why separate directory**: 
- **Organization**: Keeps outputs separate from code
- **Versioning**: Can compare runs (e.g., `runs/exp1/` vs `runs/exp2/`)
- **Reproducibility**: Each run is self-contained

---

## 5. Machine Learning Pipeline

This section explains the **complete flow** from raw CSV to trained model, step-by-step.

### Step 1: Data Loading and Validation (`dti/io.py`)

**Input**: CSV file path

**Process**:
1. **Read CSV**: `pd.read_csv(path)`
2. **Validate columns**: Check that `compound_iso_smiles`, `target_sequence`, `label` exist
3. **Type conversion**: Convert SMILES and sequences to strings
4. **Label validation**: Ensure labels are binary (0/1)
5. **SMILES validation**: Call `is_valid_smiles()` for each SMILES (uses RDKit)
6. **Sequence validation**: Call `is_valid_protein_sequence()` for each sequence
7. **Filter invalid** (if `drop_invalid=True`): Remove rows with invalid SMILES or sequences
8. **Compute statistics**: Count valid/invalid rows, positive/negative labels

**Output**: 
- Cleaned DataFrame (only valid rows)
- `DatasetStats` object (for reporting)

**Why validation matters**: 
- **Prevents errors**: Invalid SMILES would crash RDKit during featurization
- **Data quality**: Reports how much data was lost (important for scientific papers)
- **Reproducibility**: Same validation logic ensures consistent results

### Step 2: Feature Extraction (`dti/features/dti.py`)

**Input**: DataFrame with `compound_iso_smiles` and `target_sequence` columns

**Process** (inside `DTIFeaturizer.transform()`):

For each row:
1. **Extract SMILES**: `"CC(=O)OC1=CC=CC=C1C(=O)O"`
2. **Extract sequence**: `"MVKVYAPASSANMSVGFDVLGAAVTPVDGALLGDVVTVEAAETFSLNNLGQKLTKELGADVVV"`
3. **Featurize drug** (`dti/features/drug.py`):
   - Parse SMILES → RDKit Mol object
   - Compute descriptors: `[180.16, 1.19, 45.2, ...]` (14 values)
   - Compute fingerprint: `[0, 1, 0, 0, 1, ...]` (2048 bits, mostly zeros)
4. **Featurize protein** (`dti/features/protein.py`):
   - Compute AAC: `[0.05, 0.10, 0.15, ...]` (20 values, sum to 1.0)
   - Compute k-mer hash: `[0.02, -0.01, 0.03, ...]` (1024 values)
5. **Concatenate**: `[drug_desc | drug_fp | protein_aac | protein_kmerhash]`

**Output**: NumPy array of shape `(n_samples, 3106)`

**Why Pipeline**: 
- **Leakage-safe**: Featurization happens inside CV folds (no test data used for fitting)
- **Reproducible**: Same featurization logic for train and test

### Step 3: Scaling (`dti/preprocess.py`)

**Input**: Feature matrix from Step 2

**Process** (inside `SelectiveStandardScaler.transform()`):

1. **Identify continuous columns**: 
   - Descriptors (columns 0-13): Continuous → scale
   - Fingerprint (columns 14-2061): Binary → don't scale
   - AAC (columns 2062-2081): Continuous → scale
   - K-mers (columns 2082-3105): Continuous → scale

2. **Fit scaler** (on training data only, inside CV):
   - Compute mean and std for continuous columns
   - Store these statistics

3. **Transform** (apply to train and test):
   - Continuous columns: `(value - mean) / std` → standardized (mean=0, std=1)
   - Binary columns: unchanged (still 0/1)

**Output**: Scaled feature matrix (same shape, continuous columns standardized)

**Why scaling matters**:
- **Feature magnitude**: Descriptors have different scales (MolWt ≈ 500, LogP ≈ 2)
- **Model convergence**: Many models (LogReg, MLP) converge faster with scaled features
- **Feature importance**: Without scaling, large-magnitude features dominate

**Why NOT scale fingerprints**:
- **Binary nature**: 0/1 is already normalized
- **Interpretability**: "Bit is set" should mean "value = 1"
- **Model behavior**: Tree-based models (RF, XGB) don't need scaling for binary features

### Step 4: Model Training (`dti/models.py` + sklearn)

**Input**: Scaled features (X) and labels (y)

**Process** (inside `Pipeline.fit()`):

1. **Create model** (via `make_estimator()`):
   - LogReg: `LogisticRegression(class_weight="balanced", ...)`
   - RF: `RandomForestClassifier(class_weight="balanced_subsample", ...)`
   - XGBoost: `XGBClassifier(scale_pos_weight=ratio, ...)`
   - MLP: `MLPClassifier(...)` + sample weights

2. **Handle class imbalance**:
   - LogReg/RF: `class_weight` automatically upweights minority class
   - XGBoost: `scale_pos_weight = n_neg / n_pos` (e.g., if 70% negative, weight = 0.7/0.3 = 2.33)
   - MLP: `sample_weight` array passed to `fit()` (computed via `compute_sample_weight()`)

3. **Fit model**: `model.fit(X_scaled, y, sample_weight=sample_weight)`

**Output**: Trained model (stored in Pipeline's `clf` step)

**Why class imbalance handling**:
- **Prevents bias**: Without it, models predict majority class (e.g., always predict "no interaction")
- **Balanced learning**: Model learns from both classes equally
- **Better metrics**: Improves recall for minority class (important in drug discovery: missing a real interaction is costly)

### Step 5: Cross-Validation (`dti/eval.py`)

**Input**: Pipeline, features (X), labels (y), number of folds (default: 5)

**Process** (inside `cross_validate_binary_classifier()`):

1. **Create stratified K-fold splitter**: Ensures each fold has same class distribution
2. **For each fold**:
   - **Split data**: `train_idx, test_idx = skf.split(X, y)`
   - **Clone pipeline**: `est = clone(pipeline)` (fresh model, no leakage)
   - **Fit on train**: `est.fit(X[train_idx], y[train_idx])`
   - **Predict on test**: `y_proba = est.predict_proba(X[test_idx])`
   - **Compute metrics**: Accuracy, Precision, Recall, F1, ROC-AUC
3. **Aggregate**: Compute mean ± std across folds

**Output**: 
- List of `FoldMetrics` (one per fold)
- Summary dictionary with `accuracy_mean`, `accuracy_std`, etc.

**Why cross-validation**:
- **Estimates generalization**: How well will the model perform on unseen data?
- **Prevents overfitting detection**: If train accuracy >> CV accuracy, model is overfitting
- **Hyperparameter tuning**: Can compare different models/configs using CV scores

**Why stratified**:
- **Class balance**: Each fold has same proportion of positives/negatives as full dataset
- **Stable estimates**: Prevents one fold from having all negatives (would give meaningless metrics)

**Why clone pipeline**:
- **Leakage prevention**: Each fold gets a fresh scaler (fitted only on train data)
- **Reproducibility**: Same process for each fold

### Step 6: Evaluation Metrics (`dti/eval.py`)

**Metrics computed** (inside `compute_metrics()`):

1. **Accuracy**: `(TP + TN) / (TP + TN + FP + FN)`
   - **Meaning**: Fraction of correct predictions
   - **Limitation**: Misleading with class imbalance (e.g., 90% accuracy if model always predicts majority class)

2. **Precision**: `TP / (TP + FP)`
   - **Meaning**: Of all predictions labeled "interaction", how many are correct?
   - **Interpretation**: High precision = low false positive rate (good for avoiding wasted experiments)

3. **Recall**: `TP / (TP + FN)`
   - **Meaning**: Of all true interactions, how many did we find?
   - **Interpretation**: High recall = low false negative rate (good for not missing real interactions)

4. **F1-score**: `2 * (Precision * Recall) / (Precision + Recall)`
   - **Meaning**: Harmonic mean of precision and recall
   - **Interpretation**: Balanced metric (penalizes models that optimize one at the expense of the other)

5. **ROC-AUC**: Area under the Receiver Operating Characteristic curve
   - **Meaning**: Probability that model ranks a random positive higher than a random negative
   - **Interpretation**: 
     - 0.5 = random guessing
     - 1.0 = perfect separation
     - 0.9+ = excellent model
   - **Advantage**: Works well with class imbalance (doesn't depend on threshold)

**Why multiple metrics**:
- **Different perspectives**: Accuracy (overall), Precision (false positives), Recall (false negatives), ROC-AUC (ranking)
- **Domain requirements**: Drug discovery cares about both precision (don't waste money) and recall (don't miss discoveries)

### Step 7: Saving Artifacts (`train.py`)

**Artifacts saved**:

1. **`model.joblib`**: 
   - **Content**: Full sklearn Pipeline (featurizer + scaler + classifier)
   - **Usage**: Can load and predict on new data: `model = joblib.load("model.joblib"); model.predict(new_data)`
   - **Why joblib**: Standard sklearn format, preserves all fitted parameters

2. **`report.json`**:
   - **Content**: 
     - Dataset statistics (n_rows, n_invalid, class balance)
     - CV metrics (mean ± std for each metric)
     - Hyperparameters (model type, kmer_k, fp_bits, etc.)
     - Feature importance info (if available)
   - **Usage**: Readable summary for papers/presentations
   - **Why JSON**: Human-readable, easy to parse programmatically

3. **`feature_importance.csv`**:
   - **Content**: Top-K features ranked by importance
   - **Columns**: `feature`, `importance`, `method`
   - **Usage**: Interpretability (which features matter most?)
   - **Why CSV**: Easy to open in Excel, analyze, visualize

4. **`roc_curve_oof.png`** (if `--save_roc`):
   - **Content**: ROC curve plot (out-of-fold predictions)
   - **Usage**: Visual communication (show model performance graphically)
   - **Why out-of-fold**: Prevents overfitting in visualization (curve uses predictions from models that didn't train on those samples)

**Why save everything**:
- **Reproducibility**: Can reload model and reproduce results
- **Scientific communication**: Reports and plots for papers/presentations
- **Debugging**: Can inspect feature importance to understand model behavior

---

## 6. Models Used

This section explains **why** each model was chosen and **what patterns** it can learn.

### Logistic Regression (Baseline)

**What it is**: Linear classifier that learns a weighted sum of features.

**Mathematical form**: `P(interaction) = sigmoid(w₁·feature₁ + w₂·feature₂ + ... + b)`

**What it learns**:
- **Linear relationships**: "High LogP + high hydrophobic AAC → higher probability of interaction"
- **Feature weights**: Each feature gets a coefficient (positive = increases probability, negative = decreases)

**Why it's a good baseline**:
- **Interpretable**: Coefficients tell you which features matter and in which direction
- **Fast**: Trains in seconds even on large datasets
- **Stable**: Less prone to overfitting than complex models
- **Baseline comparison**: If complex models don't beat LogReg, they might be overfitting

**Limitations**:
- **Linear only**: Can't learn interactions (e.g., "LogP matters only if TPSA is low")
- **Assumes linearity**: Real drug–target interactions might be non-linear

**Hyperparameters** (in `make_logreg()`):
- `solver="saga"`: Handles large feature spaces (3,106 features) efficiently
- `class_weight="balanced"`: Automatically handles class imbalance
- `C=1.0`: Regularization strength (L2 penalty)

### Random Forest (Baseline)

**What it is**: Ensemble of decision trees, each trained on a random subset of data and features.

**What it learns**:
- **Non-linear relationships**: Can learn complex rules (e.g., "If LogP > 2 AND TPSA < 100 AND fingerprint_bit_42 == 1, then interaction")
- **Feature interactions**: Trees can combine features in complex ways
- **Robust**: Less sensitive to outliers than linear models

**Why it often performs well**:
- **Handles mixed features**: Works well with both continuous (descriptors) and binary (fingerprints) features
- **No scaling needed**: Trees split on feature values, so scaling doesn't matter (though we still scale for consistency)
- **Feature importance**: Provides interpretable feature rankings

**Limitations**:
- **Less interpretable**: Hard to explain why a specific prediction was made (compared to LogReg)
- **Memory**: Stores many trees (can be large for deep forests)

**Hyperparameters** (in `make_rf()`):
- `n_estimators=600`: Number of trees (more = better but slower)
- `max_features="sqrt"`: Random subset of features per tree (reduces overfitting)
- `class_weight="balanced_subsample"`: Handles imbalance per tree (better than global balancing)

### XGBoost (Advanced)

**What it is**: Gradient boosting (sequentially builds trees that correct previous trees' errors).

**What it learns**:
- **Complex non-linear patterns**: More powerful than Random Forest (can learn very complex relationships)
- **Feature interactions**: Deep trees can capture high-order interactions
- **Optimized**: Highly optimized implementation (fast, memory-efficient)

**Why it's "advanced"**:
- **State-of-the-art**: Often wins ML competitions
- **Handles large datasets**: Efficient even with millions of samples
- **Regularization**: Built-in L1/L2 regularization prevents overfitting

**Limitations**:
- **Less interpretable**: Harder to explain than Random Forest
- **Hyperparameter sensitive**: Many hyperparameters to tune (we use defaults)
- **Requires installation**: Not always available (falls back to MLP if missing)

**Hyperparameters** (in `make_xgb()`):
- `n_estimators=800`: Number of boosting rounds
- `learning_rate=0.05`: Step size (smaller = more conservative, needs more trees)
- `max_depth=6`: Tree depth (deeper = more complex, risk of overfitting)
- `scale_pos_weight`: Computed from class imbalance (e.g., if 70% negative, weight = 2.33)

### MLP (Neural Network, Advanced Fallback)

**What it is**: Multi-layer perceptron (neural network with hidden layers).

**What it learns**:
- **Non-linear patterns**: Can approximate any function (universal approximator)
- **Feature combinations**: Hidden layers learn combinations of input features
- **Complex relationships**: More flexible than trees (but also more prone to overfitting)

**Why it's a fallback**:
- **Available**: sklearn MLPClassifier is always available (no extra install)
- **Non-linear**: Can learn complex patterns if XGBoost isn't available

**Limitations**:
- **Overfitting**: Prone to overfitting (we use early stopping to prevent this)
- **Slow**: Trains slower than tree-based models
- **Hyperparameter sensitive**: Many hyperparameters (learning rate, architecture, etc.)

**Hyperparameters** (in `make_mlp()`):
- `hidden_layer_sizes=(256, 128)`: Two hidden layers (256 neurons, then 128)
- `early_stopping=True`: Stops training if validation loss doesn't improve (prevents overfitting)
- `alpha=1e-4`: L2 regularization strength

### Why Random Forest Often Performs Well

**Empirical observation**: On DTI datasets, Random Forest often achieves high ROC-AUC (e.g., 0.94).

**Reasons**:
1. **Mixed feature types**: RF handles both continuous (descriptors) and binary (fingerprints) features naturally
2. **Feature interactions**: Drug–target interactions likely depend on combinations of features (e.g., "hydrophobic drug + hydrophobic protein pocket")
3. **Robust to noise**: Fingerprints are sparse and noisy (many bits irrelevant), but RF can ignore irrelevant features
4. **No overfitting**: With proper hyperparameters, RF generalizes well (unlike deep networks which can overfit)

**When XGBoost might be better**:
- **Very large datasets**: XGBoost scales better to millions of samples
- **Complex interactions**: If interactions are very complex (beyond what RF can capture)
- **Feature engineering**: If features are well-engineered, XGBoost can exploit them better

**When LogReg might be sufficient**:
- **Linear relationships**: If drug–target binding is mostly linear in features
- **Interpretability**: If you need to explain predictions to domain experts
- **Small datasets**: LogReg less prone to overfitting on small data

---

## 7. Evaluation and Results Interpretation

This section explains how to **read and interpret** the evaluation results.

### Understanding `report.json`

**Example structure**:
```json
{
  "data_path": "data/dti_dataset.csv",
  "model": "rf",
  "seed": 42,
  "dataset_stats": {
    "n_rows_raw": 10000,
    "n_rows_valid": 9850,
    "n_invalid_smiles": 100,
    "n_invalid_sequences": 50,
    "n_pos": 4925,
    "n_neg": 4925
  },
  "class_balance": {
    "n_pos": 4925,
    "n_neg": 4925,
    "pos_rate": 0.5
  },
  "cv": {
    "accuracy_mean": 0.89,
    "accuracy_std": 0.02,
    "precision_mean": 0.88,
    "precision_std": 0.03,
    "recall_mean": 0.90,
    "recall_std": 0.02,
    "f1_mean": 0.89,
    "f1_std": 0.02,
    "roc_auc_mean": 0.94,
    "roc_auc_std": 0.01
  }
}
```

**What to look for**:
- **`dataset_stats`**: How much data was lost? (150 invalid rows out of 10,000 = 1.5% loss, acceptable)
- **`class_balance`**: Is it balanced? (50/50 is ideal, 60/40 is acceptable, 90/10 is problematic)
- **`cv` metrics**: Mean performance across folds (what we care about)
- **`cv` std**: Variability across folds (low std = stable, high std = unstable/unreliable)

### Interpreting Metrics (Simple Terms)

#### ROC-AUC ≈ 0.94

**What it means**:
- **0.94 = 94%**: The model can correctly rank 94% of positive–negative pairs
- **Excellent performance**: 0.9+ is considered excellent in ML
- **Better than random**: 0.5 = random guessing, 0.94 = much better

**Interpretation**:
- **Ranking ability**: Given a random interacting pair and a random non-interacting pair, the model will rank the interacting pair higher 94% of the time
- **Practical use**: You can rank drug–target pairs by predicted probability and test the top-K experimentally

**What ROC-AUC doesn't tell you**:
- **Threshold**: Doesn't tell you what probability threshold to use (that's a business decision)
- **Calibration**: A model with AUC=0.94 might not be well-calibrated (predicted probabilities might not match true probabilities)

#### F1-Score ≈ 0.89

**What it means**:
- **Balanced metric**: Harmonic mean of precision and recall
- **0.89 = 89%**: Good balance between precision and recall

**Interpretation**:
- **Precision ≈ 0.88**: Of all predictions labeled "interaction", 88% are correct (12% false positives)
- **Recall ≈ 0.90**: Of all true interactions, we find 90% (10% false negatives)

**Trade-off**:
- **High precision, low recall**: Model is conservative (few false positives, but misses many real interactions)
- **Low precision, high recall**: Model is aggressive (finds many interactions, but many are false)
- **F1 balances these**: 0.89 means good balance

#### Accuracy ≈ 0.89

**What it means**:
- **89% of predictions are correct**

**Limitation**:
- **Misleading with imbalance**: If dataset is 90% negative, a model that always predicts "no interaction" has 90% accuracy (but useless)
- **In this case**: With balanced classes (50/50), accuracy ≈ F1 is expected and meaningful

### What ROC-AUC ≈ 0.94 Says About the Model

**Strong signal**: The model has learned meaningful patterns that distinguish interacting from non-interacting pairs.

**Possible patterns learned**:
- **Molecular properties**: "Drugs with high LogP interact with hydrophobic proteins"
- **Fingerprint patterns**: "Drugs with specific chemical groups (encoded in fingerprints) bind to proteins with specific sequence motifs"
- **Combinations**: "Interaction requires both drug property X and protein property Y"

**Biological plausibility**:
- **Not random**: 0.94 is too high to be random noise
- **Feature-driven**: The features (descriptors, fingerprints, AAC, k-mers) contain signal about binding
- **Generalizable**: If CV was done correctly, this should generalize to new data

### Is There Overfitting?

**Signs of overfitting**:
- **Train accuracy >> CV accuracy**: If train accuracy = 0.99 but CV = 0.89, model memorized training data
- **High variance across folds**: If `roc_auc_std = 0.10`, model is unstable (might be overfitting)

**Signs of good generalization**:
- **Train ≈ CV**: If train accuracy ≈ CV accuracy, model generalizes well
- **Low variance**: If `roc_auc_std = 0.01`, model is stable across folds
- **Reasonable performance**: 0.94 ROC-AUC is high but not suspiciously high (1.0 would be suspicious)

**In this project**:
- **Pipeline design**: Featurization and scaling happen inside CV (leakage-safe), so CV estimates should be reliable
- **Regularization**: Models use regularization (L2 for LogReg, `max_features` for RF, `alpha` for MLP) to prevent overfitting
- **Early stopping**: MLP uses early stopping (stops if validation loss doesn't improve)

**Conclusion**: If CV metrics are stable (low std) and reasonable (not suspiciously high), the model is likely **not overfitting**.

---

## 8. Prediction Capability

### Can the Model Predict Interactions for Unseen Pairs?

**Yes, but with caveats**.

**What "unseen" means**:
- **Unseen drug–target pairs**: The model can predict on drug–target combinations it hasn't seen during training
- **Unseen drugs**: The model can predict on drugs it hasn't seen (if their SMILES can be featurized)
- **Unseen proteins**: The model can predict on proteins it hasn't seen (if their sequences can be featurized)

**How to use the model**:
```python
import joblib
import pandas as pd

# Load trained model
model = joblib.load("runs/baseline_rf/model.joblib")

# New drug–target pair
new_data = pd.DataFrame({
    "compound_iso_smiles": ["CC(=O)OC1=CC=CC=C1C(=O)O"],  # Aspirin
    "target_sequence": ["MVKVYAPASSANMSVGFDVLGAAVTPVDGALLGDVVTVEAAETFSLNNLGQKLTKELGADVVV"]
})

# Predict probability
probability = model.predict_proba(new_data)[0, 1]  # Probability of interaction
prediction = model.predict(new_data)[0]  # Binary prediction (0 or 1)

print(f"Probability of interaction: {probability:.3f}")
print(f"Prediction: {'Interaction' if prediction == 1 else 'No interaction'}")
```

### What Does "Generalization" Mean?

**Generalization**: The model's ability to make accurate predictions on **new data** it hasn't seen during training.

**Types of generalization**:

1. **Interpolation** (easier):
   - New drug–target pairs where both drug and target are **similar** to training data
   - **Example**: Training has drug A with protein X; predicting drug A' (similar to A) with protein X
   - **Expected**: Good performance (model learned patterns that apply to similar molecules)

2. **Extrapolation** (harder):
   - New drug–target pairs where drug or target is **very different** from training data
   - **Example**: Training has small molecules (< 500 Da); predicting large molecules (> 1000 Da)
   - **Expected**: Poor performance (model hasn't seen this type of data)

**In DTI prediction**:
- **Chemical space**: If training data covers diverse chemical structures, model can generalize to similar structures
- **Protein families**: If training data covers diverse protein families, model can generalize to similar families
- **Novel combinations**: Model can predict novel drug–target combinations if individual components (drug or target) are similar to training data

### When Are Predictions Reliable?

**High confidence predictions** (probability close to 0 or 1):
- **Reliable if**: Drug and target are similar to training data
- **Example**: Predicting interaction for a drug similar to training drugs with a protein similar to training proteins

**Low confidence predictions** (probability around 0.5):
- **Uncertain**: Model is unsure (might be a novel drug–target combination)
- **Action**: Treat with caution, might need experimental validation

**Unreliable predictions**:
- **Out-of-distribution**: Drug or target is very different from training data
  - **Example**: Training on small molecules, predicting on peptides
  - **Example**: Training on kinases, predicting on GPCRs
- **Invalid inputs**: SMILES or sequence that can't be featurized properly
- **Domain shift**: Training on one type of binding (e.g., competitive inhibition), predicting on another (e.g., allosteric modulation)

**Best practice**:
- **Rank predictions**: Sort by predicted probability, test top-K experimentally
- **Validate**: Always validate high-confidence predictions experimentally
- **Domain expertise**: Consult domain experts for novel drug–target combinations

### Limitations of Random Splits

**What "random split" means**: Data is randomly divided into train/test (or CV folds), without considering drug or target identity.

**Problem**: If the same drug appears in both train and test sets, the model might "memorize" that drug's properties, leading to **overly optimistic** performance estimates.

**Example**:
- **Training**: Drug A with proteins X, Y, Z (all interactions)
- **Test**: Drug A with protein W (interaction)
- **Model**: "I've seen drug A before, and it always interacts → predict interaction"
- **Reality**: Model memorized drug A, didn't learn generalizable patterns

**Better approach**: **Group-based splits**
- **Drug-based**: All pairs involving drug A go to train OR test (not both)
- **Target-based**: All pairs involving protein X go to train OR test (not both)
- **More realistic**: Simulates real-world scenario (predicting interactions for completely new drugs/targets)

**Why this project uses random splits**:
- **Simplicity**: Easier to implement and understand
- **Baseline**: Standard approach in many papers (allows comparison)
- **Future work**: Can be extended to group-based splits

**Impact**: Random splits might give **slightly optimistic** performance (e.g., ROC-AUC = 0.94 might be 0.90 with group-based splits), but the general conclusions (model learns meaningful patterns) still hold.

---

## 9. Limitations and Assumptions

### Assumptions Made by This Approach

1. **Feature representation is sufficient**:
   - **Assumption**: Descriptors + fingerprints + AAC + k-mers capture all information needed for binding prediction
   - **Reality**: Some binding mechanisms might depend on 3D structure, dynamics, or other factors not captured by these features
   - **Impact**: Model might miss some interactions that depend on unmodeled factors

2. **Binary classification is appropriate**:
   - **Assumption**: "Interaction" vs "no interaction" is a meaningful distinction
   - **Reality**: Binding strength is continuous; thresholding loses information
   - **Impact**: Can't distinguish strong vs. weak binders (both labeled "interaction")

3. **Training data is representative**:
   - **Assumption**: Training data covers the chemical/protein space we want to predict on
   - **Reality**: Datasets are biased (e.g., more kinase data than GPCR data)
   - **Impact**: Model might perform poorly on underrepresented protein families

4. **Independent and identically distributed (i.i.d.) samples**:
   - **Assumption**: Each drug–target pair is independent
   - **Reality**: Pairs involving the same drug or target are correlated
   - **Impact**: Random splits might overestimate performance (see Section 8)

5. **Linear/non-linear relationships are learnable**:
   - **Assumption**: ML models can learn the relationship between features and binding
   - **Reality**: Some relationships might be too complex or require domain knowledge
   - **Impact**: Model might plateau at certain performance levels

### Limitations of Random Splits

**Already discussed in Section 8**, but summarized:

- **Memorization**: Model might memorize specific drugs/targets rather than learning general patterns
- **Overly optimistic**: Performance estimates might be higher than true generalization
- **Not realistic**: Doesn't simulate real-world scenario (predicting for completely new drugs)

**Mitigation**: Use group-based splits for more realistic evaluation (future work).

### Why Domain Applicability Matters

**Domain**: The specific type of drug–target interactions the model was trained on.

**Examples**:
- **Kinases**: Model trained on kinase inhibitors might not work for GPCRs
- **Small molecules**: Model trained on small molecules might not work for biologics (proteins/antibodies)
- **Competitive binding**: Model trained on competitive inhibitors might not work for allosteric modulators

**Why it matters**:
- **Feature relevance**: Features that matter for one domain might not matter for another
- **Distribution shift**: Training and test data come from different distributions
- **Performance degradation**: Model performance drops when applied to different domains

**Best practice**:
- **Train on diverse data**: Include multiple protein families, drug types
- **Domain-specific models**: Train separate models for different domains (e.g., kinases vs. GPCRs)
- **Transfer learning**: Use pre-trained features, fine-tune on domain-specific data

**In this project**:
- **General features**: Descriptors and fingerprints are general (work across domains)
- **Domain-agnostic**: Model should work across domains if training data is diverse
- **Limitation**: If training data is domain-specific, predictions on other domains are unreliable

### Other Limitations

1. **No 3D structure**: Features are 1D (SMILES string, sequence), but binding happens in 3D
   - **Impact**: Might miss structure-dependent binding mechanisms
   - **Future work**: Incorporate 3D structure (docking scores, molecular dynamics)

2. **No binding site information**: Don't know where on the protein the drug binds
   - **Impact**: Can't distinguish binding to active site vs. allosteric site
   - **Future work**: Incorporate binding site predictions

3. **Static features**: Features don't capture dynamics (how molecules move, conformational changes)
   - **Impact**: Might miss dynamic binding mechanisms
   - **Future work**: Incorporate molecular dynamics simulations

4. **No experimental conditions**: Don't account for pH, temperature, concentration
   - **Impact**: Predictions assume standard conditions
   - **Future work**: Multi-task learning with experimental conditions

---

## 10. Final Summary

### What Has Been Successfully Achieved

1. **Complete ML pipeline**: From raw SMILES/sequences to trained models
2. **Feature engineering from scratch**: No pre-computed features; all computed here
3. **Multiple models**: Baseline (LogReg, RF) and advanced (XGBoost, MLP)
4. **Robust evaluation**: Cross-validation, multiple metrics, leakage-safe
5. **Interpretability**: Feature importance, clear feature names, explainable design
6. **Reproducibility**: Fixed seeds, documented hyperparameters, versioned code
7. **Production-ready**: Saves models, reports, plots; can be used for predictions

### Why This Is a Solid Graduation Project

1. **Scientific rigor**:
   - **Proper evaluation**: Cross-validation, multiple metrics, leakage prevention
   - **Documentation**: Clear explanations of design choices
   - **Reproducibility**: Can be rerun and results reproduced

2. **Technical depth**:
   - **Feature engineering**: Complex domain-specific feature extraction
   - **ML expertise**: Multiple models, hyperparameter tuning, class imbalance handling
   - **Software engineering**: Modular code, clean structure, error handling

3. **Domain relevance**:
   - **Real-world problem**: Drug discovery is an important application
   - **Benchmark dataset**: Can compare to published results (KIBA)
   - **Practical utility**: Can be used for actual drug screening

4. **Educational value**:
   - **Teaches ML concepts**: Feature engineering, cross-validation, model comparison
   - **Teaches domain knowledge**: Drug properties, protein sequences, binding
   - **Teaches software practices**: Code organization, documentation, reproducibility

### Clear Future Improvements

1. **Group-based splits**:
   - **What**: Split by drug or target identity (not random)
   - **Why**: More realistic evaluation (predicting for completely new drugs)
   - **How**: Use `GroupKFold` from sklearn, group by drug_id or target_id

2. **Regression instead of classification**:
   - **What**: Predict actual affinity values (IC50, Ki, KIBA score)
   - **Why**: More informative (distinguish strong vs. weak binders)
   - **How**: Replace classifier with regressor (RandomForestRegressor, XGBRegressor)

3. **Deep learning**:
   - **What**: Use neural networks that learn embeddings from raw SMILES/sequences
   - **Why**: Might capture patterns missed by hand-crafted features
   - **How**: 
     - **Drugs**: Graph neural networks (GNNs) on molecular graphs, or SMILES transformers
     - **Proteins**: Protein language models (ESM, ProtBERT) that learn sequence embeddings
   - **Example**: Replace `DTIFeaturizer` with a neural network that takes raw SMILES/sequences

4. **3D structure incorporation**:
   - **What**: Add features from 3D molecular structures
   - **Why**: Binding happens in 3D; 1D features might miss structure-dependent mechanisms
   - **How**: 
     - **Docking scores**: Use AutoDock, Vina to compute binding energies
     - **3D descriptors**: Shape, volume, surface area from 3D structures
     - **Molecular dynamics**: Trajectory-based features

5. **Multi-task learning**:
   - **What**: Predict multiple properties simultaneously (binding, toxicity, ADMET)
   - **Why**: Shared representations might improve all tasks
   - **How**: Neural network with multiple output heads

6. **Interpretability improvements**:
   - **What**: Better explanations of predictions
   - **Why**: Domain experts need to understand why model predicts interaction
   - **How**: 
     - **SHAP values**: Explain individual predictions
     - **Attention mechanisms**: Show which parts of SMILES/sequence matter
     - **Rule extraction**: Convert model to interpretable rules

7. **Active learning**:
   - **What**: Model suggests which drug–target pairs to test experimentally
   - **Why**: Optimize experimental budget (test most informative pairs)
   - **How**: Use uncertainty estimates (low confidence = informative) to select pairs

8. **Transfer learning**:
   - **What**: Pre-train on large datasets, fine-tune on specific domains
   - **Why**: Leverage large datasets (ChEMBL, BindingDB) for better features
   - **How**: Pre-train protein language model on UniProt, fine-tune on DTI data

### Conclusion

This project successfully demonstrates:
- **End-to-end ML pipeline** for drug–target interaction prediction
- **Feature engineering** from raw molecular representations
- **Model comparison** across multiple algorithms
- **Robust evaluation** with proper cross-validation
- **Production-ready code** that can be used for predictions

The achieved performance (ROC-AUC ≈ 0.94) indicates that the learned features contain meaningful signal about drug–target binding, and the model can distinguish interacting from non-interacting pairs with high accuracy.

**For a graduation project**, this demonstrates:
- **Technical skills**: ML, feature engineering, software engineering
- **Domain knowledge**: Understanding of drug discovery and bioinformatics
- **Scientific rigor**: Proper evaluation, reproducibility, documentation
- **Future potential**: Clear path for improvements and extensions

This is a **solid foundation** for further research, whether in academia (extending to deep learning, 3D structure) or industry (deploying for drug screening, integrating into discovery pipelines).

---

## Appendix: Quick Reference

### Running the Project

```bash
# Setup (Conda recommended on Windows)
conda env create -f environment.yml
conda activate dti

# Train a model
python train.py --data data/dti_dataset.csv --outdir runs/rf --model rf --cv 5 --drop_invalid --save_roc

# Load and use trained model
python -c "
import joblib
import pandas as pd
model = joblib.load('runs/rf/model.joblib')
new_data = pd.DataFrame({
    'compound_iso_smiles': ['CC(=O)OC1=CC=CC=C1C(=O)O'],
    'target_sequence': ['MVKVYAPASSANMSVGFDVLGAAVTPVDGALLGDVVTVEAAETFSLNNLGQKLTKELGADVVV']
})
print(f'Probability: {model.predict_proba(new_data)[0,1]:.3f}')
"
```

### Key Files

- **`train.py`**: Main entry point
- **`dti/features/drug.py`**: Drug featurization (RDKit)
- **`dti/features/protein.py`**: Protein featurization (AAC + k-mers)
- **`dti/models.py`**: Model factory
- **`dti/eval.py`**: Evaluation and metrics
- **`examples/tiny_dti.csv`**: Example dataset

### Feature Dimensions

- Drug descriptors: **14**
- Drug fingerprint: **2048**
- Protein AAC: **20**
- Protein k-mers: **1024**
- **Total: 3,106 features**

### Models Available

- `logreg`: Logistic Regression (baseline, interpretable)
- `rf`: Random Forest (baseline, often best performance)
- `xgb`: XGBoost (advanced, requires installation)
- `mlp`: Multi-layer Perceptron (advanced fallback)

---

**End of Explanation Document**
