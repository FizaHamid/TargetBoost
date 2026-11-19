# TargetBoost
![OS](https://img.shields.io/badge/os-linux%20%7C%20macOS-blue) ![Python](https://img.shields.io/badge/python-3.9-blue) ![Model](https://img.shields.io/badge/model-XGBoost-green) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<!-- Uncomment the line below once you add your flowchart image in img/flowchart.png -->
<!-- ![flowchart](img/flowchart.png) -->

**TargetBoost** is a machine learning framework designed to predict biologically functional interactions between **plant tRNA-derived fragments (tRFs)** and **messenger RNA (mRNA) targets**. Unlike purely thermodynamic scanning tools, TargetBoost integrates **sequence composition**, **local target-site accessibility**, **seed region dynamics**, **duplex architecture**, and **minimum free energy ($MFE$)** calculated via [*RNAduplex*](https://www.tbi.univie.ac.at/RNA/RNAduplex.1.html) (ViennaRNA) into a gradient-boosted decision tree model (**XGBoost**) to distinguish functional cleavage/AGO-associated interactions from non-functional binding sites.

---

## Table of Contents

1. [Installation & Requirements](#1-installation--requirements)
   - 1.1 [Prerequisites](#11-prerequisites)
   - 1.2 [Setup](#12-setup)
2. [Usage](#2-usage)
   - 2.1 [Step 1: Feature Extraction](#21-step-1-feature-extraction)
   - 2.2 [Step 2: Model Prediction](#22-step-2-model-prediction)
   - 2.3 [Input File Formats](#23-input-file-formats)
   - 2.4 [Output File Format](#24-output-file-format)
3. [Methodology](#3-methodology)
   - 3.1 [Dataset Construction](#31-dataset-construction)
   - 3.2 [Feature Engineering (66 Attributes)](#32-feature-engineering-66-attributes)
   - 3.3 [Model Architecture & Evaluation](#33-model-architecture--evaluation)
   - 3.4 [Hyperparameter Optimization](#34-hyperparameter-optimization)
4. [Directory Structure](#4-directory-structure)
5. [Citation](#5-citation)
6. [Contact](#6-contact)

---

## 1. Installation & Requirements

### 1.1 Prerequisites
TargetBoost supports **Linux** and **macOS** environments. Before running the pipeline, ensure the following software and Python libraries are installed:

* **Python**: `3.9`
* **ViennaRNA Package**: `RNAduplex` (must be accessible in your system `$PATH`)
* **Python Packages**:
  * `xgboost`
  * `pandas`
  * `numpy`
  * `scikit-learn`
  * `biopython`

### 1.2 Setup

Clone the repository and install the required Python dependencies:

```bash
git clone [https://github.com/YOUR-USERNAME/TargetBoost.git](https://github.com/YOUR-USERNAME/TargetBoost.git)
cd TargetBoost
pip install -r requirements.txt
```

Verify that `RNAduplex` is properly installed and accessible:

```bash
RNAduplex --version
```

---

## 2. Usage

TargetBoost executes in a two-step workflow: **(1)** duplex alignment and 66-feature extraction, followed by **(2)** probability scoring using the pre-trained XGBoost classifier.

### 2.1 Step 1: Feature Extraction

Run `feature_extraction.py` to compute sequence, structural, positional, and thermodynamic features for your query tRF–transcript pairs:

```bash
python3.9 scripts/feature_extraction.py
```

**Required Inputs:**
* `pairs.tsv` — Query tRF and target transcript pair list
* `trfs.fasta` — Query tRF sequences in FASTA format
* `transcripts.fasta` — Reference target transcript / CDS sequences in FASTA format

**Output:**
* `feature.tsv` — A 66-column feature matrix ready for model inference.

---

### 2.2 Step 2: Model Prediction

Run `model.py` using the pre-trained XGBoost model file (`.pkl`) and the feature matrix generated in Step 1:

```bash
python3.9 scripts/model.py
```

**Required Inputs:**
* `models/trained_xgboost_model.pkl` — Pre-trained TargetBoost XGBoost classifier
* `feature.tsv` — Feature table generated in Step 1

**Output:**
* `predictions.tsv` — Final target interaction predictions with probability scores.

---

### 2.3 Input File Formats

1. **`trfs.fasta`** (Query tRF sequences):
   ```fasta
   >tRF_001
   UCCCGGACUAGCUCAAGUCGG
   >tRF_002
   AUCCCGGUCUAGGCUAGCUGA
   ```

2. **`transcripts.fasta`** (Target mRNA / CDS sequences):
   ```fasta
   >AT1G01010.1
   ATGGAGGATCAAGTTGGGTTTGGGTTCCGTCCGAACGACGAGGAGCTCGTCGGTCACTATCTCCGTAAC...
   ```

3. **`pairs.tsv`** (Tab-separated query–target pairs):

| tRF_ID | Transcript_ID |
| :--- | :--- |
| `tRF_001` | `AT1G01010.1` |
| `tRF_002` | `AT1G02050.2` |

---

### 2.4 Output File Format

The final prediction output (`predictions.tsv`) reports the interaction confidence for each queried tRF–target pair:

| Column | Description |
| :--- | :--- |
| `tRF_ID` | ID of the query tRF sequence. |
| `Transcript_ID` | ID of the candidate target transcript. |
| `MFE` | Minimum Free Energy ($\text{kcal/mol}$) of the tRF–mRNA duplex computed via `RNAduplex`. |
| `Prediction_Score` | Predicted probability ($0.0$ to $1.0$) of a functional interaction from XGBoost. |
| `Predicted_Class` | Binary classification label (`1` = Functional Target, `0` = Non-Target). |

---

## 3. Methodology

### 3.1 Dataset Construction

To prevent trivial classification and ensure biological relevance in plants, TargetBoost was trained on rigorously curated positive and negative datasets:

* **Positive Dataset (`Target_Label = 1`)**:
  * Collected from experimentally supported plant **PARE/degradome sequencing** datasets (CleaveLand outputs), published literature, and plant small RNA databases.
  * Filtered strictly to retain interactions exhibiting clear evidence of target cleavage or Argonaute (AGO) association.

* **Negative Dataset (`Target_Label = 0`)**:
  * Generated by pairing tRFs with non-target transcript regions lacking degradome support, alongside nucleotide-composition-preserving sequence controls.
  * Filtered via `RNAduplex` (`Degradome_Flag == 0`) so that negative pairs maintain comparable nucleotide composition and hybridization potential ($MFE$) to positive interactions, eliminating composition bias.

---

### 3.2 Feature Engineering (66 Attributes)

For every candidate tRF–mRNA pair, TargetBoost extracts **66 quantitative features** across six biological categories:

| Category | Features Extracted | Biological Rationale |
| :--- | :--- | :--- |
| **A. tRF Sequence** | `Length_of_tRF`, `tRF_GC%` | tRF length and GC content govern RISC loading efficiency and duplex stability. |
| **B. Target Site Context** | 16 dinucleotide frequencies (`AA`–`UU`), flanking `AU%` and `GC%` ($\pm 30\text{ nt}$) | AU-rich flanking regions reduce local secondary structure and increase site accessibility. |
| **C. Positional Features** | Distance to $5'$ end, Distance to $3'$ end | Captures positional targeting biases across plant transcript regions (CDS vs. UTRs). |
| **D. Seed Region** | Position 1 match, Position 8 match, Position 1 Adenine indicator, flanking base identities ($-2, -1, +1, +2$ one-hot encoded) | Position 8 pairing and flanking nucleotide identities are key determinants of functional target recognition. |
| **E. Duplex Architecture** | Seed length, binding region length, longest consecutive pairing, position of longest pairing, paired positions, gaps, unpaired positions | Captures alignment geometry, catalytic cleavage compatibility, and bulge penalties. |
| **F. Thermodynamics** | Minimum Free Energy ($MFE$ in $\text{kcal/mol}$) via `RNAduplex` | Quantifies overall thermodynamic hybridization stability (top individual predictor). |

---

### 3.3 Model Architecture & Evaluation

Five machine learning algorithms—**Random Forest**, **XGBoost**, **LightGBM**, **CatBoost**, and **Support Vector Machine (SVM)**—were benchmarked using **10-fold stratified cross-validation**.

Models were evaluated across seven performance metrics:
* **Accuracy**
* **Precision**
* **Recall**
* **F1-Score**
* **Matthews Correlation Coefficient (MCC)**
* **ROC-AUC**
* **PR-AUC**

**XGBoost** achieved the strongest overall balance across MCC, PR-AUC, and ROC-AUC, and was selected as the core classification engine for TargetBoost.

---

### 3.4 Hyperparameter Optimization

The final XGBoost classifier was tuned using Bayesian optimization (**Optuna**) over 5-fold stratified cross-validation, optimizing directly for **Matthews Correlation Coefficient (MCC)** across `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, and `gamma`.

---

## 4. Directory Structure

```text
TargetBoost/
├── data/
│   ├── pairs.tsv                     # Sample query-target pairs
│   ├── trfs.fasta                    # Sample tRF sequences
│   └── transcripts.fasta             # Sample target CDS/transcript sequences
├── scripts/
│   ├── feature_extraction.py         # Step 1: 66-feature extraction using RNAduplex
│   └── model.py                      # Step 2: Inference using trained XGBoost model
├── models/
│   └── trained_xgboost_model.pkl     # Pre-trained TargetBoost classifier
├── docs/
│   ├── _config.yml                   # GitHub Pages configuration
│   └── index.md                      # GitHub Pages documentation
├── requirements.txt                  # Python package dependencies
└── README.md                         # Project documentation
```

---

## 5. Citation

If you use **TargetBoost** in your research, please cite our repository/manuscript:

> PtRFXplorer: Differentially Expressed tRFs in Plants with Validated Degradome Targets. GitHub: https://github.com/hamidfiza25-commits/TargetBoost*

---

## 6. Contact

For questions, bug reports, or feature requests, please open an issue on the [TargetBoost GitHub Issues](https://github.com/hamidfiza25-commits/TargetBoost/issues) page.
