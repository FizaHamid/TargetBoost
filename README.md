# TargetBoost
![OS](https://img.shields.io/badge/os-linux%20%7C%20macOS-blue) ![Python](https://img.shields.io/badge/python-3.9-blue) ![Model](https://img.shields.io/badge/model-XGBoost-green) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<!-- Uncomment the line below once you add your flowchart image in img/flowchart.png -->
<!-- ![flowchart](img/flowchart.png) -->

**TargetBoost** is a machine learning framework designed to predict biologically functional interactions between **plant tRNA-derived fragments (tRFs)** and **messenger RNA (mRNA) targets**. Unlike purely thermodynamic scanning tools, TargetBoost integrates **sequence composition**, **local target-site accessibility**, **seed region dynamics**, **duplex architecture**, and **minimum free energy ($MFE$)** into a gradient-boosted decision tree model (**XGBoost**) to distinguish functional cleavage/AGO-associated interactions from non-functional binding sites.

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
   - 3.2 [Feature Engineering (65 Attributes)](#32-feature-engineering-65-attributes)
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
git clone https://github.com/hamidfiza25-commits/TargetBoost.git
cd TargetBoost
pip install -r requirements.txt
```

Verify that `RNAduplex` is properly installed and accessible:

```bash
RNAduplex --version
```

---

## 2. Usage

TargetBoost executes in a two-step workflow: **(1)** duplex alignment and 65-feature extraction, followed by **(2)** probability scoring using the pre-trained XGBoost classifier.

### 2.1 Step 1: Feature Extraction

Run `feature_extraction.py` to compute sequence, structural, positional, and thermodynamic features for your query tRF–transcript pairs:

```bash
python feature_extraction.py \
    --trf trfs.fasta \
    --transcripts transcripts.fasta \
    --pairs pairs.tsv \
    --output feature.tsv
```

**Required Inputs:**
* `pairs.tsv`: Query tRF and target transcript pair list
* `trfs.fasta`: Query tRF sequences in FASTA format
* `transcripts.fasta`: Reference target transcript / CDS sequences in FASTA format

**Output:**
* `feature.tsv`: A 65-column feature matrix ready for model inference.

---

### 2.2 Step 2: Model Prediction

Run `model.py` using the pre-trained XGBoost model file (`.pkl`) and the feature matrix generated in Step 1:

```bash
python3.9 scripts/model.py
```

**Required Inputs:**
* `models/trained_xgboost_model.pkl`: Pre-trained TargetBoost XGBoost classifier
* `feature.tsv`: Feature table generated in Step 1

**Output:**
* `predictions.tsv`: Final target interaction predictions with probability scores.

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

The TargetBoost pipeline follows a structured six-stage workflow integrating sequence composition, target-site accessibility, positional context, seed region dynamics, duplex architecture, and thermodynamic stability to distinguish functional plant tRF–mRNA interactions from non-functional binding events.

### 3.1 Dataset Construction

To prevent trivial classification and ensure biological relevance in plants, models were trained on curated positive and negative interaction datasets:

* **Positive Dataset (`Target_Label = 1`)**:
  * Collected from experimentally supported plant **degradome sequencing datasets** (CleaveLand outputs).
  * Filtered strictly to retain only interactions showing clear evidence of target cleavage or Argonaute (AGO) association.

* **Negative Dataset (`Target_Label = 0`)**:
  * Generated by pairing tRFs with target transcript regions lacking degradome support, alongside nucleotide-composition-preserving sequence controls.
  * Filtered so negative sites retain similar nucleotide composition and comparable hybridization potential ($MFE$) to positive sites, preventing composition bias.

---

### 3.2 Candidate Target Site Identification

Prior to feature computation, potential binding regions across plant transcripts are identified through a four-step screening process:

1. **Transcriptome Scanning**: Scan target transcripts for candidate binding sites using sequence complementarity rules.
2. **Duplex Generation**: Predict tRF–mRNA secondary structure duplexes.
3. **Structural & Thermodynamic Profiling**: Compute duplex alignment geometry and binding free energy parameters.
4. **Context Extraction**: Extract the local flanking sequence ($\pm 30\text{ nt}$) surrounding each candidate binding site.

---

### 3.3 Feature Engineering & Processing (65 Attributes)

For every candidate tRF–mRNA pair, TargetBoost extracts **65 quantitative and categorical features** spanning six biological categories:

| Category | Features Extracted | Formula / Implementation Details |
| :--- | :--- | :--- |
| **A. tRF Sequence** | `Length_of_tRF`, `tRF_GC%` | Total nucleotide length and $\text{GC}\% = \frac{G+C}{\text{Length}} \times 100$. Influences RISC loading efficiency and duplex stability. |
| **B. Target Site Context** | 16 dinucleotide frequencies (`AA`–`UU`), flanking `AU%` and `GC%` | Computed across a $\pm 30\text{ nt}$ window ($60\text{ nt}$ total) flanking the target site: $\text{AU}\% = \frac{A+U}{60} \times 100$ and $\text{GC}\% = \frac{G+C}{60} \times 100$. AU-rich regions indicate higher structural accessibility. |
| **C. Positional Features** | Distance to $5'$ end, Distance to $3'$ end | Distance from the target site to the transcript start and end, capturing positional targeting biases. |
| **D. Seed Region** | Position 1 match, Position 8 match, Position 1 Adenine (`A`) indicator, flanking base identities ($-2, -1, +1, +2$) | Binary match indicators (`1` = matched, `0` = mismatched), `1`/`0` indicator for Adenine opposite position 1, and one-hot encoded flanking nucleotide identities. |
| **E. Duplex Architecture** | Seed length, binding region length, longest consecutive pairing, position of longest pairing (relative to tRF $5'$ end), paired positions, gaps (indels), unpaired positions (mismatches/bulges) | Captures alignment geometry, uninterrupted pairing stretches, and structural bulge/gap penalties. |
| **F. Thermodynamics** | Minimum Free Energy ($MFE$ in $\text{kcal/mol}$). |

**Feature Processing:**
* **Numerical Features**: Standardized using **Z-score normalization** or **Min-Max scaling**.
* **Categorical Features**: Converted via **one-hot encoding** for flanking nucleotide identities (`A`, `U`, `G`, `C`).

---

### 3.4 Model Development & Evaluation

Five machine learning classifiers were benchmarked to identify the optimal predictive architecture:

* **Random Forest**: Captures non-linear relationships, handles correlated features robustly, and provides intrinsic feature importance (following *tRForest*).
* **XGBoost**: Delivers high predictive accuracy on mixed feature types with built-in regularization and feature ranking.
* **LightGBM**: Efficient histogram-based gradient boosting for fast training and leaf-wise tree growth.
* **CatBoost**: Gradient boosting with symmetric trees that reduces overfitting on tabular biological features.
* **Support Vector Machine (SVM)**: Effective margin-based classification on small-to-medium datasets (following *tRFTars*).

**Validation & Metrics:**
All models were evaluated using **10-fold stratified cross-validation** across seven standard metrics:
* **Accuracy**
* **Precision**
* **Recall**
* **F1-Score**
* **Matthews Correlation Coefficient (MCC)**
* **ROC-AUC**
* **PR-AUC**

**XGBoost** achieved the strongest overall performance across MCC, PR-AUC, and ROC-AUC, and was selected as the core classification engine for TargetBoost (further refined via Bayesian hyperparameter optimization using **Optuna**).

---

## 4. Directory Structure

```text
TargetBoost/
├── data/
│   ├── pairs.tsv                     # Sample query-target pairs
│   ├── trfs.fasta                    # Sample tRF sequences
│   └── transcripts.fasta             # Sample target CDS/transcript sequences
├── scripts/
│   ├── feature_extraction.py         # Step 1: 65-feature extraction using RNAduplex
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
