#!/usr/bin/env python3

"""
Stage 3B — Complete tRF Model Development

Tasks
-----
1. Load fixed Stage 3A train/validation/test splits.
2. Align Stage 2 engineered features by sequence_id.
3. Remove target-derived/leakage features.
4. Perform training-only variance filtering.
5. Perform training-only Spearman correlation pruning.
6. Standardize engineered features using training data only.
7. Calculate training-only class weights.
8. Train/evaluate:
       Binary:
           - XGBoost
           - SVM-RBF
           - 1D-CNN

       Subclass:
           - XGBoost
           - Random Forest
           - 1D-CNN

9. Perform 5-fold stratified CV on training data.
10. Generate validation predictions.
11. Select the final model using validation macro-F1.
12. Optimize binary tRF threshold on validation only.
13. Evaluate ONLY the locked final models on frozen test data.
14. Save model files.
15. Save ROC/PR/CV/confusion/per-class/threshold data for plotting.

No SMOTE.
No synthetic sequence generation.
No test-set model selection.
No target-derived features.
"""

import os
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    log_loss,
)
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ============================================================
# OPTIONAL CNN
# ============================================================

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(
    "/lustre/home/sklab202/sourav_ml/sheetal_ml"
)

STAGE2_DIR = BASE_DIR / "stage2_features"
STAGE3_DIR = BASE_DIR / "stage3_model_development"

DATASET_DIR = STAGE3_DIR / "03_datasets"

FEATURE_DIR = STAGE3_DIR / "02_feature_selection"
WEIGHT_DIR = STAGE3_DIR / "04_class_weights"
CV_DIR = STAGE3_DIR / "05_cross_validation"
MODEL_DIR = STAGE3_DIR / "06_models"
VAL_DIR = STAGE3_DIR / "07_validation"
TEST_DIR = STAGE3_DIR / "08_test"
PLOT_DIR = STAGE3_DIR / "09_plot_data"
LOG_DIR = STAGE3_DIR / "10_logs"

for d in [
    FEATURE_DIR,
    WEIGHT_DIR,
    CV_DIR,
    MODEL_DIR,
    VAL_DIR,
    TEST_DIR,
    PLOT_DIR,
    LOG_DIR,
]:
    d.mkdir(parents=True, exist_ok=True)


SEED = 42
N_FOLDS = 5

CORRELATION_THRESHOLD = 0.90

# Binary threshold grid
THRESHOLDS = np.arange(
    0.05,
    0.951,
    0.005
)

# SVM
SVM_MAX_TRAIN = 50000

# CNN
CNN_BATCH_SIZE = 512
CNN_CV_EPOCHS = 10
CNN_FINAL_EPOCHS = 30
CNN_PATIENCE = 5
CNN_LEARNING_RATE = 1e-3
CNN_WEIGHT_DECAY = 1e-4

# Use all training data for CNN
CNN_MAX_LEN = None

np.random.seed(SEED)

if TORCH_AVAILABLE:
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


# ============================================================
# INPUT FILES
# ============================================================

MASTER_FILE = (
    STAGE2_DIR
    / "01_dataset"
    / "master_dataset.tsv"
)

FEATURE_FILE = (
    STAGE2_DIR
    / "02_features"
    / "numeric_feature_matrix.tsv"
)


# ============================================================
# DATA LOADING
# ============================================================

def load_split(name, task):

    file = DATASET_DIR / f"{name}_{task}.tsv"

    if not file.exists():
        raise FileNotFoundError(file)

    df = pd.read_csv(file, sep="\t")

    print(
        f"Loaded {name} {task}: "
        f"{len(df):,} rows"
    )

    return df


def identify_column(df, candidates):

    lower_map = {
        c.lower(): c
        for c in df.columns
    }

    for candidate in candidates:

        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    raise ValueError(
        f"Could not find any of {candidates} "
        f"in columns: {list(df.columns)}"
    )


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("STAGE 3B — COMPLETE MODEL DEVELOPMENT")
print("=" * 70)

print("\nLoading datasets...")

train_bin = load_split("train", "binary")
val_bin = load_split("validation", "binary")
test_bin = load_split("test", "binary")

train_sub = load_split("train", "subclass")
val_sub = load_split("validation", "subclass")
test_sub = load_split("test", "subclass")


ID_COL = identify_column(
    train_bin,
    ["sequence_id", "id", "seq_id"]
)

SEQ_COL = identify_column(
    train_bin,
    ["sequence", "seq"]
)

BIN_LABEL_COL = identify_column(
    train_bin,
    ["label", "binary_label", "label_numeric"]
)

SUBCLASS_COL = identify_column(
    train_sub,
    ["subclass", "subtype", "class"]
)


# ============================================================
# STANDARDIZE TARGET REPRESENTATION
# ============================================================

def make_binary_target(df):

    y = (
        df[BIN_LABEL_COL]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    mapping = {
        "non-trf": 0,
        "non_trf": 0,
        "negative": 0,
        "0": 0,

        "trf": 1,
        "t-rf": 1,
        "positive": 1,
        "1": 1,
    }

    y_mapped = y.map(mapping)

    if y_mapped.isna().any():

        bad_values = (
            df.loc[
                y_mapped.isna(),
                BIN_LABEL_COL
            ]
            .astype(str)
            .unique()
            .tolist()
        )

        raise ValueError(
            "Unexpected binary label values found: "
            f"{bad_values}"
        )

    return y_mapped.astype(int).values


y_train_bin = make_binary_target(train_bin)
y_val_bin = make_binary_target(val_bin)
y_test_bin = make_binary_target(test_bin)

y_train_sub = train_sub[SUBCLASS_COL].astype(str).values
y_val_sub = val_sub[SUBCLASS_COL].astype(str).values
y_test_sub = test_sub[SUBCLASS_COL].astype(str).values


# ============================================================
# VERIFY SPLIT ALIGNMENT
# ============================================================

# Subclass datasets contain only tRF sequences.
# Therefore, every subclass ID must be present in the
# corresponding binary split, but binary splits also contain
# non-tRF sequences.

assert set(train_sub[ID_COL]).issubset(set(train_bin[ID_COL]))
assert set(val_sub[ID_COL]).issubset(set(val_bin[ID_COL]))
assert set(test_sub[ID_COL]).issubset(set(test_bin[ID_COL]))

# Verify there is no train/validation/test overlap
# in the binary dataset.

assert len(
    set(train_bin[ID_COL]) &
    set(val_bin[ID_COL])
) == 0

assert len(
    set(train_bin[ID_COL]) &
    set(test_bin[ID_COL])
) == 0

assert len(
    set(val_bin[ID_COL]) &
    set(test_bin[ID_COL])
) == 0

# Verify there is no train/validation/test overlap
# in the subclass dataset.

assert len(
    set(train_sub[ID_COL]) &
    set(val_sub[ID_COL])
) == 0

assert len(
    set(train_sub[ID_COL]) &
    set(test_sub[ID_COL])
) == 0

assert len(
    set(val_sub[ID_COL]) &
    set(test_sub[ID_COL])
) == 0

print("\nSplit integrity checks passed.")
print(f"  Binary train      : {len(train_bin):,}")
print(f"  Binary validation : {len(val_bin):,}")
print(f"  Binary test       : {len(test_bin):,}")
print(f"  Subclass train    : {len(train_sub):,}")
print(f"  Subclass validation: {len(val_sub):,}")
print(f"  Subclass test     : {len(test_sub):,}")


# ============================================================
# LOAD MASTER DATA
# ============================================================

master = pd.read_csv(
    MASTER_FILE,
    sep="\t"
)

print(
    f"\nCurrent sequence-unique master dataset: "
    f"{len(master):,}"
)

current_ids = set(master[ID_COL].astype(str))


# ============================================================
# LOAD ENGINEERED FEATURES
# ============================================================

print("\nLoading engineered feature matrix...")

features = pd.read_csv(
    FEATURE_FILE,
    sep="\t"
)

features[ID_COL] = features[ID_COL].astype(str)

print(
    f"Original Stage 2 feature rows: "
    f"{len(features):,}"
)

# ------------------------------------------------------------
# IMPORTANT:
# Stage 2 was generated before the final sequence deduplication.
# Restrict features to IDs present in the current master dataset.
# ------------------------------------------------------------

features = features[
    features[ID_COL].isin(current_ids)
].copy()

# Sequence-derived features should be identical for duplicate IDs.
features = features.drop_duplicates(
    subset=[ID_COL],
    keep="first"
)

print(
    f"Feature rows after current-dataset filtering: "
    f"{len(features):,}"
)

assert features[ID_COL].is_unique


# ============================================================
# REMOVE TARGET / LEAKAGE COLUMNS
# ============================================================

NON_FEATURE_EXACT = {
    ID_COL.lower(),
    SEQ_COL.lower(),

    "label",
    "label_numeric",
    "binary_label",
    "binary_target",

    "class",
    "class_numeric",

    "subclass",
    "subclass_numeric",
    "subtype",
    "subtype_numeric",

    "target",
    "target_numeric",
    "y",
}

feature_columns = []

for c in features.columns:

    cl = c.lower()

    if cl in NON_FEATURE_EXACT:
        continue

    # Remove obvious target-derived columns.
    if any(
        token in cl
        for token in [
            "label_numeric",
            "target_numeric",
            "class_numeric",
            "subclass_numeric",
            "binary_target",
        ]
    ):
        continue

    feature_columns.append(c)


# Keep numeric columns only
numeric_features = features[
    feature_columns
].select_dtypes(
    include=[np.number]
).columns.tolist()

print(
    f"\nCandidate engineered features: "
    f"{len(numeric_features):,}"
)

if "label_numeric" in numeric_features:
    raise RuntimeError(
        "TARGET LEAKAGE: label_numeric is still present!"
    )


# ============================================================
# JOIN FEATURES TO SPLITS
# ============================================================

feature_table = features[
    [ID_COL] + numeric_features
].copy()

feature_table = feature_table.set_index(ID_COL)


def get_feature_matrix(df):

    ids = df[ID_COL].astype(str)

    missing = set(ids) - set(feature_table.index)

    if missing:
        raise RuntimeError(
            f"{len(missing)} sequence IDs are missing "
            f"from feature matrix."
        )

    X = feature_table.loc[
        ids
    ].copy()

    X.index = ids

    return X


X_train_raw = get_feature_matrix(train_bin)
X_val_raw = get_feature_matrix(val_bin)
X_test_raw = get_feature_matrix(test_bin)


# ============================================================
# TRAINING-ONLY FEATURE SELECTION
# ============================================================

print("\nPerforming training-only feature selection...")

# Replace infinities
X_train_raw = X_train_raw.replace(
    [np.inf, -np.inf],
    np.nan
)

X_val_raw = X_val_raw.replace(
    [np.inf, -np.inf],
    np.nan
)

X_test_raw = X_test_raw.replace(
    [np.inf, -np.inf],
    np.nan
)

# Median imputation using training data only
train_medians = X_train_raw.median()

X_train_raw = X_train_raw.fillna(train_medians)
X_val_raw = X_val_raw.fillna(train_medians)
X_test_raw = X_test_raw.fillna(train_medians)


# ------------------------------------------------------------
# Variance filter
# ------------------------------------------------------------

variance_selector = VarianceThreshold(
    threshold=0.0
)

variance_selector.fit(
    X_train_raw
)

variance_columns = (
    X_train_raw.columns[
        variance_selector.get_support()
    ].tolist()
)

X_train_var = X_train_raw[
    variance_columns
]

X_val_var = X_val_raw[
    variance_columns
]

X_test_var = X_test_raw[
    variance_columns
]

print(
    f"After variance filter: "
    f"{len(variance_columns):,}"
)


# ------------------------------------------------------------
# Correlation pruning
# ------------------------------------------------------------

corr = X_train_var.corr(
    method="spearman"
).abs()

upper = corr.where(
    np.triu(
        np.ones(corr.shape),
        k=1
    ).astype(bool)
)

drop_columns = [
    column
    for column in upper.columns
    if any(
        upper[column] >
        CORRELATION_THRESHOLD
    )
]

selected_features = [
    c
    for c in variance_columns
    if c not in drop_columns
]

print(
    f"After Spearman correlation pruning: "
    f"{len(selected_features):,}"
)


# Save feature selection
pd.DataFrame({
    "feature": variance_columns,
    "variance_retained": [
        True
        for _ in variance_columns
    ],
    "correlation_pruned": [
        c in drop_columns
        for c in variance_columns
    ],
    "selected": [
        c in selected_features
        for c in variance_columns
    ]
}).to_csv(
    FEATURE_DIR / "feature_selection_summary.tsv",
    sep="\t",
    index=False
)

with open(
    FEATURE_DIR / "selected_features.txt",
    "w"
) as f:

    for c in selected_features:
        f.write(c + "\n")


# ------------------------------------------------------------
# Final selected matrices
# ------------------------------------------------------------

X_train = X_train_var[
    selected_features
].copy()

X_val = X_val_var[
    selected_features
].copy()

X_test = X_test_var[
    selected_features
].copy()


# ============================================================
# STANDARDIZATION
# ============================================================

# ------------------------------------------------------------
# BINARY MODEL MATRICES
# ------------------------------------------------------------

binary_scaler = StandardScaler()

X_train_scaled = binary_scaler.fit_transform(
    X_train
)

X_val_scaled = binary_scaler.transform(
    X_val
)

X_test_scaled = binary_scaler.transform(
    X_test
)

joblib.dump(
    binary_scaler,
    FEATURE_DIR / "binary_standard_scaler.joblib"
)


# ------------------------------------------------------------
# SUBCLASS MODEL MATRICES
# ------------------------------------------------------------
# The subclass datasets contain only tRF sequences.
# Therefore, extract subclass IDs from the corresponding
# binary feature matrices.

train_bin_feature_df = pd.DataFrame(
    X_train,
    index=train_bin[ID_COL].astype(str).values,
    columns=selected_features
)

val_bin_feature_df = pd.DataFrame(
    X_val,
    index=val_bin[ID_COL].astype(str).values,
    columns=selected_features
)

test_bin_feature_df = pd.DataFrame(
    X_test,
    index=test_bin[ID_COL].astype(str).values,
    columns=selected_features
)


# Extract exactly the subclass sequences
X_train_sub = train_bin_feature_df.loc[
    train_sub[ID_COL].astype(str).values
].copy()

X_val_sub = val_bin_feature_df.loc[
    val_sub[ID_COL].astype(str).values
].copy()

X_test_sub = test_bin_feature_df.loc[
    test_sub[ID_COL].astype(str).values
].copy()


# Verify alignment
assert len(X_train_sub) == len(train_sub)
assert len(X_val_sub) == len(val_sub)
assert len(X_test_sub) == len(test_sub)

assert list(X_train_sub.index) == list(
    train_sub[ID_COL].astype(str)
)

assert list(X_val_sub.index) == list(
    val_sub[ID_COL].astype(str)
)

assert list(X_test_sub.index) == list(
    test_sub[ID_COL].astype(str)
)


# ------------------------------------------------------------
# Subclass-specific standardization
# ------------------------------------------------------------

subclass_scaler = StandardScaler()

X_train_sub_scaled = subclass_scaler.fit_transform(
    X_train_sub
)

X_val_sub_scaled = subclass_scaler.transform(
    X_val_sub
)

X_test_sub_scaled = subclass_scaler.transform(
    X_test_sub
)

joblib.dump(
    subclass_scaler,
    FEATURE_DIR / "subclass_standard_scaler.joblib"
)

joblib.dump(
    train_medians,
    FEATURE_DIR / "training_feature_medians.joblib"
)


# ============================================================
# SAVE SELECTED MATRICES
# ============================================================

def save_matrix(path, ids, X):

    out = pd.DataFrame(
        X,
        columns=selected_features
    )

    out.insert(
        0,
        ID_COL,
        ids.values
    )

    out.to_csv(
        path,
        sep="\t",
        index=False
    )


# Binary matrices
save_matrix(
    FEATURE_DIR / "selected_features_train.tsv",
    train_bin[ID_COL],
    X_train_scaled
)

save_matrix(
    FEATURE_DIR / "selected_features_validation.tsv",
    val_bin[ID_COL],
    X_val_scaled
)

save_matrix(
    FEATURE_DIR / "selected_features_test.tsv",
    test_bin[ID_COL],
    X_test_scaled
)


# Subclass matrices
save_matrix(
    FEATURE_DIR / "selected_features_subclass_train.tsv",
    train_sub[ID_COL],
    X_train_sub_scaled
)

save_matrix(
    FEATURE_DIR / "selected_features_subclass_validation.tsv",
    val_sub[ID_COL],
    X_val_sub_scaled
)

save_matrix(
    FEATURE_DIR / "selected_features_subclass_test.tsv",
    test_sub[ID_COL],
    X_test_sub_scaled
)


print("\nFeature matrix dimensions:")
print(
    f"  Binary train       : {X_train_scaled.shape}"
)
print(
    f"  Binary validation  : {X_val_scaled.shape}"
)
print(
    f"  Binary test        : {X_test_scaled.shape}"
)
print(
    f"  Subclass train     : {X_train_sub_scaled.shape}"
)
print(
    f"  Subclass validation: {X_val_sub_scaled.shape}"
)
print(
    f"  Subclass test      : {X_test_sub_scaled.shape}"
)

# ============================================================
# SAVE SELECTED MATRICES
# ============================================================

def save_matrix(path, ids, X):

    out = pd.DataFrame(
        X,
        columns=selected_features
    )

    out.insert(
        0,
        ID_COL,
        ids.values
    )

    out.to_csv(
        path,
        sep="\t",
        index=False
    )


save_matrix(
    FEATURE_DIR / "selected_features_train.tsv",
    train_bin[ID_COL],
    X_train_scaled
)

save_matrix(
    FEATURE_DIR / "selected_features_validation.tsv",
    val_bin[ID_COL],
    X_val_scaled
)

save_matrix(
    FEATURE_DIR / "selected_features_test.tsv",
    test_bin[ID_COL],
    X_test_scaled
)


# ============================================================
# CLASS WEIGHTS
# ============================================================

def calculate_class_weights(y):

    classes, counts = np.unique(
        y,
        return_counts=True
    )

    n = len(y)
    k = len(classes)

    weights = {
        str(cls): float(
            n / (k * count)
        )
        for cls, count in zip(
            classes,
            counts
        )
    }

    return weights


binary_weights = calculate_class_weights(
    y_train_bin
)

subclass_weights = calculate_class_weights(
    y_train_sub
)

pd.DataFrame({
    "class": list(binary_weights.keys()),
    "weight": list(binary_weights.values())
}).to_csv(
    WEIGHT_DIR / "binary_class_weights.tsv",
    sep="\t",
    index=False
)

pd.DataFrame({
    "class": list(subclass_weights.keys()),
    "weight": list(subclass_weights.values())
}).to_csv(
    WEIGHT_DIR / "subclass_class_weights.tsv",
    sep="\t",
    index=False
)

with open(
    WEIGHT_DIR / "class_weights.json",
    "w"
) as f:

    json.dump(
        {
            "binary": binary_weights,
            "subclass": subclass_weights
        },
        f,
        indent=2
    )


# ============================================================
# METRIC FUNCTIONS
# ============================================================

def binary_metrics(
    y_true,
    probability,
    threshold=0.5
):

    pred = (
        probability >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        pred,
        labels=[0, 1]
    ).ravel()

    result = {
        "accuracy": accuracy_score(
            y_true, pred
        ),
        "balanced_accuracy": balanced_accuracy_score(
            y_true, pred
        ),
        "precision": precision_score(
            y_true,
            pred,
            zero_division=0
        ),
        "recall": recall_score(
            y_true,
            pred,
            zero_division=0
        ),
        "specificity": (
            tn / (tn + fp)
            if (tn + fp) > 0
            else 0
        ),
        "f1": f1_score(
            y_true,
            pred,
            zero_division=0
        ),
        "mcc": matthews_corrcoef(
            y_true,
            pred
        ),
        "roc_auc": roc_auc_score(
            y_true,
            probability
        ),
        "pr_auc": average_precision_score(
            y_true,
            probability
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    return result


def multiclass_metrics(
    y_true,
    probability,
    classes
):

    pred_idx = np.argmax(
        probability,
        axis=1
    )

    pred = np.asarray(
        classes
    )[pred_idx]

    y_bin = label_binarize(
        y_true,
        classes=classes
    )

    result = {
        "accuracy": accuracy_score(
            y_true,
            pred
        ),
        "balanced_accuracy": balanced_accuracy_score(
            y_true,
            pred
        ),
        "macro_precision": precision_score(
            y_true,
            pred,
            average="macro",
            zero_division=0
        ),
        "macro_recall": recall_score(
            y_true,
            pred,
            average="macro",
            zero_division=0
        ),
        "macro_f1": f1_score(
            y_true,
            pred,
            average="macro",
            zero_division=0
        ),
        "weighted_f1": f1_score(
            y_true,
            pred,
            average="weighted",
            zero_division=0
        ),
        "mcc": matthews_corrcoef(
            y_true,
            pred
        ),
        "macro_roc_auc": roc_auc_score(
            y_bin,
            probability,
            average="macro",
            multi_class="ovr"
        ),
        "macro_pr_auc": average_precision_score(
            y_bin,
            probability,
            average="macro"
        ),
    }

    return result, pred


# ============================================================
# BINARY XGBOOST
# ============================================================

print("\n" + "=" * 70)
print("BINARY XGBOOST")
print("=" * 70)

xgb_binary = XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=SEED,
    tree_method="hist",
    n_jobs=-1
)

xgb_binary.fit(
    X_train_scaled,
    y_train_bin,
    sample_weight=np.array([
        binary_weights[str(y)]
        for y in y_train_bin
    ])
)

xgb_val_prob = xgb_binary.predict_proba(
    X_val_scaled
)[:, 1]

joblib.dump(
    xgb_binary,
    MODEL_DIR / "binary_XGBoost.joblib"
)


# ============================================================
# BINARY SVM
# ============================================================

print("\n" + "=" * 70)
print("BINARY SVM-RBF")
print("=" * 70)

rng = np.random.default_rng(SEED)

if len(X_train_scaled) > SVM_MAX_TRAIN:

    svm_idx = rng.choice(
        len(X_train_scaled),
        size=SVM_MAX_TRAIN,
        replace=False
    )

else:

    svm_idx = np.arange(
        len(X_train_scaled)
    )

svm_binary = SVC(
    kernel="rbf",
    C=2.0,
    gamma="scale",
    class_weight="balanced",
    probability=True,
    random_state=SEED
)

svm_binary.fit(
    X_train_scaled[svm_idx],
    y_train_bin[svm_idx]
)

svm_val_prob = svm_binary.predict_proba(
    X_val_scaled
)[:, 1]

joblib.dump(
    svm_binary,
    MODEL_DIR / "binary_SVM_RBF.joblib"
)


# ============================================================
# BINARY CV — XGBOOST
# ============================================================

print("\nRunning binary XGBoost 5-fold CV...")

skf = StratifiedKFold(
    n_splits=N_FOLDS,
    shuffle=True,
    random_state=SEED
)

cv_rows = []
cv_predictions = []

for fold, (tr_idx, va_idx) in enumerate(
    skf.split(X_train_scaled, y_train_bin),
    start=1
):

    model = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=SEED + fold,
        tree_method="hist",
        n_jobs=-1
    )

    model.fit(
        X_train_scaled[tr_idx],
        y_train_bin[tr_idx],
        sample_weight=np.array([
            binary_weights[str(y)]
            for y in y_train_bin[tr_idx]
        ])
    )

    prob = model.predict_proba(
        X_train_scaled[va_idx]
    )[:, 1]

    metrics = binary_metrics(
        y_train_bin[va_idx],
        prob,
        threshold=0.5
    )

    metrics["fold"] = fold
    metrics["model"] = "XGBoost"

    cv_rows.append(metrics)

    for i, idx in enumerate(va_idx):

        cv_predictions.append({
            "model": "XGBoost",
            "fold": fold,
            "sequence_id":
                train_bin.iloc[idx][ID_COL],
            "y_true":
                int(y_train_bin[idx]),
            "probability":
                float(prob[i])
        })


# ============================================================
# BINARY CV — SVM
# ============================================================

print("\nRunning binary SVM 5-fold CV...")

for fold, (tr_idx, va_idx) in enumerate(
    skf.split(X_train_scaled, y_train_bin),
    start=1
):

    local_idx = tr_idx

    if len(local_idx) > SVM_MAX_TRAIN:

        local_idx = rng.choice(
            local_idx,
            size=SVM_MAX_TRAIN,
            replace=False
        )

    model = SVC(
        kernel="rbf",
        C=2.0,
        gamma="scale",
        class_weight="balanced",
        probability=False,
        random_state=SEED + fold
    )

    model.fit(
        X_train_scaled[local_idx],
        y_train_bin[local_idx]
    )

    score = model.decision_function(
        X_train_scaled[va_idx]
    )

    probability = 1 / (
        1 + np.exp(-score)
    )

    metrics = binary_metrics(
        y_train_bin[va_idx],
        probability,
        threshold=0.5
    )

    metrics["fold"] = fold
    metrics["model"] = "SVM_RBF"

    cv_rows.append(metrics)

    for i, idx in enumerate(va_idx):

        cv_predictions.append({
            "model": "SVM_RBF",
            "fold": fold,
            "sequence_id":
                train_bin.iloc[idx][ID_COL],
            "y_true":
                int(y_train_bin[idx]),
            "probability":
                float(probability[i])
        })


pd.DataFrame(cv_rows).to_csv(
    CV_DIR / "binary_cv_results.tsv",
    sep="\t",
    index=False
)

pd.DataFrame(cv_predictions).to_csv(
    CV_DIR / "binary_cv_predictions.tsv",
    sep="\t",
    index=False
)


# ============================================================
# CNN
# ============================================================

if not TORCH_AVAILABLE:

    raise ImportError(
        "\nPyTorch is required for the 1D-CNN component.\n"
        "Install a compatible PyTorch version in the "
        "trf_stage1 environment before running Stage 3B."
    )


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(
    f"\nCNN device: {DEVICE}"
)


# ------------------------------------------------------------
# DNA/RNA one-hot encoding
# ------------------------------------------------------------

BASE_TO_INDEX = {
    "A": 0,
    "C": 1,
    "G": 2,
    "U": 3
}


def encode_sequence(seq, max_len):

    seq = str(seq).upper()

    arr = np.zeros(
        (4, max_len),
        dtype=np.float32
    )

    for i, base in enumerate(
        seq[:max_len]
    ):

        if base in BASE_TO_INDEX:

            arr[
                BASE_TO_INDEX[base],
                i
            ] = 1.0

    return arr


def determine_max_length(
    train_sequences
):

    lengths = np.array([
        len(str(s))
        for s in train_sequences
    ])

    if CNN_MAX_LEN is not None:
        return CNN_MAX_LEN

    return int(
        lengths.max()
    )


CNN_MAX_LENGTH = determine_max_length(
    train_bin[SEQ_COL]
)

print(
    f"CNN maximum sequence length: "
    f"{CNN_MAX_LENGTH}"
)


class SequenceDataset(Dataset):

    def __init__(
        self,
        sequences,
        labels,
        max_len
    ):

        self.sequences = list(
            sequences
        )

        self.labels = np.asarray(
            labels
        )

        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        x = encode_sequence(
            self.sequences[idx],
            self.max_len
        )

        y = self.labels[idx]

        return (
            torch.tensor(
                x,
                dtype=torch.float32
            ),
            torch.tensor(
                y,
                dtype=torch.long
            )
        )


class TRFCNN(nn.Module):

    def __init__(
        self,
        n_classes
    ):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv1d(
                4,
                64,
                kernel_size=7,
                padding=3
            ),

            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Conv1d(
                64,
                128,
                kernel_size=5,
                padding=2
            ),

            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Conv1d(
                128,
                128,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Dropout(0.25),

            nn.AdaptiveMaxPool1d(1)
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                128,
                64
            ),

            nn.ReLU(),

            nn.Dropout(0.30),

            nn.Linear(
                64,
                n_classes
            )
        )

    def forward(self, x):

        x = self.features(x)

        return self.classifier(x)


def train_cnn(
    sequences,
    labels,
    n_classes,
    class_weights,
    epochs,
    validation_sequences=None,
    validation_labels=None
):

    model = TRFCNN(
        n_classes
    ).to(DEVICE)

    weights = torch.tensor(
        [
            class_weights.get(
                str(i),
                1.0
            )
            for i in range(n_classes)
        ],
        dtype=torch.float32,
        device=DEVICE
    )

    criterion = nn.CrossEntropyLoss(
        weight=weights
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CNN_LEARNING_RATE,
        weight_decay=CNN_WEIGHT_DECAY
    )

    train_dataset = SequenceDataset(
        sequences,
        labels,
        CNN_MAX_LENGTH
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=CNN_BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    best_state = None
    best_score = -np.inf
    patience_counter = 0

    if validation_sequences is not None:

        val_dataset = SequenceDataset(
            validation_sequences,
            validation_labels,
            CNN_MAX_LENGTH
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=CNN_BATCH_SIZE,
            shuffle=False,
            num_workers=0
        )

    history = []

    for epoch in range(epochs):

        model.train()

        running_loss = 0.0

        for xb, yb in train_loader:

            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad()

            logits = model(xb)

            loss = criterion(
                logits,
                yb
            )

            loss.backward()

            optimizer.step()

            running_loss += (
                loss.item() *
                len(yb)
            )

        train_loss = (
            running_loss /
            len(train_dataset)
        )

        if validation_sequences is not None:

            val_prob, val_true = cnn_predict(
                model,
                val_loader
            )

            if n_classes == 2:

                score = f1_score(
                    val_true,
                    np.argmax(
                        val_prob,
                        axis=1
                    ),
                    average="macro"
                )

            else:

                score = f1_score(
                    val_true,
                    np.argmax(
                        val_prob,
                        axis=1
                    ),
                    average="macro"
                )

            history.append({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "validation_macro_f1": score
            })

            if score > best_score:

                best_score = score

                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }

                patience_counter = 0

            else:

                patience_counter += 1

                if (
                    patience_counter >=
                    CNN_PATIENCE
                ):

                    break

    if best_state is not None:

        model.load_state_dict(
            best_state
        )

    return model, history


def cnn_predict(
    model,
    loader
):

    model.eval()

    probabilities = []
    labels = []

    with torch.no_grad():

        for xb, yb in loader:

            xb = xb.to(DEVICE)

            logits = model(xb)

            prob = torch.softmax(
                logits,
                dim=1
            )

            probabilities.append(
                prob.cpu().numpy()
            )

            labels.append(
                yb.numpy()
            )

    return (
        np.vstack(probabilities),
        np.concatenate(labels)
    )


# ============================================================
# BINARY CNN VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("BINARY 1D-CNN")
print("=" * 70)

binary_cnn_weights = {
    "0": binary_weights["0"],
    "1": binary_weights["1"]
}

binary_cnn, cnn_history = train_cnn(
    train_bin[SEQ_COL].values,
    y_train_bin,
    2,
    binary_cnn_weights,
    CNN_FINAL_EPOCHS,
    val_bin[SEQ_COL].values,
    y_val_bin
)

torch.save(
    {
        "model_state_dict":
            binary_cnn.state_dict(),
        "max_length":
            CNN_MAX_LENGTH,
        "classes":
            [0, 1]
    },
    MODEL_DIR / "binary_CNN.pt"
)

pd.DataFrame(cnn_history).to_csv(
    VAL_DIR / "binary_CNN_training_history.tsv",
    sep="\t",
    index=False
)

val_cnn_dataset = SequenceDataset(
    val_bin[SEQ_COL].values,
    y_val_bin,
    CNN_MAX_LENGTH
)

val_cnn_loader = DataLoader(
    val_cnn_dataset,
    batch_size=CNN_BATCH_SIZE,
    shuffle=False
)

cnn_val_probability_matrix, _ = cnn_predict(
    binary_cnn,
    val_cnn_loader
)

cnn_val_prob = (
    cnn_val_probability_matrix[:, 1]
)


# ============================================================
# BINARY VALIDATION THRESHOLD OPTIMIZATION
# ============================================================

print(
    "\nOptimizing binary thresholds on validation..."
)

threshold_rows = []

candidate_binary_models = {
    "XGBoost": xgb_val_prob,
    "SVM_RBF": svm_val_prob,
    "CNN": cnn_val_prob
}

for model_name, probability in candidate_binary_models.items():

    for threshold in THRESHOLDS:

        metrics = binary_metrics(
            y_val_bin,
            probability,
            threshold
        )

        threshold_rows.append({
            "model": model_name,
            "threshold": threshold,
            **metrics
        })

threshold_df = pd.DataFrame(
    threshold_rows
)

threshold_df.to_csv(
    VAL_DIR / "binary_threshold_results.tsv",
    sep="\t",
    index=False
)


# Select threshold by validation macro-F1.
# Binary macro-F1 is calculated explicitly.
best_thresholds = {}

for model_name in candidate_binary_models:

    sub = threshold_df[
        threshold_df["model"] ==
        model_name
    ].copy()

    sub = sub.sort_values(
        [
            "f1",
            "balanced_accuracy",
            "mcc"
        ],
        ascending=False
    )

    best_thresholds[
        model_name
    ] = float(
        sub.iloc[0]["threshold"]
    )


# ============================================================
# BINARY VALIDATION MODEL COMPARISON
# ============================================================

binary_validation_rows = []

for model_name, probability in candidate_binary_models.items():

    threshold = best_thresholds[
        model_name
    ]

    metrics = binary_metrics(
        y_val_bin,
        probability,
        threshold
    )

    binary_validation_rows.append({
        "task": "binary",
        "model": model_name,
        "threshold": threshold,
        **metrics
    })

binary_validation_df = pd.DataFrame(
    binary_validation_rows
)


# Save validation predictions
binary_val_predictions = pd.DataFrame({
    ID_COL:
        val_bin[ID_COL].values,
    "y_true":
        y_val_bin,
    "XGBoost_probability":
        xgb_val_prob,
    "SVM_RBF_probability":
        svm_val_prob,
    "CNN_probability":
        cnn_val_prob
})

binary_val_predictions.to_csv(
    VAL_DIR / "binary_validation_predictions.tsv",
    sep="\t",
    index=False
)


# ============================================================
# BINARY MODEL SELECTION
# ============================================================

# Pre-specified selection criterion:
# validation macro-F1, followed by balanced accuracy and MCC.

binary_selected = (
    binary_validation_df
    .sort_values(
        [
            "f1",
            "balanced_accuracy",
            "mcc"
        ],
        ascending=False
    )
    .iloc[0]
)

binary_final_model_name = (
    binary_selected["model"]
)

binary_final_threshold = float(
    binary_selected["threshold"]
)

print(
    "\nLOCKED BINARY MODEL:",
    binary_final_model_name
)

print(
    "LOCKED BINARY THRESHOLD:",
    binary_final_threshold
)


# ============================================================
# SUBCLASS XGBOOST
# ============================================================

subclass_classes = sorted(
    np.unique(y_train_sub)
)

class_to_int = {
    cls: i
    for i, cls in enumerate(
        subclass_classes
    )
}

y_train_sub_int = np.array([
    class_to_int[x]
    for x in y_train_sub
])

y_val_sub_int = np.array([
    class_to_int[x]
    for x in y_val_sub
])

y_test_sub_int = np.array([
    class_to_int[x]
    for x in y_test_sub
])

subclass_int_weights = {
    str(class_to_int[k]): v
    for k, v in subclass_weights.items()
}


print("\n" + "=" * 70)
print("SUBCLASS XGBOOST")
print("=" * 70)

xgb_subclass = XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softprob",
    num_class=len(subclass_classes),
    eval_metric="mlogloss",
    random_state=SEED,
    tree_method="hist",
    n_jobs=-1
)

xgb_subclass.fit(
    X_train_sub_scaled,
    y_train_sub_int,
    sample_weight=np.array([
        subclass_int_weights[str(y)]
        for y in y_train_sub_int
    ])
)

xgb_sub_val_prob = xgb_subclass.predict_proba(
    X_val_sub_scaled
)

joblib.dump(
    xgb_subclass,
    MODEL_DIR / "subclass_XGBoost.joblib"
)


# ============================================================
# SUBCLASS RANDOM FOREST
# ============================================================

print("\n" + "=" * 70)
print("SUBCLASS RANDOM FOREST")
print("=" * 70)

rf_subclass = RandomForestClassifier(
    n_estimators=500,
    max_features="sqrt",
    min_samples_leaf=2,
    class_weight="balanced",
    n_jobs=-1,
    random_state=SEED
)

rf_subclass.fit(
    X_train_sub_scaled,
    y_train_sub_int
)

rf_sub_val_prob = rf_subclass.predict_proba(
    X_val_sub_scaled
)

joblib.dump(
    rf_subclass,
    MODEL_DIR / "subclass_RandomForest.joblib"
)


# ============================================================
# SUBCLASS CNN
# ============================================================

print("\n" + "=" * 70)
print("SUBCLASS 1D-CNN")
print("=" * 70)

subclass_cnn, subclass_cnn_history = train_cnn(
    train_sub[SEQ_COL].values,
    y_train_sub_int,
    len(subclass_classes),
    subclass_int_weights,
    CNN_FINAL_EPOCHS,
    val_sub[SEQ_COL].values,
    y_val_sub_int
)

torch.save(
    {
        "model_state_dict":
            subclass_cnn.state_dict(),
        "max_length":
            CNN_MAX_LENGTH,
        "classes":
            subclass_classes
    },
    MODEL_DIR / "subclass_CNN.pt"
)

pd.DataFrame(
    subclass_cnn_history
).to_csv(
    VAL_DIR / "subclass_CNN_training_history.tsv",
    sep="\t",
    index=False
)

sub_val_dataset = SequenceDataset(
    val_sub[SEQ_COL].values,
    y_val_sub_int,
    CNN_MAX_LENGTH
)

sub_val_loader = DataLoader(
    sub_val_dataset,
    batch_size=CNN_BATCH_SIZE,
    shuffle=False
)

cnn_sub_val_prob, _ = cnn_predict(
    subclass_cnn,
    sub_val_loader
)


# ============================================================
# SUBCLASS VALIDATION METRICS
# ============================================================

subclass_validation_rows = []

for model_name, probability in {

    "XGBoost":
        xgb_sub_val_prob,

    "RandomForest":
        rf_sub_val_prob,

    "CNN":
        cnn_sub_val_prob

}.items():

    metrics, prediction = multiclass_metrics(
        y_val_sub_int,
        probability,
        list(range(len(subclass_classes)))
    )

    subclass_validation_rows.append({
        "task": "subclass",
        "model": model_name,
        **metrics
    })


subclass_validation_df = pd.DataFrame(
    subclass_validation_rows
)

subclass_validation_df.to_csv(
    VAL_DIR / "subclass_validation_metrics.tsv",
    sep="\t",
    index=False
)


# ============================================================
# SUBCLASS VALIDATION PREDICTIONS
# ============================================================

subclass_val_prediction_table = pd.DataFrame({
    ID_COL:
        val_sub[ID_COL].values,
    "true_subclass":
        y_val_sub
})

for model_name, probability in {

    "XGBoost":
        xgb_sub_val_prob,

    "RandomForest":
        rf_sub_val_prob,

    "CNN":
        cnn_sub_val_prob

}.items():

    for i, cls in enumerate(
        subclass_classes
    ):

        subclass_val_prediction_table[
            f"{model_name}_{cls}_probability"
        ] = probability[:, i]


subclass_val_prediction_table.to_csv(
    VAL_DIR / "subclass_validation_predictions.tsv",
    sep="\t",
    index=False
)


# ============================================================
# SUBCLASS MODEL SELECTION
# ============================================================

subclass_selected = (
    subclass_validation_df
    .sort_values(
        [
            "macro_f1",
            "balanced_accuracy",
            "mcc"
        ],
        ascending=False
    )
    .iloc[0]
)

subclass_final_model_name = (
    subclass_selected["model"]
)

print(
    "\nLOCKED SUBCLASS MODEL:",
    subclass_final_model_name
)


# ============================================================
# FINAL TEST EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("FROZEN TEST EVALUATION")
print("=" * 70)

# ------------------------------------------------------------
# Binary
# ------------------------------------------------------------

if binary_final_model_name == "XGBoost":

    final_binary_probability = (
        xgb_binary.predict_proba(
            X_test_scaled
        )[:, 1]
    )

elif binary_final_model_name == "SVM_RBF":

    final_binary_probability = (
        svm_binary.predict_proba(
            X_test_scaled
        )[:, 1]
    )

elif binary_final_model_name == "CNN":

    test_cnn_dataset = SequenceDataset(
        test_bin[SEQ_COL].values,
        y_test_bin,
        CNN_MAX_LENGTH
    )

    test_cnn_loader = DataLoader(
        test_cnn_dataset,
        batch_size=CNN_BATCH_SIZE,
        shuffle=False
    )

    final_binary_probability = cnn_predict(
        binary_cnn,
        test_cnn_loader
    )[0][:, 1]

else:

    raise RuntimeError(
        "Unknown binary model"
    )


binary_test_metrics = binary_metrics(
    y_test_bin,
    final_binary_probability,
    binary_final_threshold
)

binary_test_metrics[
    "model"
] = binary_final_model_name

binary_test_metrics[
    "threshold"
] = binary_final_threshold

pd.DataFrame([
    binary_test_metrics
]).to_csv(
    TEST_DIR / "binary_test_metrics.tsv",
    sep="\t",
    index=False
)

binary_test_predictions = pd.DataFrame({

    ID_COL:
        test_bin[ID_COL].values,

    "y_true":
        y_test_bin,

    "probability":
        final_binary_probability,

    "predicted_label":
        (
            final_binary_probability >=
            binary_final_threshold
        ).astype(int)

})

binary_test_predictions.to_csv(
    TEST_DIR / "binary_test_predictions.tsv",
    sep="\t",
    index=False
)


# ------------------------------------------------------------
# Binary ROC
# ------------------------------------------------------------

fpr, tpr, roc_thresholds = roc_curve(
    y_test_bin,
    final_binary_probability
)

pd.DataFrame({
    "fpr": fpr,
    "tpr": tpr,
    "threshold": roc_thresholds
}).to_csv(
    PLOT_DIR / "binary_ROC_curve.tsv",
    sep="\t",
    index=False
)


# ------------------------------------------------------------
# Binary PR
# ------------------------------------------------------------

precision, recall, pr_thresholds = (
    precision_recall_curve(
        y_test_bin,
        final_binary_probability
    )
)

pd.DataFrame({
    "precision": precision,
    "recall": recall,
    "threshold":
        np.append(
            pr_thresholds,
            np.nan
        )
}).to_csv(
    PLOT_DIR / "binary_PR_curve.tsv",
    sep="\t",
    index=False
)


# ------------------------------------------------------------
# Binary confusion matrix
# ------------------------------------------------------------

binary_cm = confusion_matrix(
    y_test_bin,
    (
        final_binary_probability >=
        binary_final_threshold
    ).astype(int),
    labels=[0, 1]
)

pd.DataFrame(
    binary_cm,
    index=[
        "true_non_tRF",
        "true_tRF"
    ],
    columns=[
        "pred_non_tRF",
        "pred_tRF"
    ]
).to_csv(
    PLOT_DIR / "binary_confusion_matrix.tsv",
    sep="\t"
)


# ============================================================
# FINAL SUBCLASS TEST
# ============================================================

if subclass_final_model_name == "XGBoost":

    final_sub_probability = (
        xgb_subclass.predict_proba(
            X_test_sub_scaled
        )
    )

elif subclass_final_model_name == "RandomForest":

    final_sub_probability = (
        rf_subclass.predict_proba(
            X_test_sub_scaled
        )
    )

elif subclass_final_model_name == "CNN":

    test_sub_dataset = SequenceDataset(
        test_sub[SEQ_COL].values,
        y_test_sub_int,
        CNN_MAX_LENGTH
    )

    test_sub_loader = DataLoader(
        test_sub_dataset,
        batch_size=CNN_BATCH_SIZE,
        shuffle=False
    )

    final_sub_probability = cnn_predict(
        subclass_cnn,
        test_sub_loader
    )[0]

else:

    raise RuntimeError(
        "Unknown subclass model"
    )


subclass_test_metrics, subclass_test_pred_int = (
    multiclass_metrics(
        y_test_sub_int,
        final_sub_probability,
        list(range(len(subclass_classes)))
    )
)

subclass_test_pred = np.asarray(
    subclass_classes
)[subclass_test_pred_int]

subclass_test_metrics[
    "model"
] = subclass_final_model_name

pd.DataFrame([
    subclass_test_metrics
]).to_csv(
    TEST_DIR / "subclass_test_metrics.tsv",
    sep="\t",
    index=False
)


# ============================================================
# SUBCLASS TEST PREDICTIONS
# ============================================================

subclass_test_predictions = pd.DataFrame({

    ID_COL:
        test_sub[ID_COL].values,

    "true_subclass":
        y_test_sub,

    "predicted_subclass":
        subclass_test_pred
})

for i, cls in enumerate(
    subclass_classes
):

    subclass_test_predictions[
        f"{cls}_probability"
    ] = final_sub_probability[:, i]


subclass_test_predictions.to_csv(
    TEST_DIR / "subclass_test_predictions.tsv",
    sep="\t",
    index=False
)


# ============================================================
# SUBCLASS PER-CLASS METRICS
# ============================================================

per_class_precision = precision_score(
    y_test_sub_int,
    subclass_test_pred_int,
    average=None,
    labels=list(range(len(subclass_classes))),
    zero_division=0
)

per_class_recall = recall_score(
    y_test_sub_int,
    subclass_test_pred_int,
    average=None,
    labels=list(range(len(subclass_classes))),
    zero_division=0
)

per_class_f1 = f1_score(
    y_test_sub_int,
    subclass_test_pred_int,
    average=None,
    labels=list(range(len(subclass_classes))),
    zero_division=0
)

support = np.bincount(
    y_test_sub_int,
    minlength=len(subclass_classes)
)

per_class_df = pd.DataFrame({

    "subclass":
        subclass_classes,

    "precision":
        per_class_precision,

    "recall":
        per_class_recall,

    "f1":
        per_class_f1,

    "support":
        support

})

per_class_df.to_csv(
    TEST_DIR / "per_class_metrics.tsv",
    sep="\t",
    index=False
)

per_class_df.to_csv(
    PLOT_DIR / "per_class_metrics.tsv",
    sep="\t",
    index=False
)


# ============================================================
# SUBCLASS CONFUSION MATRIX
# ============================================================

sub_cm = confusion_matrix(
    y_test_sub_int,
    subclass_test_pred_int,
    labels=list(range(len(subclass_classes)))
)

pd.DataFrame(
    sub_cm,
    index=[
        f"true_{x}"
        for x in subclass_classes
    ],
    columns=[
        f"pred_{x}"
        for x in subclass_classes
    ]
).to_csv(
    PLOT_DIR / "subclass_confusion_matrix.tsv",
    sep="\t"
)


# ============================================================
# SUBCLASS ROC/PR CURVES
# ============================================================

y_test_sub_bin = label_binarize(
    y_test_sub_int,
    classes=list(range(len(subclass_classes)))
)

roc_rows = []
pr_rows = []

for i, cls in enumerate(
    subclass_classes
):

    fpr, tpr, thresholds = roc_curve(
        y_test_sub_bin[:, i],
        final_sub_probability[:, i]
    )

    for a, b, c in zip(
        fpr,
        tpr,
        thresholds
    ):

        roc_rows.append({
            "subclass": cls,
            "fpr": a,
            "tpr": b,
            "threshold": c
        })

    precision, recall, thresholds = (
        precision_recall_curve(
            y_test_sub_bin[:, i],
            final_sub_probability[:, i]
        )
    )

    thresholds = np.append(
        thresholds,
        np.nan
    )

    for a, b, c in zip(
        precision,
        recall,
        thresholds
    ):

        pr_rows.append({
            "subclass": cls,
            "precision": a,
            "recall": b,
            "threshold": c
        })


pd.DataFrame(roc_rows).to_csv(
    PLOT_DIR / "subclass_ROC_curves.tsv",
    sep="\t",
    index=False
)

pd.DataFrame(pr_rows).to_csv(
    PLOT_DIR / "subclass_PR_curves.tsv",
    sep="\t",
    index=False
)


# ============================================================
# MODEL COMPARISON
# ============================================================

model_comparison = pd.concat([
    binary_validation_df,
    subclass_validation_df
], ignore_index=True)

model_comparison.to_csv(
    PLOT_DIR / "model_comparison_validation.tsv",
    sep="\t",
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

summary = {

    "seed": SEED,

    "n_folds": N_FOLDS,

    "total_master_sequences":
        int(len(master)),

    "training_sequences":
        int(len(train_bin)),

    "validation_sequences":
        int(len(val_bin)),

    "test_sequences":
        int(len(test_bin)),

    "candidate_engineered_features":
        int(len(numeric_features)),

    "variance_retained_features":
        int(len(variance_columns)),

    "correlation_pruned_features":
        int(len(selected_features)),

    "binary_candidate_models": [
        "XGBoost",
        "SVM_RBF",
        "CNN"
    ],

    "subclass_candidate_models": [
        "XGBoost",
        "RandomForest",
        "CNN"
    ],

    "binary_locked_model":
        binary_final_model_name,

    "binary_locked_threshold":
        binary_final_threshold,

    "subclass_locked_model":
        subclass_final_model_name,

    "binary_test_metrics":
        binary_test_metrics,

    "subclass_test_metrics":
        subclass_test_metrics,

    "cnn_device":
        str(DEVICE),

    "cnn_max_length":
        int(CNN_MAX_LENGTH)

}

with open(
    STAGE3_DIR /
    "stage3B_final_summary.json",
    "w"
) as f:

    json.dump(
        summary,
        f,
        indent=2
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("STAGE 3B COMPLETE")
print("=" * 70)

print(
    f"\nSelected engineered features: "
    f"{len(selected_features):,}"
)

print(
    "\nBinary locked model:",
    binary_final_model_name
)

print(
    "Binary threshold:",
    binary_final_threshold
)

print(
    "\nSubclass locked model:",
    subclass_final_model_name
)

print("\nBinary TEST:")
for k, v in binary_test_metrics.items():
    print(f"  {k}: {v}")

print("\nSubclass TEST:")
for k, v in subclass_test_metrics.items():
    print(f"  {k}: {v}")

print(
    "\nOutputs written to:"
)

print(
    STAGE3_DIR
)

print("\n" + "=" * 70)
