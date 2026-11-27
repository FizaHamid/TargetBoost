#!/usr/bin/env python3

"""
STAGE 3A
tRF / non-tRF MODEL DEVELOPMENT — FIXED DATASET SPLITTING

Purpose
-------
Create one permanent, leakage-safe train / validation / test split
for all subsequent tRF classification and tRF subclassification models.

Tasks
-----
1. Load the cleaned dataset generated after Stage 1.
2. Reconstruct sequence-level labels.
3. Verify dataset integrity.
4. Create a fixed 70/15/15 split.
5. Stratify the split by:
       - tRF vs non-tRF for binary classification
       - tRF subclass for tRF-only classification
6. Save sequence IDs for each partition.
7. Save binary and subclass datasets.
8. Generate split-balance and integrity reports.

IMPORTANT
---------
No feature selection is performed here.

No test-set information is used for:
    - feature selection
    - model training
    - threshold selection
    - calibration
    - hyperparameter tuning

The resulting test set must remain frozen for all later stages.

Author: Ananya Sourav
"""

from pathlib import Path
from datetime import datetime
import hashlib
import json
import sys

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(
    "/lustre/home/sklab202/sourav_ml/sheetal_ml"
)

INPUT_DIR = BASE_DIR / "stage2_features" / "01_dataset"

# Stage 2 master dataset
MASTER_DATASET = INPUT_DIR / "master_dataset.tsv"

OUTPUT_DIR = BASE_DIR / "stage3_model_development"

SPLIT_DIR = OUTPUT_DIR / "01_splits"
DATASET_DIR = OUTPUT_DIR / "03_datasets"
LOG_DIR = OUTPUT_DIR / "logs"

# ------------------------------------------------------------
# Split configuration
# ------------------------------------------------------------

TRAIN_SIZE = 0.70
VALIDATION_SIZE = 0.15
TEST_SIZE = 0.15

RANDOM_SEED = 42

# ------------------------------------------------------------
# Expected labels
# ------------------------------------------------------------

EXPECTED_BINARY_LABELS = {
    "tRF",
    "non-tRF"
}

EXPECTED_SUBCLASSES = {
    "other",
    "tRF-5",
    "tRF-3",
    "5tRH",
    "tRF-1",
    "3tRH",
    "leader"
}


# ============================================================
# DIRECTORY SETUP
# ============================================================

for directory in [
    OUTPUT_DIR,
    SPLIT_DIR,
    DATASET_DIR,
    LOG_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOGGING
# ============================================================

LOG_FILE = LOG_DIR / "stage3_prepare_splits.log"


def log(message=""):
    """Print and append message to log file."""
    print(message)

    with open(LOG_FILE, "a", encoding="utf-8") as handle:
        handle.write(str(message) + "\n")


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def md5_file(path):
    """Return MD5 checksum for a file."""
    md5 = hashlib.md5()

    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            md5.update(chunk)

    return md5.hexdigest()


def find_column(df, candidates, description):
    """
    Find a column using case-insensitive matching.

    Parameters
    ----------
    df : pandas.DataFrame
    candidates : list[str]
    description : str

    Returns
    -------
    str
        Matching column name.
    """

    lower_map = {
        str(column).strip().lower(): column
        for column in df.columns
    }

    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    raise ValueError(
        f"Could not find {description} column.\n"
        f"Expected one of: {candidates}\n"
        f"Available columns:\n{list(df.columns)}"
    )


def normalize_binary_label(value):
    """Normalize binary class labels."""

    value = str(value).strip()

    mapping = {
        "trf": "tRF",
        "tRF": "tRF",
        "positive": "tRF",
        "1": "tRF",

        "non-trf": "non-tRF",
        "nontrf": "non-tRF",
        "non_tRF": "non-tRF",
        "negative": "non-tRF",
        "0": "non-tRF",
    }

    key = value.lower()

    if key in mapping:
        return mapping[key]

    return value


def print_class_counts(df, label_column, title):
    """Print class counts and percentages."""

    log("")
    log("=" * 70)
    log(title)
    log("=" * 70)

    counts = df[label_column].value_counts(dropna=False)

    total = len(df)

    for label, count in counts.items():
        percentage = 100.0 * count / total

        log(
            f"{str(label):20s} "
            f"{count:10,d} "
            f"({percentage:7.3f}%)"
        )

    log(f"{'TOTAL':20s} {total:10,d}")


def check_sequence_overlap(
    train_df,
    validation_df,
    test_df,
    sequence_column,
):
    """
    Verify that exact sequences do not occur across partitions.
    """

    log("")
    log("=" * 70)
    log("CROSS-PARTITION SEQUENCE INTEGRITY")
    log("=" * 70)

    train_sequences = set(
        train_df[sequence_column].astype(str)
    )

    validation_sequences = set(
        validation_df[sequence_column].astype(str)
    )

    test_sequences = set(
        test_df[sequence_column].astype(str)
    )

    train_val = train_sequences & validation_sequences
    train_test = train_sequences & test_sequences
    validation_test = validation_sequences & test_sequences

    log(
        f"Train ∩ Validation : "
        f"{len(train_val):,}"
    )

    log(
        f"Train ∩ Test       : "
        f"{len(train_test):,}"
    )

    log(
        f"Validation ∩ Test  : "
        f"{len(validation_test):,}"
    )

    if train_val:
        raise RuntimeError(
            "Exact sequence overlap detected between "
            "TRAIN and VALIDATION."
        )

    if train_test:
        raise RuntimeError(
            "Exact sequence overlap detected between "
            "TRAIN and TEST."
        )

    if validation_test:
        raise RuntimeError(
            "Exact sequence overlap detected between "
            "VALIDATION and TEST."
        )

    log("PASS: No exact sequence overlap between partitions.")


def save_ids(df, id_column, path):
    """Save sequence IDs."""

    output = (
        df[[id_column]]
        .drop_duplicates()
        .sort_values(id_column)
    )

    output.to_csv(
        path,
        sep="\t",
        index=False,
    )


# ============================================================
# START
# ============================================================

start_time = datetime.now()

log("=" * 70)
log("STAGE 3A — FIXED DATASET SPLITTING")
log("=" * 70)

log(f"Start time      : {start_time}")
log(f"Base directory  : {BASE_DIR}")
log(f"Input dataset   : {MASTER_DATASET}")
log(f"Output directory: {OUTPUT_DIR}")
log(f"Random seed     : {RANDOM_SEED}")
log(
    f"Split           : "
    f"{TRAIN_SIZE:.0%} train / "
    f"{VALIDATION_SIZE:.0%} validation / "
    f"{TEST_SIZE:.0%} test"
)


# ============================================================
# CHECK INPUT
# ============================================================

if not MASTER_DATASET.exists():

    log("")
    log("ERROR: Master dataset not found.")
    log(f"Expected: {MASTER_DATASET}")

    sys.exit(1)


# ============================================================
# LOAD MASTER DATASET
# ============================================================

log("")
log("=" * 70)
log("LOADING MASTER DATASET")
log("=" * 70)

log(f"Reading: {MASTER_DATASET}")

df = pd.read_csv(
    MASTER_DATASET,
    sep="\t",
    low_memory=False,
)

log(f"Rows loaded    : {len(df):,}")
log(f"Columns loaded : {len(df.columns):,}")


# ============================================================
# IDENTIFY IMPORTANT COLUMNS
# ============================================================

id_column = find_column(
    df,
    [
        "sequence_id",
        "seq_id",
        "id",
        "ID",
    ],
    "sequence ID",
)

sequence_column = find_column(
    df,
    [
        "sequence",
        "Sequence",
        "seq",
    ],
    "sequence",
)

binary_column = find_column(
    df,
    [
        "binary_class",
        "binary_label",
        "class",
        "label",
        "label_name",
    ],
    "binary class",
)

subclass_column = find_column(
    df,
    [
        "subclass",
        "tRF_subclass",
        "trf_subclass",
        "subtype",
        "subclass_assignment",
    ],
    "tRF subclass",
)


log("")
log("Identified columns:")
log(f"Sequence ID : {id_column}")
log(f"Sequence    : {sequence_column}")
log(f"Binary      : {binary_column}")
log(f"Subclass    : {subclass_column}")


# ============================================================
# BASIC INTEGRITY
# ============================================================

log("")
log("=" * 70)
log("INPUT DATASET INTEGRITY")
log("=" * 70)

# Missing values

for column in [
    id_column,
    sequence_column,
    binary_column,
    subclass_column,
]:

    missing = df[column].isna().sum()

    log(
        f"Missing {column:20s}: "
        f"{missing:,}"
    )

    if missing > 0:
        raise ValueError(
            f"Missing values detected in required column: "
            f"{column}"
        )


# ------------------------------------------------------------
# Duplicate sequence IDs
# ------------------------------------------------------------

duplicate_ids = (
    df[id_column]
    .duplicated()
    .sum()
)

log(
    f"Duplicate sequence IDs: "
    f"{duplicate_ids:,}"
)

if duplicate_ids > 0:
    raise ValueError(
        "Duplicate sequence IDs detected. "
        "Stage 3 requires one row per sequence."
    )


# ------------------------------------------------------------
# Duplicate exact sequences
# ------------------------------------------------------------

duplicate_sequences = (
    df[sequence_column]
    .astype(str)
    .duplicated()
    .sum()
)

log(
    f"Duplicate exact sequences: "
    f"{duplicate_sequences:,}"
)

if duplicate_sequences > 0:
    raise ValueError(
        "Duplicate exact sequences detected in the Stage 2 "
        "master dataset. Resolve this before Stage 3."
    )


# ============================================================
# NORMALIZE LABELS
# ============================================================

df["binary_label"] = (
    df[binary_column]
    .apply(normalize_binary_label)
)

# Standardize subclass representation

df["subclass_label"] = (
    df[subclass_column]
    .astype(str)
    .str.strip()
)


# ============================================================
# VALIDATE BINARY LABELS
# ============================================================

observed_binary = set(
    df["binary_label"].unique()
)

log("")
log("=" * 70)
log("BINARY LABEL VALIDATION")
log("=" * 70)

log(
    f"Observed labels: "
    f"{sorted(observed_binary)}"
)

unexpected_binary = (
    observed_binary -
    EXPECTED_BINARY_LABELS
)

if unexpected_binary:

    raise ValueError(
        "Unexpected binary labels detected: "
        f"{sorted(unexpected_binary)}"
    )


# ============================================================
# VALIDATE SUBCLASS LABELS
# ============================================================

trf_df = df[
    df["binary_label"] == "tRF"
].copy()

non_trf_df = df[
    df["binary_label"] == "non-tRF"
].copy()

observed_subclasses = set(
    trf_df["subclass_label"].unique()
)

log("")
log("=" * 70)
log("tRF SUBCLASS VALIDATION")
log("=" * 70)

log(
    f"Observed subclasses: "
    f"{sorted(observed_subclasses)}"
)

unexpected_subclasses = (
    observed_subclasses -
    EXPECTED_SUBCLASSES
)

if unexpected_subclasses:

    raise ValueError(
        "Unexpected tRF subclasses detected: "
        f"{sorted(unexpected_subclasses)}"
    )

missing_subclasses = (
    EXPECTED_SUBCLASSES -
    observed_subclasses
)

if missing_subclasses:

    raise ValueError(
        "Expected subclasses missing: "
        f"{sorted(missing_subclasses)}"
    )


# ============================================================
# VERIFY NON-tRF SUBCLASS
# ============================================================

non_trf_subclasses = set(
    non_trf_df["subclass_label"].unique()
)

log("")
log(
    f"Non-tRF subclass labels: "
    f"{sorted(non_trf_subclasses)}"
)

# We don't require a specific representation here because
# non-tRF subclass values are not used for subclassification.


# ============================================================
# DATASET COUNTS
# ============================================================

print_class_counts(
    df,
    "binary_label",
    "FULL DATASET — BINARY DISTRIBUTION",
)

print_class_counts(
    trf_df,
    "subclass_label",
    "FULL tRF DATASET — SUBCLASS DISTRIBUTION",
)


# ============================================================
# SPLIT DATA
# ============================================================

log("")
log("=" * 70)
log("CREATING FIXED SPLITS")
log("=" * 70)

# ------------------------------------------------------------
# First split:
#
# 70% train
# 30% temporary
# ------------------------------------------------------------

train_df, temp_df = train_test_split(
    df,
    train_size=TRAIN_SIZE,
    random_state=RANDOM_SEED,
    stratify=df["binary_label"],
)


# ------------------------------------------------------------
# Second split:
#
# temp = 30%
#
# validation = 15% overall
# test       = 15% overall
#
# Therefore:
#
# validation:test = 50:50
# ------------------------------------------------------------

validation_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=RANDOM_SEED,
    stratify=temp_df["binary_label"],
)


# Reset indexes

train_df = train_df.reset_index(drop=True)
validation_df = validation_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)


# ============================================================
# SPLIT COUNTS
# ============================================================

log("")
log(
    f"Training sequences   : "
    f"{len(train_df):,}"
)

log(
    f"Validation sequences : "
    f"{len(validation_df):,}"
)

log(
    f"Test sequences       : "
    f"{len(test_df):,}"
)

total_split = (
    len(train_df)
    + len(validation_df)
    + len(test_df)
)

if total_split != len(df):

    raise RuntimeError(
        "Split sizes do not sum to the original dataset."
    )


# ============================================================
# BINARY BALANCE
# ============================================================

print_class_counts(
    train_df,
    "binary_label",
    "TRAIN — BINARY DISTRIBUTION",
)

print_class_counts(
    validation_df,
    "binary_label",
    "VALIDATION — BINARY DISTRIBUTION",
)

print_class_counts(
    test_df,
    "binary_label",
    "TEST — BINARY DISTRIBUTION",
)


# ============================================================
# tRF SUBCLASS BALANCE
# ============================================================

train_trf = train_df[
    train_df["binary_label"] == "tRF"
].copy()

validation_trf = validation_df[
    validation_df["binary_label"] == "tRF"
].copy()

test_trf = test_df[
    test_df["binary_label"] == "tRF"
].copy()


print_class_counts(
    train_trf,
    "subclass_label",
    "TRAIN tRF — SUBCLASS DISTRIBUTION",
)

print_class_counts(
    validation_trf,
    "subclass_label",
    "VALIDATION tRF — SUBCLASS DISTRIBUTION",
)

print_class_counts(
    test_trf,
    "subclass_label",
    "TEST tRF — SUBCLASS DISTRIBUTION",
)


# ============================================================
# SUBCLASS REPRESENTATION CHECK
# ============================================================

log("")
log("=" * 70)
log("SUBCLASS REPRESENTATION CHECK")
log("=" * 70)

for name, subset in [
    ("TRAIN", train_trf),
    ("VALIDATION", validation_trf),
    ("TEST", test_trf),
]:

    observed = set(
        subset["subclass_label"].unique()
    )

    missing = (
        EXPECTED_SUBCLASSES -
        observed
    )

    log(
        f"{name:12s} missing subclasses: "
        f"{sorted(missing) if missing else 'NONE'}"
    )

    if missing:

        raise RuntimeError(
            f"{name} split does not contain all expected "
            f"tRF subclasses: {sorted(missing)}"
        )


# ============================================================
# EXACT SEQUENCE OVERLAP CHECK
# ============================================================

check_sequence_overlap(
    train_df,
    validation_df,
    test_df,
    sequence_column,
)


# ============================================================
# SEQUENCE ID OVERLAP CHECK
# ============================================================

log("")
log("=" * 70)
log("SEQUENCE ID OVERLAP CHECK")
log("=" * 70)

train_ids = set(
    train_df[id_column]
)

validation_ids = set(
    validation_df[id_column]
)

test_ids = set(
    test_df[id_column]
)

log(
    f"Train ∩ Validation: "
    f"{len(train_ids & validation_ids):,}"
)

log(
    f"Train ∩ Test      : "
    f"{len(train_ids & test_ids):,}"
)

log(
    f"Validation ∩ Test : "
    f"{len(validation_ids & test_ids):,}"
)

if (
    train_ids & validation_ids
    or train_ids & test_ids
    or validation_ids & test_ids
):

    raise RuntimeError(
        "Sequence ID overlap detected between partitions."
    )

log("PASS: Sequence IDs are mutually exclusive.")


# ============================================================
# SAVE SPLIT IDs
# ============================================================

log("")
log("=" * 70)
log("SAVING SPLIT IDS")
log("=" * 70)

save_ids(
    train_df,
    id_column,
    SPLIT_DIR / "train_ids.tsv",
)

save_ids(
    validation_df,
    id_column,
    SPLIT_DIR / "validation_ids.tsv",
)

save_ids(
    test_df,
    id_column,
    SPLIT_DIR / "test_ids.tsv",
)

log("Saved train_ids.tsv")
log("Saved validation_ids.tsv")
log("Saved test_ids.tsv")


# ============================================================
# SAVE DATASETS
# ============================================================

log("")
log("=" * 70)
log("SAVING BINARY DATASETS")
log("=" * 70)

# Keep original columns + standardized labels

binary_columns = list(df.columns)

train_df[
    binary_columns
].to_csv(
    DATASET_DIR / "train_binary.tsv",
    sep="\t",
    index=False,
)

validation_df[
    binary_columns
].to_csv(
    DATASET_DIR / "validation_binary.tsv",
    sep="\t",
    index=False,
)

test_df[
    binary_columns
].to_csv(
    DATASET_DIR / "test_binary.tsv",
    sep="\t",
    index=False,
)

log("Saved train_binary.tsv")
log("Saved validation_binary.tsv")
log("Saved test_binary.tsv")


# ============================================================
# SAVE tRF SUBCLASS DATASETS
# ============================================================

log("")
log("=" * 70)
log("SAVING tRF SUBCLASS DATASETS")
log("=" * 70)

train_trf.to_csv(
    DATASET_DIR / "train_subclass.tsv",
    sep="\t",
    index=False,
)

validation_trf.to_csv(
    DATASET_DIR / "validation_subclass.tsv",
    sep="\t",
    index=False,
)

test_trf.to_csv(
    DATASET_DIR / "test_subclass.tsv",
    sep="\t",
    index=False,
)

log("Saved train_subclass.tsv")
log("Saved validation_subclass.tsv")
log("Saved test_subclass.tsv")


# ============================================================
# CREATE SPLIT SUMMARY TABLE
# ============================================================

log("")
log("=" * 70)
log("CREATING SPLIT SUMMARY")
log("=" * 70)

summary_rows = []

for split_name, subset in [
    ("FULL", df),
    ("TRAIN", train_df),
    ("VALIDATION", validation_df),
    ("TEST", test_df),
]:

    total = len(subset)

    binary_counts = (
        subset["binary_label"]
        .value_counts()
        .to_dict()
    )

    trf_count = binary_counts.get(
        "tRF",
        0,
    )

    non_trf_count = binary_counts.get(
        "non-tRF",
        0,
    )

    summary_rows.append(
        {
            "split": split_name,
            "total_sequences": total,
            "tRF": trf_count,
            "non_tRF": non_trf_count,
            "tRF_fraction": (
                trf_count / total
                if total > 0
                else np.nan
            ),
            "non_tRF_fraction": (
                non_trf_count / total
                if total > 0
                else np.nan
            ),
        }
    )


split_summary = pd.DataFrame(
    summary_rows
)

split_summary.to_csv(
    SPLIT_DIR / "split_summary.tsv",
    sep="\t",
    index=False,
)

log("Saved split_summary.tsv")


# ============================================================
# SUBCLASS SUMMARY
# ============================================================

subclass_rows = []

for split_name, subset in [
    ("FULL", df[df["binary_label"] == "tRF"]),
    ("TRAIN", train_trf),
    ("VALIDATION", validation_trf),
    ("TEST", test_trf),
]:

    counts = (
        subset["subclass_label"]
        .value_counts()
    )

    total = len(subset)

    for subclass in sorted(
        EXPECTED_SUBCLASSES
    ):

        count = int(
            counts.get(
                subclass,
                0,
            )
        )

        subclass_rows.append(
            {
                "split": split_name,
                "subclass": subclass,
                "count": count,
                "fraction": (
                    count / total
                    if total > 0
                    else np.nan
                ),
            }
        )


subclass_summary = pd.DataFrame(
    subclass_rows
)

subclass_summary.to_csv(
    SPLIT_DIR / "subclass_split_summary.tsv",
    sep="\t",
    index=False,
)

log(
    "Saved subclass_split_summary.tsv"
)


# ============================================================
# SAVE SPLIT INDICES
# ============================================================

np.savez(
    SPLIT_DIR / "split_indices.npz",
    train_indices=train_df.index.to_numpy(),
    validation_indices=validation_df.index.to_numpy(),
    test_indices=test_df.index.to_numpy(),
)

log("Saved split_indices.npz")


# ============================================================
# SAVE CONFIGURATION
# ============================================================

configuration = {
    "stage": "3A_fixed_dataset_splitting",
    "analysis_date": str(datetime.now()),
    "base_directory": str(BASE_DIR),
    "input_dataset": str(MASTER_DATASET),
    "input_dataset_md5": md5_file(MASTER_DATASET),
    "random_seed": RANDOM_SEED,
    "train_fraction": TRAIN_SIZE,
    "validation_fraction": VALIDATION_SIZE,
    "test_fraction": TEST_SIZE,
    "total_sequences": int(len(df)),
    "train_sequences": int(len(train_df)),
    "validation_sequences": int(len(validation_df)),
    "test_sequences": int(len(test_df)),
    "sequence_id_column": id_column,
    "sequence_column": sequence_column,
    "binary_column_original": binary_column,
    "subclass_column_original": subclass_column,
    "expected_binary_labels": sorted(
        EXPECTED_BINARY_LABELS
    ),
    "expected_subclasses": sorted(
        EXPECTED_SUBCLASSES
    ),
    "feature_selection_performed": False,
    "model_training_performed": False,
    "test_set_used_for_model_development": False,
}

with open(
    SPLIT_DIR / "split_configuration.json",
    "w",
    encoding="utf-8",
) as handle:

    json.dump(
        configuration,
        handle,
        indent=4,
    )


log(
    "Saved split_configuration.json"
)


# ============================================================
# FINAL INTEGRITY REPORT
# ============================================================

log("")
log("=" * 70)
log("FINAL INTEGRITY CHECK")
log("=" * 70)

checks = {
    "No missing required values": True,
    "Unique sequence IDs": duplicate_ids == 0,
    "Unique exact sequences": duplicate_sequences == 0,
    "Expected binary labels": not unexpected_binary,
    "Expected subclasses": not unexpected_subclasses,
    "All subclasses represented in train": (
        EXPECTED_SUBCLASSES
        <= set(train_trf["subclass_label"])
    ),
    "All subclasses represented in validation": (
        EXPECTED_SUBCLASSES
        <= set(validation_trf["subclass_label"])
    ),
    "All subclasses represented in test": (
        EXPECTED_SUBCLASSES
        <= set(test_trf["subclass_label"])
    ),
    "No exact sequence overlap": True,
    "No sequence ID overlap": True,
    "All sequences assigned": (
        total_split == len(df)
    ),
}


all_passed = True

for check_name, passed in checks.items():

    status = "PASS" if passed else "FAIL"

    log(
        f"{status:6s}  {check_name}"
    )

    if not passed:
        all_passed = False


if not all_passed:

    raise RuntimeError(
        "One or more Stage 3 integrity checks failed."
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

end_time = datetime.now()

log("")
log("=" * 70)
log("STAGE 3A COMPLETED SUCCESSFULLY")
log("=" * 70)

log(f"End time: {end_time}")

log("")
log("FIXED DATASET SPLIT")
log("-------------------")
log(
    f"Train      : {len(train_df):,} "
    f"({len(train_df)/len(df):.2%})"
)
log(
    f"Validation : {len(validation_df):,} "
    f"({len(validation_df)/len(df):.2%})"
)
log(
    f"Test       : {len(test_df):,} "
    f"({len(test_df)/len(df):.2%})"
)

log("")
log("IMPORTANT")
log("---------")
log(
    "The test set is now frozen and must not be used "
    "for feature selection, hyperparameter tuning, "
    "threshold optimization, calibration, or model selection."
)

log("")
log(
    "Next stage: training-only feature selection."
)

log("")
log(
    f"Output directory: {OUTPUT_DIR}"
)
