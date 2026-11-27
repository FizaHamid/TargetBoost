#!/usr/bin/env python3

import os
import pandas as pd
import numpy as np
from collections import defaultdict

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = (
    "/lustre/home/sklab202/sourav_ml/sheetal_ml/"
    "simulated_testing/14oct2026/subclass_model_evaluation"
)

INPUT_FILE = os.path.join(
    BASE_DIR,
    "topk_analysis",
    "04_topk_prediction_details.tsv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "topk_analysis",
    "misclassification_analysis"
)

CHUNKSIZE = 250000

CLASSES = [
    "3tRH",
    "5tRH",
    "leader",
    "other",
    "tRF-1",
    "tRF-3",
    "tRF-5",
]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# ACCUMULATORS
# ============================================================

top1_confusion = defaultdict(int)

top3_miss_top1 = defaultdict(int)

top3_miss_top2 = defaultdict(int)

top3_miss_top3 = defaultdict(int)

top3_miss_count = defaultdict(int)

total_count = defaultdict(int)

top1_wrong_count = defaultdict(int)

top1_wrong_prediction = defaultdict(int)

# ============================================================
# PROCESS FILE IN CHUNKS
# ============================================================

print("=" * 80)
print("TOP-K MISCLASSIFICATION ANALYSIS")
print("=" * 80)

reader = pd.read_csv(
    INPUT_FILE,
    sep="\t",
    chunksize=CHUNKSIZE,
    low_memory=False
)

total_records = 0

for chunk_number, df in enumerate(reader, start=1):

    df["true_subclass"] = df["true_subclass"].astype(str)

    df["top1_subclass"] = df["top1_subclass"].astype(str)
    df["top2_subclass"] = df["top2_subclass"].astype(str)
    df["top3_subclass"] = df["top3_subclass"].astype(str)

    df["top3_correct"] = pd.to_numeric(
        df["top3_correct"],
        errors="coerce"
    ).fillna(0)

    df["top1_correct"] = pd.to_numeric(
        df["top1_correct"],
        errors="coerce"
    ).fillna(0)

    total_records += len(df)

    # --------------------------------------------------------
    # 1. TRUE CLASS COUNTS
    # --------------------------------------------------------

    true_counts = df["true_subclass"].value_counts()

    for subclass, count in true_counts.items():
        total_count[subclass] += int(count)

    # --------------------------------------------------------
    # 2. TOP-1 CONFUSION MATRIX
    # --------------------------------------------------------

    cm = pd.crosstab(
        df["true_subclass"],
        df["top1_subclass"]
    )

    for true_class in cm.index:
        for predicted_class in cm.columns:

            value = int(cm.loc[true_class, predicted_class])

            if value > 0:
                top1_confusion[
                    (true_class, predicted_class)
                ] += value

    # --------------------------------------------------------
    # 3. TOP-3 MISSES
    # --------------------------------------------------------

    misses = df[df["top3_correct"] == 0]

    if len(misses) > 0:

        miss_counts = misses["true_subclass"].value_counts()

        for subclass, count in miss_counts.items():
            top3_miss_count[subclass] += int(count)

        # What was predicted as Top-1 for Top-3 failures?
        grouped = misses.groupby(
            ["true_subclass", "top1_subclass"]
        ).size()

        for (true_class, predicted_class), count in grouped.items():
            top3_miss_top1[
                (true_class, predicted_class)
            ] += int(count)

        # Top-2 candidates among Top-3 failures
        grouped = misses.groupby(
            ["true_subclass", "top2_subclass"]
        ).size()

        for (true_class, predicted_class), count in grouped.items():
            top3_miss_top2[
                (true_class, predicted_class)
            ] += int(count)

        # Top-3 candidates among Top-3 failures
        grouped = misses.groupby(
            ["true_subclass", "top3_subclass"]
        ).size()

        for (true_class, predicted_class), count in grouped.items():
            top3_miss_top3[
                (true_class, predicted_class)
            ] += int(count)

    # --------------------------------------------------------
    # 4. TOP-1 WRONG PREDICTIONS
    # --------------------------------------------------------

    wrong_top1 = df[df["top1_correct"] == 0]

    wrong_counts = wrong_top1["true_subclass"].value_counts()

    for subclass, count in wrong_counts.items():
        top1_wrong_count[subclass] += int(count)

    grouped = wrong_top1.groupby(
        ["true_subclass", "top1_subclass"]
    ).size()

    for (true_class, predicted_class), count in grouped.items():
        top1_wrong_prediction[
            (true_class, predicted_class)
        ] += int(count)

    print(
        f"Processed chunk {chunk_number:,} | "
        f"records: {total_records:,}"
    )

# ============================================================
# OUTPUT 1: TOP-1 CONFUSION MATRIX
# ============================================================

cm_df = pd.DataFrame(
    0,
    index=CLASSES,
    columns=CLASSES,
    dtype=np.int64
)

for (true_class, predicted_class), count in top1_confusion.items():

    if true_class in CLASSES and predicted_class in CLASSES:

        cm_df.loc[true_class, predicted_class] = count

cm_path = os.path.join(
    OUTPUT_DIR,
    "01_top1_confusion_matrix.tsv"
)

cm_df.to_csv(cm_path, sep="\t")

# ============================================================
# OUTPUT 2: TOP-3 MISSES BY TRUE CLASS
# ============================================================

miss_rows = []

for subclass in CLASSES:

    total = total_count[subclass]
    misses = top3_miss_count[subclass]

    miss_rows.append({
        "subclass": subclass,
        "total_sequences": total,
        "top3_misses": misses,
        "top3_miss_rate": (
            misses / total if total > 0 else 0
        ),
        "top3_correct": total - misses,
        "top3_recall": (
            (total - misses) / total if total > 0 else 0
        )
    })

miss_summary = pd.DataFrame(miss_rows)

miss_summary = miss_summary.sort_values(
    "top3_miss_rate",
    ascending=False
)

miss_path = os.path.join(
    OUTPUT_DIR,
    "02_top3_miss_summary.tsv"
)

miss_summary.to_csv(
    miss_path,
    sep="\t",
    index=False
)

# ============================================================
# OUTPUT 3: TOP-3 MISS -> TOP-1 PREDICTION
# ============================================================

rows = []

for (true_class, predicted_class), count in top3_miss_top1.items():

    denominator = top3_miss_count[true_class]

    rows.append({
        "true_subclass": true_class,
        "incorrect_top1_prediction": predicted_class,
        "count": count,
        "fraction_of_top3_misses": (
            count / denominator if denominator > 0 else 0
        )
    })

miss_top1_df = pd.DataFrame(rows)

if len(miss_top1_df) > 0:

    miss_top1_df = miss_top1_df.sort_values(
        ["true_subclass", "count"],
        ascending=[True, False]
    )

miss_top1_path = os.path.join(
    OUTPUT_DIR,
    "03_top3_misses_top1_predictions.tsv"
)

miss_top1_df.to_csv(
    miss_top1_path,
    sep="\t",
    index=False
)

# ============================================================
# OUTPUT 4: TOP-3 MISS -> TOP-2 PREDICTION
# ============================================================

rows = []

for (true_class, predicted_class), count in top3_miss_top2.items():

    denominator = top3_miss_count[true_class]

    rows.append({
        "true_subclass": true_class,
        "incorrect_top2_prediction": predicted_class,
        "count": count,
        "fraction_of_top3_misses": (
            count / denominator if denominator > 0 else 0
        )
    })

miss_top2_df = pd.DataFrame(rows)

if len(miss_top2_df) > 0:

    miss_top2_df = miss_top2_df.sort_values(
        ["true_subclass", "count"],
        ascending=[True, False]
    )

miss_top2_path = os.path.join(
    OUTPUT_DIR,
    "04_top3_misses_top2_predictions.tsv"
)

miss_top2_df.to_csv(
    miss_top2_path,
    sep="\t",
    index=False
)

# ============================================================
# OUTPUT 5: TOP-3 MISS -> TOP-3 PREDICTION
# ============================================================

rows = []

for (true_class, predicted_class), count in top3_miss_top3.items():

    denominator = top3_miss_count[true_class]

    rows.append({
        "true_subclass": true_class,
        "incorrect_top3_prediction": predicted_class,
        "count": count,
        "fraction_of_top3_misses": (
            count / denominator if denominator > 0 else 0
        )
    })

miss_top3_df = pd.DataFrame(rows)

if len(miss_top3_df) > 0:

    miss_top3_df = miss_top3_df.sort_values(
        ["true_subclass", "count"],
        ascending=[True, False]
    )

miss_top3_path = os.path.join(
    OUTPUT_DIR,
    "05_top3_misses_top3_predictions.tsv"
)

miss_top3_df.to_csv(
    miss_top3_path,
    sep="\t",
    index=False
)

# ============================================================
# OUTPUT 6: TOP-1 ERROR BREAKDOWN
# ============================================================

rows = []

for (true_class, predicted_class), count in top1_wrong_prediction.items():

    denominator = top1_wrong_count[true_class]

    rows.append({
        "true_subclass": true_class,
        "wrongly_predicted_as": predicted_class,
        "wrong_prediction_count": count,
        "fraction_of_top1_errors": (
            count / denominator if denominator > 0 else 0
        )
    })

top1_error_df = pd.DataFrame(rows)

if len(top1_error_df) > 0:

    top1_error_df = top1_error_df.sort_values(
        ["true_subclass", "wrong_prediction_count"],
        ascending=[True, False]
    )

top1_error_path = os.path.join(
    OUTPUT_DIR,
    "06_top1_error_breakdown.tsv"
)

top1_error_df.to_csv(
    top1_error_path,
    sep="\t",
    index=False
)

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

print(f"Total records analyzed: {total_records:,}")

print("\nTop-3 misses by subclass:")
print(miss_summary.to_string(index=False))

print("\nOutput directory:")
print(OUTPUT_DIR)

print("\nGenerated files:")
print("01_top1_confusion_matrix.tsv")
print("02_top3_miss_summary.tsv")
print("03_top3_misses_top1_predictions.tsv")
print("04_top3_misses_top2_predictions.tsv")
print("05_top3_misses_top3_predictions.tsv")
print("06_top1_error_breakdown.tsv")

print("=" * 80)
