#!/usr/bin/env python3

from pathlib import Path
import json
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score,
    log_loss,
    brier_score_loss,
)

# ---------------------------------------------------------
# PATHS AND SETTINGS
# ---------------------------------------------------------

BASE = Path("/lustre/home/sklab202/sourav_ml/sheetal_ml")

INPUT_FILE = (
    BASE
    / "simulated_testing/14oct2026/model_evaluation/"
      "all_simulated_predictions.tsv"
)

OUTPUT_DIR = (
    BASE
    / "simulated_testing/14oct2026/model_evaluation/"
      "header_label_evaluation"
)

CHUNK_SIZE = 200_000

# Existing predictions were generated using this threshold.
LOCKED_THRESHOLD = 0.495

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 72)
print("SIMULATED SEQUENCE EVALUATION USING HEADER-DERIVED LABELS")
print("=" * 72)
print(f"Input:       {INPUT_FILE}")
print(f"Output:      {OUTPUT_DIR}")
print(f"Threshold:   {LOCKED_THRESHOLD}")
print(f"Chunk size:  {CHUNK_SIZE:,}")
print()

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

# ---------------------------------------------------------
# READ PREDICTIONS AND EXTRACT TRUE LABELS
# ---------------------------------------------------------

true_labels = []
predicted_labels = []
trf_probabilities = []

total_records = 0
positive_records = 0
negative_records = 0
unrecognized_records = 0

true_label_counts = {"non_tRF": 0, "tRF": 0}
predicted_label_counts = {"non_tRF": 0, "tRF": 0}

required_columns = {
    "simulation_header",
    "tRF_probability",
    "predicted_binary",
}

reader = pd.read_csv(
    INPUT_FILE,
    sep="\t",
    chunksize=CHUNK_SIZE,
    usecols=lambda col: col in required_columns,
    dtype={
        "simulation_header": "string",
        "tRF_probability": "float32",
        "predicted_binary": "int8",
    },
)

for chunk_number, chunk in enumerate(reader, start=1):

    missing_columns = required_columns - set(chunk.columns)
    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    headers = chunk["simulation_header"].str.strip()

    # pos_ = true tRF; neg_ = true non-tRF.
    is_positive = headers.str.startswith("pos_", na=False)
    is_negative = headers.str.startswith("neg_", na=False)

    recognized = is_positive | is_negative

    total_records += len(chunk)
    unrecognized_records += int((~recognized).sum())

    if not recognized.any():
        continue

    valid = chunk.loc[recognized]

    y_true = is_positive.loc[recognized].astype(np.int8).to_numpy()
    y_pred = valid["predicted_binary"].to_numpy(dtype=np.int8)
    y_prob = valid["tRF_probability"].to_numpy(dtype=np.float32)

    if not np.isfinite(y_prob).all():
        raise ValueError(
            f"Non-finite probabilities found in chunk {chunk_number}"
        )

    if ((y_prob < 0) | (y_prob > 1)).any():
        raise ValueError(
            f"Probabilities outside [0, 1] in chunk {chunk_number}"
        )

    if not np.isin(y_pred, [0, 1]).all():
        raise ValueError(
            f"Unexpected predicted_binary values in chunk {chunk_number}"
        )

    # Verify that stored predictions are consistent with the locked
    # probability threshold, allowing for floating-point representation.
    expected_pred = (y_prob >= LOCKED_THRESHOLD).astype(np.int8)

    if not np.array_equal(y_pred, expected_pred):
        mismatches = int(np.sum(y_pred != expected_pred))
        raise ValueError(
            f"{mismatches} stored predictions in chunk {chunk_number} "
            f"do not agree with threshold {LOCKED_THRESHOLD}. "
            "Check the original inference script before proceeding."
        )

    true_labels.append(y_true)
    predicted_labels.append(y_pred)
    trf_probabilities.append(y_prob)

    positive_records += int(y_true.sum())
    negative_records += int(len(y_true) - y_true.sum())

    true_label_counts["tRF"] += int(y_true.sum())
    true_label_counts["non_tRF"] += int(len(y_true) - y_true.sum())

    predicted_label_counts["tRF"] += int(y_pred.sum())
    predicted_label_counts["non_tRF"] += int(len(y_pred) - y_pred.sum())

    if chunk_number % 5 == 0:
        print(
            f"Processed {total_records:,} records; "
            f"recognized labels: "
            f"{positive_records + negative_records:,}"
        )

if not true_labels:
    raise RuntimeError("No records with recognized pos_/neg_ labels found.")

y_true = np.concatenate(true_labels)
y_pred = np.concatenate(predicted_labels)
y_prob = np.concatenate(trf_probabilities)

n = len(y_true)

# ---------------------------------------------------------
# METRICS
# ---------------------------------------------------------

tn, fp, fn, tp = confusion_matrix(
    y_true, y_pred, labels=[0, 1]
).ravel()

metrics = {
    "evaluation_basis": "labels extracted from simulation_header",
    "positive_label": "pos_ = tRF",
    "negative_label": "neg_ = non-tRF",
    "locked_threshold": LOCKED_THRESHOLD,
    "total_input_records": int(total_records),
    "records_evaluated": int(n),
    "unrecognized_header_records": int(unrecognized_records),
    "true_tRF_records": int(positive_records),
    "true_non_tRF_records": int(negative_records),
    "predicted_tRF_records": int(predicted_label_counts["tRF"]),
    "predicted_non_tRF_records": int(predicted_label_counts["non_tRF"]),
    "TN": int(tn),
    "FP": int(fp),
    "FN": int(fn),
    "TP": int(tp),
    "accuracy": float(accuracy_score(y_true, y_pred)),
    "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    "precision_tRF": float(precision_score(
        y_true, y_pred, zero_division=0
    )),
    "recall_tRF_sensitivity": float(recall_score(
        y_true, y_pred, zero_division=0
    )),
    "specificity_non_tRF": float(tn / (tn + fp)) if (tn + fp) else None,
    "F1_tRF": float(f1_score(y_true, y_pred, zero_division=0)),
    "MCC": float(matthews_corrcoef(y_true, y_pred)),
    "ROC_AUC": float(roc_auc_score(y_true, y_prob)),
    "average_precision_PR_AUC": float(
        average_precision_score(y_true, y_prob)
    ),
    "log_loss": float(log_loss(y_true, y_prob, labels=[0, 1])),
    "Brier_score": float(brier_score_loss(y_true, y_prob)),
}

# ---------------------------------------------------------
# SAVE RESULTS
# ---------------------------------------------------------

# Summary JSON
with open(OUTPUT_DIR / "metrics_summary.json", "w") as f:
    json.dump(metrics, f, indent=4)

# Human-readable summary
with open(OUTPUT_DIR / "metrics_summary.txt", "w") as f:
    f.write("=" * 72 + "\n")
    f.write("SIMULATED SEQUENCE EVALUATION: HEADER-DERIVED LABELS\n")
    f.write("=" * 72 + "\n\n")

    f.write("LABEL DEFINITIONS\n")
    f.write("pos_ = true tRF (1)\n")
    f.write("neg_ = true non-tRF (0)\n\n")

    f.write("DATA COUNTS\n")
    f.write(f"Total input records:           {total_records:,}\n")
    f.write(f"Records evaluated:             {n:,}\n")
    f.write(f"Unrecognized header records:   {unrecognized_records:,}\n")
    f.write(f"True tRF records:              {positive_records:,}\n")
    f.write(f"True non-tRF records:           {negative_records:,}\n")
    f.write(
        f"Predicted tRF records:          "
        f"{predicted_label_counts['tRF']:,}\n"
    )
    f.write(
        f"Predicted non-tRF records:      "
        f"{predicted_label_counts['non_tRF']:,}\n"
    )

    f.write(f"\nLOCKED THRESHOLD: {LOCKED_THRESHOLD}\n\n")

    f.write("CONFUSION MATRIX COUNTS\n")
    f.write(f"TN: {tn:,}\n")
    f.write(f"FP: {fp:,}\n")
    f.write(f"FN: {fn:,}\n")
    f.write(f"TP: {tp:,}\n")

    f.write("\nPERFORMANCE METRICS\n")
    for key, value in metrics.items():
        if key in {
            "evaluation_basis",
            "positive_label",
            "negative_label",
        }:
            continue

        if isinstance(value, float):
            f.write(f"{key}: {value:.8f}\n")
        else:
            f.write(f"{key}: {value}\n")

    f.write("\nCLASSIFICATION REPORT\n")
    f.write(
        classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["non_tRF", "tRF"],
            digits=6,
            zero_division=0,
        )
    )

# Classification report TSV
report = classification_report(
    y_true,
    y_pred,
    labels=[0, 1],
    target_names=["non_tRF", "tRF"],
    output_dict=True,
    zero_division=0,
)

pd.DataFrame(report).T.to_csv(
    OUTPUT_DIR / "classification_report.tsv",
    sep="\t",
    index_label="class",
)

# Confusion matrix TSV
cm = pd.DataFrame(
    [[tn, fp], [fn, tp]],
    index=["true_non_tRF", "true_tRF"],
    columns=["predicted_non_tRF", "predicted_tRF"],
)

cm.to_csv(OUTPUT_DIR / "confusion_matrix.tsv", sep="\t")

# Counts for a simple true-vs-predicted class comparison
pd.DataFrame([
    {
        "class": "non_tRF",
        "true_count": negative_records,
        "predicted_count": predicted_label_counts["non_tRF"],
    },
    {
        "class": "tRF",
        "true_count": positive_records,
        "predicted_count": predicted_label_counts["tRF"],
    },
]).to_csv(
    OUTPUT_DIR / "class_counts.tsv",
    sep="\t",
    index=False,
)

# Per-record probabilities and labels for plotting ROC/PR curves later.
# This is much smaller than duplicating the full original TSV.
pd.DataFrame({
    "true_label": y_true,
    "predicted_label": y_pred,
    "tRF_probability": y_prob,
}).to_csv(
    OUTPUT_DIR / "evaluation_scores.tsv",
    sep="\t",
    index=False,
    float_format="%.8g",
)

print("\n" + "=" * 72)
print("EVALUATION COMPLETE")
print("=" * 72)
print(f"Records evaluated: {n:,}")
print(f"Unrecognized headers: {unrecognized_records:,}")
print(f"True tRF: {positive_records:,}")
print(f"True non-tRF: {negative_records:,}")
print()
print(f"TN: {tn:,}   FP: {fp:,}")
print(f"FN: {fn:,}   TP: {tp:,}")
print()
print(f"Accuracy:          {metrics['accuracy']:.6f}")
print(f"Balanced accuracy: {metrics['balanced_accuracy']:.6f}")
print(f"Precision (tRF):  {metrics['precision_tRF']:.6f}")
print(f"Recall (tRF):     {metrics['recall_tRF_sensitivity']:.6f}")
print(f"Specificity:      {metrics['specificity_non_tRF']:.6f}")
print(f"F1-score (tRF):   {metrics['F1_tRF']:.6f}")
print(f"MCC:              {metrics['MCC']:.6f}")
print(f"ROC-AUC:          {metrics['ROC_AUC']:.6f}")
print(f"Average precision:{metrics['average_precision_PR_AUC']:.6f}")
print()
print(f"Results saved to:\n{OUTPUT_DIR}")
