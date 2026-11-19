#!/usr/bin/env python3
"""Predict tRF-target interactions with a trained XGBoost model.

The input is a tab-separated feature table (as produced by
``feature_extraction.py``). The script rebuilds the training feature matrix,
aligns it to the columns the model was trained on, and writes a tab-separated
file with predicted probabilities and binary calls. If a ``Target_Label``
column is present, accuracy and MCC are also reported.

Example
-------
    python model_cli.py --input features.tsv --output predictions.tsv

Security note: models are loaded with ``pickle``. Only load model files from
sources you trust.
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb  # noqa: F401  (required so XGBoost classes unpickle)
from sklearn.metrics import accuracy_score, matthews_corrcoef

logger = logging.getLogger("trf_target_predictor")

DEFAULT_MODEL_PATH = Path("model/best_xgboost_trf_target_model.pkl")
DEFAULT_OUTPUT_PATH = Path("predictions.tsv")
DEFAULT_THRESHOLD = 0.6020  # Decision threshold selected during model tuning

METADATA_COLUMNS = ["SiteID", "Query_tRF", "TranscriptID", "Minimum_Free_Energy"]

NUMERIC_FEATURES = [
    "Length_of_tRF",
    "GC_Percent_tRF",
    "GC_Percent_Seed",
    "Weighted_Local_AU_Content",
    "Distance_to_5_End",
    "Distance_to_3_End",
    "Minimum_Free_Energy",
    "Target_Length_involved_in_base_Pairing",
    "Length_Longest_Consecutive_Pairing",
    "Num_Paired_Positions",
    "Num_Unpaired_Positions",
]

CATEGORICAL_FEATURES = [
    "Base_Identity_2nt_Upstream_Pos2",
    "Base_Identity_2nt_Upstream_Pos1",
    "Base_Identity_2nt_Downstream_Pos1",
    "Base_Identity_2nt_Downstream_Pos2",
    "Position_1_Match",
    "Position_1",
]

SEED_MATCH_COLUMN = "Match_7mer_m8"
LABEL_COLUMN = "Target_Label"


def load_model(model_path: Path):
    """Unpickle the trained model."""
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    logger.info("Loading model from %s", model_path)
    with model_path.open("rb") as handle:
        return pickle.load(handle)


def get_expected_features(model) -> list[str]:
    """Return the feature names the model was trained with."""
    if hasattr(model, "feature_names_in_"):
        names = list(model.feature_names_in_)
    elif hasattr(model, "get_booster"):
        names = model.get_booster().feature_names
    else:
        names = getattr(model, "feature_names_", None)

    if not names:
        raise ValueError("Could not extract feature names from the loaded model.")
    return list(names)


def read_features(path: Path) -> pd.DataFrame:
    """Read the feature table and normalise column names."""
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")
    logger.info("Reading %s", path)
    df = pd.read_csv(path, sep="\t")
    df.columns = df.columns.str.strip()
    return df


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Rebuild the feature matrix using the same logic as training."""
    categorical = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    numeric = [c for c in NUMERIC_FEATURES if c in df.columns]

    # Dinucleotide / base-composition columns are numeric unless categorical.
    extra_numeric = [
        c
        for c in df.columns
        if ("Dini_" in c or "Base_" in c) and c not in categorical
    ]
    numeric = list(dict.fromkeys(numeric + extra_numeric))  # de-duplicate, keep order

    parts = [df[numeric]]
    if categorical:
        # Keep all dummy levels (no drop_first): which level gets dropped
        # would otherwise depend on this dataset. Alignment to the model's
        # columns later discards any levels the model does not use.
        parts.append(pd.get_dummies(df[categorical]))

    if SEED_MATCH_COLUMN in df.columns:
        parts.append(df[[SEED_MATCH_COLUMN]])
    else:
        logger.warning(
            "Column '%s' is missing; defaulting to 0.0.", SEED_MATCH_COLUMN
        )
        parts.append(pd.DataFrame({SEED_MATCH_COLUMN: 0.0}, index=df.index))

    return pd.concat(parts, axis=1).astype(float)


def align_to_model(X: pd.DataFrame, expected: list[str]) -> pd.DataFrame:
    """Add missing columns (as 0) and reorder to the model's training layout."""
    missing = [c for c in expected if c not in X.columns]
    if missing:
        logger.info("Filling %d model feature(s) absent from input with 0.", len(missing))
        X = X.assign(**{c: 0.0 for c in missing})
    return X[expected]


def predict(model, X: pd.DataFrame, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (probabilities, binary calls)."""
    probabilities = model.predict_proba(X)[:, 1]
    return probabilities, (probabilities >= threshold).astype(int)


def report_summary(
    df: pd.DataFrame, calls: np.ndarray, threshold: float
) -> None:
    """Log accuracy/MCC if labels exist, otherwise class counts."""
    if LABEL_COLUMN in df.columns:
        logger.info("Ground-truth labels found; computing metrics.")
        logger.info("Accuracy : %.4f", accuracy_score(df[LABEL_COLUMN], calls))
        logger.info("MCC      : %.4f", matthews_corrcoef(df[LABEL_COLUMN], calls))
        return

    total = len(calls)
    positives = int(calls.sum())
    pct = (lambda n: 100 * n / total if total else 0.0)
    logger.info("Samples processed : %d", total)
    logger.info("Predicted targets : %d (%.2f%%)", positives, pct(positives))
    logger.info("Non-targets       : %d (%.2f%%)", total - positives, pct(total - positives))
    logger.info("Threshold used    : %.4f", threshold)


def run_inference(
    model_path: Path,
    input_path: Path,
    output_path: Path,
    threshold: float = DEFAULT_THRESHOLD,
) -> pd.DataFrame:
    """Full pipeline: load, featurise, align, predict, export."""
    model = load_model(model_path)
    expected = get_expected_features(model)
    logger.info("Model expects %d features.", len(expected))

    df = read_features(input_path)

    metadata = [c for c in METADATA_COLUMNS if c in df.columns]
    results = df[metadata].copy() if metadata else pd.DataFrame(index=df.index)

    X = align_to_model(build_feature_matrix(df), expected)
    probabilities, calls = predict(model, X, threshold)

    results["Predicted_Probability"] = probabilities
    results["Match_Prediction"] = calls
    if LABEL_COLUMN in df.columns:
        results["True_Label"] = df[LABEL_COLUMN]

    report_summary(df, calls, threshold)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, sep="\t", index=False)
    logger.info("Predictions written to %s", output_path)
    return results


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict tRF-target interactions using a trained XGBoost model."
    )
    parser.add_argument(
        "-i", "--input", type=Path, required=True,
        help="Feature TSV file generated by feature_extraction.py",
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=DEFAULT_OUTPUT_PATH,
        help="Output predictions TSV (default: %(default)s)",
    )
    parser.add_argument(
        "-m", "--model", type=Path, default=DEFAULT_MODEL_PATH,
        help="Path to the pickled XGBoost model (default: %(default)s)",
    )
    parser.add_argument(
        "-t", "--threshold", type=float, default=DEFAULT_THRESHOLD,
        help="Probability cutoff for a positive call (default: %(default)s)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        run_inference(args.model, args.input, args.output, args.threshold)
    except (FileNotFoundError, ValueError) as err:
        logger.error("%s", err)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
