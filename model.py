import os
import sys
import numpy as np
import pandas as pd
import pickle
import xgboost as xgb  # Ensures XGBoost classes unpickle correctly


def run_unlabeled_independent_inference(
    model_path, independent_data_path, output_predictions_path
):
    # =========================================================================
    # 1. INITIAL SYSTEM CHECK & MODEL LOADING
    # =========================================================================
    if not os.path.exists(model_path):
        print(f"❌ Error: Champion model file not found at '{model_path}'")
        sys.exit(1)
    if not os.path.exists(independent_data_path):
        print(
            f"❌ Error: Independent dataset not found at '{independent_data_path}'"
        )
        sys.exit(1)

    print("🔓 Unpickling optimized XGBoost champion model asset...")
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Extract feature names reliably from XGBoost model
    if hasattr(model, "feature_names_in_"):
        model_expected_features = list(model.feature_names_in_)
    elif hasattr(model, "get_booster"):
        model_expected_features = model.get_booster().feature_names
    else:
        model_expected_features = getattr(model, "feature_names_", None)

    if model_expected_features is None:
        print(
            "❌ Error: Could not extract expected feature names from the loaded XGBoost model."
        )
        sys.exit(1)

    print(
        f"📋 Model architecture requires exactly {len(model_expected_features)} structured features."
    )

    # =========================================================================
    # 2. LOAD INDEPENDENT DATA AND APPLY BIOLOGICAL FEATURE ENGINE
    # =========================================================================
    print(f"📖 Reading independent dataset: {independent_data_path}...")
    df = pd.read_csv(independent_data_path, sep="\t")
    df.columns = df.columns.str.strip()

    # Store structural metadata columns separately to paste onto results later
    metadata_cols = [
        "SiteID",
        "Query_tRF",
        "TranscriptID",
        "Target_Structure",
        "tRF_Structure",
    ]
    present_metadata = [col for col in metadata_cols if col in df.columns]

    if present_metadata:
        results_df = df[present_metadata].copy()
    else:
        results_df = pd.DataFrame(index=df.index)

    # Replicate exact training feature selection logic
    target_numeric_features = [
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
    numeric_features = [
        col for col in target_numeric_features if col in df.columns
    ]

    target_categorical = [
        "Base_Identity_2nt_Upstream_Pos2",
        "Base_Identity_2nt_Upstream_Pos1",
        "Base_Identity_2nt_Downstream_Pos1",
        "Base_Identity_2nt_Downstream_Pos2",
        "Position_1_Match",
        "Position_1",
    ]
    categorical_features = [
        col for col in target_categorical if col in df.columns
    ]

    all_dini_base_cols = [
        col for col in df.columns if "Dini_" in col or "Base_" in col
    ]
    dini_base_numeric_features = [
        col for col in all_dini_base_cols if col not in categorical_features
    ]
    numeric_features.extend(dini_base_numeric_features)

    # Separate, process, and combine
    X_numeric_base = df[numeric_features]
    if categorical_features:
        X_categorical_base = pd.get_dummies(
            df[categorical_features], drop_first=True
        )
        X_base = pd.concat([X_numeric_base, X_categorical_base], axis=1)
    else:
        X_base = X_numeric_base.copy()

    # Bind structural target metric
    if "Match_7mer_m8" in df.columns:
        X_full = pd.concat([X_base, df[["Match_7mer_m8"]]], axis=1).astype(float)
    else:
        print(
            "⚠️ Warning: Target seed column 'Match_7mer_m8' missing from independent data. Defaulting to 0.0"
        )
        X_base["Match_7mer_m8"] = 0.0
        X_full = X_base.astype(float)

    # =========================================================================
    # 3. COLUMN ALIGNMENT ENGINE (Prevents shape & layout crashes)
    # =========================================================================
    print("⚙️ Aligning independent data dimensions with model footprint...")

    # Add any columns that the model expects but are missing in this dataset (fill with 0)
    missing_cols = set(model_expected_features) - set(X_full.columns)
    for col in missing_cols:
        X_full[col] = 0.0

    # Reorder columns to perfectly match the training matrix layout
    X_aligned = X_full[model_expected_features]

    # =========================================================================
    # 4. RUN INFERENCE USING OPTIMAL CUSTOM THRESHOLD
    # =========================================================================
    print("🔮 Projecting model vectors and running probability scoring matrix...")
    probabilities = model.predict_proba(X_aligned)[:, 1]

    # Threshold as specified in your script
    OPTIMAL_THRESHOLD = 0.6020
    binary_predictions = (probabilities >= OPTIMAL_THRESHOLD).astype(int)

    # =========================================================================
    # 5. COMPILE AND EXPORT ANNOTATED RESULTS
    # =========================================================================
    results_df["Predicted_Probability"] = probabilities
    results_df["Match_Prediction"] = binary_predictions

    # Check for ground truth label if present; otherwise, output summary statistics
    if "Target_Label" in df.columns:
        results_df["True_Label"] = df["Target_Label"]
        print("ℹ️ Ground truth 'Target_Label' found. Computing test set metrics...")
        from sklearn.metrics import (
            accuracy_score,
            classification_report,
            matthews_corrcoef,
        )

        acc = accuracy_score(df["Target_Label"], binary_predictions)
        mcc = matthews_corrcoef(df["Target_Label"], binary_predictions)
        print(f"  • Accuracy (ACC) : {acc:.4f}")
        print(f"  • Matthews (MCC) : {mcc:.4f}")
    else:
        total_samples = len(results_df)
        positives = int(binary_predictions.sum())
        negatives = total_samples - positives

        print("\n" + "=" * 55)
        print("📊 INFERENCE SUMMARY (Unlabeled Dataset)")
        print("=" * 55)
        print(f"  • Total Samples Processed : {total_samples}")
        print(
            f"  • Predicted Targets (1)  : {positives} ({positives / total_samples * 100:.2f}%)"
        )
        print(
            f"  • Non-Targets (0)        : {negatives} ({negatives / total_samples * 100:.2f}%)"
        )
        print("=" * 55 + "\n")

    results_df.to_csv(output_predictions_path, sep="\t", index=False)
    print(
        f"🎉 Success! Annotated predictions exported to: '{output_predictions_path}'"
    )


if __name__ == "__main__":
    # Configure your paths here
    MODEL_PATH = "./model/best_xgboost_trf_target_model.pkl"
    INDEPENDENT_DATA_PATH = "extracted_features_rnaduplex.tsv"  # Path to your unlabeled input file
    OUTPUT_RESULTS_PATH = "independent_RNAduplex_predictions_output.tsv"

    run_unlabeled_independent_inference(
        MODEL_PATH, INDEPENDENT_DATA_PATH, OUTPUT_RESULTS_PATH
    )
