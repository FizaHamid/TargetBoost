import warnings
import datetime
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    accuracy_score,
    confusion_matrix,
    average_precision_score
)
from xgboost import XGBClassifier
import optuna

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ==========================================
# HELPER: THRESHOLD OPTIMIZER VIA MCC
# ==========================================
def find_optimal_threshold(y_true, y_proba):
    thresholds = np.linspace(0.01, 0.99, 100)
    best_mcc = -1.0
    opt_thresh = 0.5
    for t in thresholds:
        preds = (y_proba >= t).astype(int)
        mcc = matthews_corrcoef(y_true, preds)
        if mcc > best_mcc:
            best_mcc = mcc
            opt_thresh = t
    return opt_thresh, best_mcc

# Cache to store optimal thresholds across Optuna cross-validation trials
trial_thresholds = {}

# ==========================================
# OPTUNA CV OBJECTIVE FUNCTION
# ==========================================
def objective(trial, X, y):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1500),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'eval_metric': 'logloss',
        'random_state': 42,
        'n_jobs': -1,
        'use_label_encoder': False
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    mcc_scores = []
    fold_thresholds = []

    for train_idx, val_idx in cv.split(X, y):
        X_train_cv, X_val_cv = X.iloc[train_idx], X.iloc[val_idx]
        y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]

        model = XGBClassifier(**params)
        model.fit(
            X_train_cv, y_train_cv,
            eval_set=[(X_val_cv, y_val_cv)],
            verbose=False
        )

        val_proba = model.predict_proba(X_val_cv)[:, 1]
        opt_thresh, val_mcc = find_optimal_threshold(y_val_cv, val_proba)

        mcc_scores.append(val_mcc)
        fold_thresholds.append(opt_thresh)

    mean_mcc = np.mean(mcc_scores)
    trial_thresholds[trial.number] = np.mean(fold_thresholds)
    return mean_mcc

def main():
    # ==========================================
    # 1. LOAD DATA & BUILD FEATURE MATRIX
    # ==========================================
    print("Reading dataset...")
    df = pd.read_csv("labeled_dataset.tsv", sep='\t')
    df.columns = df.columns.str.strip()

    target_numeric_features = [
        'Length_of_tRF', 'GC_Percent_tRF', 'GC_Percent_Seed', 'Weighted_Local_AU_Content',
        'Distance_to_5_End', 'Distance_to_3_End', 'Minimum_Free_Energy',
        'Target_Length_involved_in_base_Pairing', 'Length_Longest_Consecutive_Pairing',
        'Num_Paired_Positions', 'Num_Unpaired_Positions'
    ]
    numeric_features = [col for col in target_numeric_features if col in df.columns]

    target_categorical = [
        'Base_Identity_2nt_Upstream_Pos2', 'Base_Identity_2nt_Upstream_Pos1',
        'Base_Identity_2nt_Downstream_Pos1', 'Base_Identity_2nt_Downstream_Pos2',
        'Position_1_Match', 'Position_1'
    ]
    categorical_features = [col for col in target_categorical if col in df.columns]

    all_dini_base_cols = [col for col in df.columns if 'Dini_' in col or 'Base_' in col]
    dini_base_numeric_features = [col for col in all_dini_base_cols if col not in categorical_features]
    numeric_features.extend(dini_base_numeric_features)

    X_numeric_base = df[numeric_features]
    X_categorical_base = pd.get_dummies(df[categorical_features], drop_first=True)
    X_base = pd.concat([X_numeric_base, X_categorical_base], axis=1)

    # Attach selected winning seed feature set: Match_7mer_m8
    target_seed = ['Match_7mer_m8']
    X_full = pd.concat([X_base, df[target_seed]], axis=1).astype(float)
    y = df['Target_Label']

    # 80/20 train-test split aligned with benchmark logic
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X_full, y, test_size=0.20, random_state=42, stratify=y
    )

    print(f"Data Matrix Shapes -> Train/CV Split: {X_train_full.shape}, Test Split: {X_test.shape}")

    # File Logging Setup
    LOG_FILE = "optuna_tuning_history_xgboost_match_7mer_m8.txt"
    MODEL_FILE = "best_xgboost_trf_target_model.pkl"

    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("Optuna Bayesian Hyperparameter Optimization Execution Log (XGBoost)\n")
        f.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*90 + "\n")
        f.write("Trial\tAvg CV MCC\tBest CV MCC\tOptimal Thresh\tParameters Evaluated\n")
        f.write("-"*90 + "\n")

    # ==========================================
    # 2. RUN BAYESIAN HYPERPARAMETER SEARCH
    # ==========================================
    print("\nStarting Optuna Hyperparameter Optimization (30 Trials)...")
    study = optuna.create_study(direction="maximize")

    def tracking_callback(study, trial):
        current_thresh = trial_thresholds.get(trial.number, 0.5)
        msg = f" Trial {trial.number:02d}/30 Completed | Best CV MCC: {study.best_value:.4f} | Trial Score: {trial.value:.4f} | Optimal Thresh: {current_thresh:.4f}"
        print(msg)

        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"{trial.number:02d}\t{trial.value:.5f}\t{study.best_value:.5f}\t{current_thresh:.4f}\t{trial.params}\n")

    study.optimize(lambda trial: objective(trial, X_train_full, y_train_full), n_trials=30, callbacks=[tracking_callback])

    best_opt_thresh = trial_thresholds.get(study.best_trial.number, 0.5)

    # ==========================================
    # 3. RETRAIN CHAMPION & SAVE MODEL
    # ==========================================
    print("\nRetraining final champion model on 80% train split using best hyperparameters...")
    best_params = study.best_params
    best_params.update({
        'eval_metric': 'logloss',
        'random_state': 42,
        'n_jobs': -1,
        'use_label_encoder': False
    })

    final_model = XGBClassifier(**best_params)
    final_model.fit(X_train_full, y_train_full)

    with open(MODEL_FILE, "wb") as f:
        pickle.dump(final_model, f)
    
    # Save best feature names
    with open("best_model_features.txt", "w") as f:
        for feature in X_full.columns:
            f.write(f"{feature}\n")

    print(f"✅ Tuned champion model saved as '{MODEL_FILE}'")

    # ==========================================
    # 4. FINAL TEST EVALUATION
    # ==========================================
    test_proba = final_model.predict_proba(X_test)[:, 1]
    y_test_pred = (test_proba >= best_opt_thresh).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
    sn = tp / (tp + fn) if (tp + fn) > 0 else 0
    sp = tn / (tn + fp) if (tn + fp) > 0 else 0
    acc = accuracy_score(y_test, y_test_pred)
    auc = roc_auc_score(y_test, test_proba)
    pr_auc = average_precision_score(y_test, test_proba)
    mcc = matthews_corrcoef(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)

    print("\n" + "="*60)
    print("🏆 HYPERPARAMETER TUNING COMPLETE")
    print("="*60)
    print("Best Hyperparameters:")
    for k, v in study.best_params.items():
        print(f"  • {k:<20}: {v}")
    print(f"Optimal Threshold (CV Tuned)   : {best_opt_thresh:.4f}")
    print("="*60)
    print("\nFinal Test Metrics (Evaluated on Unseen Split):")
    print(f"  • Accuracy (ACC) : {acc:.4f}")
    print(f"  • ROC-AUC        : {auc:.4f}")
    print(f"  • PR-AUC         : {pr_auc:.4f}")
    print(f"  • Sensitivity(Sn): {sn:.4f}")
    print(f"  • Specificity(Sp): {sp:.4f}")
    print(f"  • F1-Score       : {f1:.4f}")
    print(f"  • Matthews (MCC) : {mcc:.4f}  <-- Tuned Model Score")

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write("\n" + "="*60 + "\n")
        f.write("FINAL PERFORMANCE SUMMARY ON TEST SPLIT\n")
        f.write(f"Optimal Threshold: {best_opt_thresh:.4f}\n")
        f.write(f"Test ACC: {acc:.4f} | Test ROC-AUC: {auc:.4f} | Test PR-AUC: {pr_auc:.4f} | Test MCC: {mcc:.4f}\n")
        f.write("="*60 + "\n")

    print(f"✅ Tuning log saved to '{LOG_FILE}'")

if __name__ == "__main__":
    main()
