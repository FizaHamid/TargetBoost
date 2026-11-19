import warnings
import numpy as np
import pandas as pd
import shap
import joblib
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    accuracy_score,
    confusion_matrix,
    roc_curve,            # Added for ROC Curve extraction
    precision_recall_curve, # Added for PR Curve extraction
    average_precision_score # Added for PR-AUC calculation
)

# Algorithms
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC

warnings.filterwarnings('ignore')

# Helper function to compute optimal threshold by maximizing MCC
def find_optimal_threshold(y_true, y_proba):
    thresholds = np.linspace(0.01, 0.99, 100)
    best_mcc = -1
    opt_thresh = 0.5
    for t in thresholds:
        preds = (y_proba >= t).astype(int)
        mcc = matthews_corrcoef(y_true, preds)
        if mcc > best_mcc:
            best_mcc = mcc
            opt_thresh = t
    return opt_thresh, best_mcc

# ==========================================
# 1. LOAD AND CLEAN YOUR DATASET
# ==========================================
print("Reading dataset...")
df = pd.read_csv("labeled_dataset.tsv", sep='\t')
df.columns = df.columns.str.strip()

# ==========================================
# 2. DEFINE BASELINE AND SEED FEATURES
# ==========================================
seed_features = ['Match_8mer', 'Match_7mer_m8', 'Match_7mer_m1', 'Match_7mer_A1', 'Match_6mer']

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
X_remaining = pd.concat([X_numeric_base, X_categorical_base], axis=1)

y = df['Target_Label']

# ==========================================
# 3. TRAIN / VALIDATE / TEST SPLIT
# ==========================================
X_temp, X_test_base, y_temp, y_test = train_test_split(
    X_remaining, y, test_size=0.20, random_state=42, stratify=y
)

X_train_base, X_val_base, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.20, random_state=42, stratify=y_temp
)

print(f"Data Shapes -> Train: {X_train_base.shape}, Validate: {X_val_base.shape}, Test: {X_test_base.shape}")

# ==========================================
# 4. ALL-INCLUSIVE MODEL DICTIONARY
# ==========================================
models = {
    'CatBoost': CatBoostClassifier(iterations=200, random_seed=42, verbose=0),
    'XGBoost': XGBClassifier(n_estimators=200, random_state=42, eval_metric='logloss'),
    'RandomForest': RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    'LightGBM': LGBMClassifier(n_estimators=200, random_state=42, verbose=-1),
    'SVM': SVC(probability=True, random_state=42, kernel='rbf')
}

available_seeds = [s for s in seed_features if s in df.columns]
feature_sets = {seed: [seed] for seed in available_seeds}
if available_seeds:
    feature_sets['ALL_SEEDS_COMBINED'] = available_seeds

combination_results = []

# Lists to collect coordinate granular arrays across iterations
roc_curve_data_list = []
pr_curve_data_list = []

best_mcc = -1
best_model_obj = None
best_model_name = ""
best_feature_set_name = ""
best_feature_list = []

# ==========================================
# 5. TRAINING, EVALUATION, AND SHAP LOOP
# ==========================================
for feat_name, seed_cols in feature_sets.items():
    print(f"\nProcessing Feature Set: Remaining Features + {feat_name}...")

    X_train_curr = pd.concat([X_train_base, df.loc[X_train_base.index, seed_cols]], axis=1).astype(float)
    X_val_curr = pd.concat([X_val_base, df.loc[X_val_base.index, seed_cols]], axis=1).astype(float)
    X_test_curr = pd.concat([X_test_base, df.loc[X_test_base.index, seed_cols]], axis=1).astype(float)

    for algo_name, base_model in models.items():
        print(f"   ↳ Training {algo_name}...")

        model = clone(base_model)
        
        # Scale handling check for distance/kernel based architectures (SVM)
        if algo_name == 'SVM':
            scaler = StandardScaler()
            X_train_mod = scaler.fit_transform(X_train_curr)
            X_val_mod = scaler.transform(X_val_curr)
            X_test_mod = scaler.transform(X_test_curr)
        else:
            X_train_mod = X_train_curr.values
            X_val_mod = X_val_curr.values
            X_test_mod = X_test_curr.values

        # Model Fitting
        model.fit(X_train_mod, y_train)

        # 1. Tune classification decision threshold on the Validation Subset
        val_proba = model.predict_proba(X_val_mod)[:, 1]
        opt_thresh, _ = find_optimal_threshold(y_val, val_proba)

        # 2. Evaluate strictly on Unseen Holdout Test set using tuned threshold
        test_proba = model.predict_proba(X_test_mod)[:, 1]
        y_test_pred = (test_proba >= opt_thresh).astype(int)

        # Calculate Complete Confusion Matrix Metrics
        tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
        sn = tp / (tp + fn) if (tp + fn) > 0 else 0
        sp = tn / (tn + fp) if (tn + fp) > 0 else 0
        acc = accuracy_score(y_test, y_test_pred)
        auc = roc_auc_score(y_test, test_proba)
        mcc = matthews_corrcoef(y_test, y_test_pred)
        f1 = f1_score(y_test, y_test_pred)
        
        # New Metric additions: PR-AUC (Average Precision score)
        pr_auc = average_precision_score(y_test, test_proba)

        # Generate Curve coordinates points arrays
        fpr_arr, tpr_arr, roc_thresholds = roc_curve(y_test, test_proba)
        precision_arr, recall_arr, pr_thresholds = precision_recall_curve(y_test, test_proba)

        # Append structured arrays to list
        for f, t, th in zip(fpr_arr, tpr_arr, roc_thresholds):
            roc_curve_data_list.append({
                'Feature Set': feat_name, 'Model': algo_name, 'FPR': f, 'TPR': t, 'Threshold': th
            })

        # precision_arr and recall_arr have length len(pr_thresholds) + 1, pad threshold for consistency
        padded_pr_th = np.append(pr_thresholds, np.nan)
        for p, r, th in zip(precision_arr, recall_arr, padded_pr_th):
            pr_curve_data_list.append({
                'Feature Set': feat_name, 'Model': algo_name, 'Precision': p, 'Recall': r, 'Threshold': th
            })

        # Track Champion Architecture Pipeline Configuration
        if mcc > best_mcc:
            best_mcc = mcc
            best_model_obj = (model, scaler) if algo_name == 'SVM' else model
            best_model_name = algo_name
            best_feature_set_name = feat_name
            best_feature_list = list(X_train_curr.columns)

        # 3. Calculate SHAP metrics using optimal explainers
        seed_indices = [X_test_curr.columns.get_loc(s) for s in seed_cols]
        
        if algo_name == 'SVM':
            sub_sample_size = min(100, X_test_mod.shape[0])
            X_test_sub = X_test_mod[:sub_sample_size]
            
            background = shap.kmeans(X_train_mod, 5) 
            explainer = shap.KernelExplainer(model.predict_proba, background)
            shap_vals = explainer.shap_values(X_test_sub)
            if isinstance(shap_vals, list) and len(shap_vals) > 1:
                shap_vals = shap_vals[1]
        else:
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(X_test_mod)
            if isinstance(shap_vals, list) and len(shap_vals) > 1:
                shap_vals = shap_vals[1]

        total_seed_shap = np.abs(shap_vals[:, seed_indices]).sum(axis=1).mean()

        combination_results.append({
            'Feature Set': feat_name,
            'Model': algo_name,
            'Opt_Thresh (Val)': opt_thresh,
            'ACC': acc,
            'AUC': auc,
            'PR-AUC': pr_auc,  # Added to scoreboard track metrics
            'Sn (Sens)': sn,
            'Sp (Spec)': sp,
            'MCC': mcc,
            'F1': f1,
            'Seed_SHAP': total_seed_shap
        })

# ==========================================
# 6. EXPORT CURVES DATA TO CSV FILES
# ==========================================
print("\nExporting Curve Coordinate points vectors...")
pd.DataFrame(roc_curve_data_list).to_csv("roc_curves_data.csv", index=False)
pd.DataFrame(pr_curve_data_list).to_csv("pr_curves_data.csv", index=False)
print("✅ Export complete: 'roc_curves_data.csv' and 'pr_curves_data.csv' generated successfully!")

# ==========================================
# 7. PRINT COMPREHENSIVE SCOREBOARD
# ==========================================
df_results = pd.DataFrame(combination_results)
df_results = df_results.sort_values(by='MCC', ascending=False).reset_index(drop=True)

print("\n" + "="*135)
print(" "*40 + "MULTI-MODEL INDEPENDENT TEST PERFORMANCE SCOREBOARD")
print("="*135)
print(df_results.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print("="*135)
print("* Note: Optimal thresholds tuned on validation set. Performance evaluated on Unseen Test Data.")

# ==========================================
# 8. SAVE WINNING CHAMPION PIPELINE
# ==========================================
print(f"\n🏆 Best Overall Configuration: {best_model_name} with '{best_feature_set_name}' (Test MCC: {best_mcc:.4f})")
model_filename = f"best_{best_model_name.lower()}_trf_target_model.pkl"
joblib.dump(best_model_obj, model_filename)

with open("best_model_features.txt", "w") as f:
    for feature in best_feature_list:
        f.write(f"{feature}\n")

print(f"✅ Champion Model saved securely as '{model_filename}' alongside its profile 'best_model_features.txt'!")
