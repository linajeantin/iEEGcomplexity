"""
Created on 2025.04.23
Updated on 2026.02.18
Author: Lina Jeantin

Code used to train a random forest classifier to detect sleep stages from markers of iEEG complexity.

Requirements:
    results: dict containing the results of spectral and complexity analysis (see get_complexity_results.py)
            Organised as: results[sleep_stage][lobe][subregion][marker_name][recording_name]

"""

# %%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pickle

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, roc_auc_score, roc_curve, auc
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


# %%

### ============ Extract source data from results dict in a dataframe

def extract_patient_id(recording_name):
    cleaned = str(recording_name).strip("'").strip('"')
    if len(cleaned) >= 5 and cleaned[2:5].isdigit():
        return cleaned[2:5]
    warnings.warn(f"Bad recording format for patient_id: {recording_name}", RuntimeWarning)
    return None

def extract_results_source_data(results):
    """
    One row per (recording_id, sleep_stage, lobe, region)
    Columns: marker mean + marker_std
    """
    rows = {}  # key -> row dict

    for sleep_stage, d_stage in results.items():
        for lobe, d_lobe in d_stage.items():
            for region, d_region in d_lobe.items():
                for marker, d_marker in d_region.items():
                    for recording, value in d_marker.items():
                        key = (recording, sleep_stage, lobe, region)

                        # Create row if not exists
                        if key not in rows:
                            rows[key] = {
                                "recording_id": recording,
                                "sleep_stage": sleep_stage,
                                "lobe": lobe,
                                "region": region,
                            }

                        # Keep behavior: only handle numpy arrays
                        if isinstance(value, np.ndarray):
                            mean_val = np.nanmean(value)
                            std_val = np.nanstd(value)
                        else:
                            # Same intent as your print, but avoids crash
                            # (does not change results when data are arrays)
                            continue

                        if not np.isnan(mean_val):
                            rows[key][marker] = mean_val
                            rows[key][f"{marker}_std"] = std_val

    return pd.DataFrame(rows.values())


# %%

with open("path/to/results.pkl", "rb") as f:
    results = pickle.load(f)

df = extract_results_source_data(results)

markers_wstd = ['pe_gamma', 'pe_gamma_std', 'pe_beta', 'pe_beta_std', 'pe_alpha', 'pe_alpha_std', 'pe_theta', 'pe_theta_std', 'pe_delta', 'pe_delta_std', 
              'komp', 'komp_std', 'spectral_entropy', 'spectral_entropy_std']

features = [c for c in markers_wstd if c in df.columns]
missing = sorted(set(markers_wstd) - set(features))
if missing:
    warnings.warn(f"Missing features in df (will be ignored): {missing}", RuntimeWarning)

# Labels
le = LabelEncoder()
df["sleep_stage_encoded"] = le.fit_transform(df["sleep_stage"])
class_names = list(le.classes_)

X = df[features].copy()
y = df["sleep_stage_encoded"].values

mask_ok = np.isfinite(X.to_numpy()).all(axis=1)
if not mask_ok.all():
    warnings.warn(f"Dropping {(~mask_ok).sum()} rows due to NaNs in features.", RuntimeWarning)
    X = X.loc[mask_ok].reset_index(drop=True)
    y = y[mask_ok]

# Colors for ROC
try:
    customcolors
except NameError:
    customcolors = {'W':'#e64b35','N2':'#4dbbd5','N3':'#3c5488','REM':'#f39b7f'}
micro_color = '#008b45'  # micro-ROC

# Params grid for inner CV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2'],
    'bootstrap': [True, False],
}

# Outer/inner CV
skf_outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
skf_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

rf_base = RandomForestClassifier(random_state=42)

scoring_inner = 'balanced_accuracy'

# Nested CV
oof_true, oof_pred, oof_proba = [], [], []
outer_scores_auc, outer_scores_ba = [], []
outer_best_params, outer_feature_importances = [], []

for fold_id, (tr_idx, te_idx) in enumerate(skf_outer.split(X, y), start=1):
    X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
    y_tr, y_te = y[tr_idx], y[te_idx]

    gscv = GridSearchCV(
        estimator=rf_base,
        param_grid=param_grid,
        cv=skf_inner,
        n_jobs=-1,
        scoring=scoring_inner, 
        refit=True,
        verbose=0
    )
    gscv.fit(X_tr, y_tr)

    best_est = gscv.best_estimator_
    outer_best_params.append(gscv.best_params_)

    y_pred = best_est.predict(X_te)
    y_prob = best_est.predict_proba(X_te)

    try:
        auc_ovr_w = roc_auc_score(y_te, y_prob, multi_class='ovr', average='weighted')
    except ValueError:
        auc_ovr_w = np.nan
    ba = balanced_accuracy_score(y_te, y_pred)

    outer_scores_auc.append(auc_ovr_w)
    outer_scores_ba.append(ba)

    oof_true.append(y_te)
    oof_pred.append(y_pred)
    oof_proba.append(y_prob)

    outer_feature_importances.append(best_est.feature_importances_)

    print(f"[Outer fold {fold_id}] BA={ba:.3f} | AUC OvR (weighted)={auc_ovr_w:.3f} | params={gscv.best_params_}")

oof_true = np.concatenate(oof_true, axis=0)
oof_pred = np.concatenate(oof_pred, axis=0)
oof_proba = np.vstack(oof_proba)

print("\n=== Nested CV (outer) summary ===")
print(f"Mean Balanced Acc      : {np.mean(outer_scores_ba):.3f} ± {np.std(outer_scores_ba):.3f}")
valid_auc = np.array([v for v in outer_scores_auc if np.isfinite(v)])
if valid_auc.size:
    print(f"Mean AUC OvR (weighted): {valid_auc.mean():.3f} ± {valid_auc.std():.3f}")
else:
    print("Mean AUC OvR (weighted): NA")

# ===== Figs

# Confusion matrix
cm = confusion_matrix(oof_true, oof_pred)
plt.figure(figsize=(8,6))
ax = sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=class_names, yticklabels=class_names,
    annot_kws={'fontsize': 14} 
)
ax.set_title('Confusion Matrix (OOF, 5-fold outer CV)', fontsize=18)
ax.set_xlabel('Predicted', fontsize=15)
ax.set_ylabel('True', fontsize=15)
ax.tick_params(axis='both', labelsize=12)
plt.tight_layout()
plt.show()

# OVR + micro
plt.figure(figsize=(10,8))
for i, cname in enumerate(class_names):
    fpr, tpr, _ = roc_curve(oof_true == i, oof_proba[:, i])
    plt.plot(fpr, tpr,
             label=f'{cname} vs Rest (AUC = {auc(fpr,tpr):.2f})',
             color=customcolors.get(cname, None))

y_true_onehot = np.eye(len(class_names))[oof_true]
fpr_mi, tpr_mi, _ = roc_curve(y_true_onehot.ravel(), oof_proba.ravel())
plt.plot(fpr_mi, tpr_mi, color=micro_color, linestyle='--',
         label=f'Micro-average (AUC = {auc(fpr_mi,tpr_mi):.2f})')

plt.plot([0,1],[0,1], color='gray', linestyle='--', label='Chance (AUC = 0.5)')
plt.title('ROC Curves — OvR OOF + Micro (outer CV)')
plt.xlabel('False Positive Rate', fontsize=15); plt.ylabel('True Positive Rate', fontsize=15)
plt.legend(loc='lower right', fontsize=15); 
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

# ---- Feature importances
mean_importances = np.mean(np.vstack(outer_feature_importances), axis=0)
order = np.argsort(mean_importances)[::-1]
features_sorted = np.array(features)[order]
importances_sorted = mean_importances[order]

try:
    featuresdict
except NameError:
    featuresdict = {}  
labels_sorted = [featuresdict.get(f, f) for f in features_sorted]

plt.figure(figsize=(10, max(6, len(labels_sorted)*0.25)))
plt.barh(labels_sorted, importances_sorted)
plt.gca().invert_yaxis()
plt.xlabel('Mean Feature Importance (across outer folds)')
plt.title('Random Forest — Feature Importances (OOF)')
plt.tight_layout(); plt.show()

seen = set()
finalists = []
for p in outer_best_params:
    key = tuple(sorted(p.items()))
    if key not in seen:
        seen.add(key)
        finalists.append(p)

print(f"\nFinal hyperparam candidates (unique): {len(finalists)}")
for i, p in enumerate(finalists, 1):
    print(f"#{i}: {p}")

finalists_wrapped = [{k: [v] for k, v in cand.items()} for cand in finalists]

skf_final = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
gscv_final = GridSearchCV(
    estimator=RandomForestClassifier(random_state=123),
    param_grid=finalists_wrapped,
    cv=skf_final,
    scoring='balanced_accuracy',  
    n_jobs=-1,
    refit=True,
    verbose=1
)

gscv_final.fit(X, y)  
rf_final = gscv_final.best_estimator_

print("\n=== Final model selected on full dataset (CV) ===")
print("Best params:", gscv_final.best_params_)
print(f"CV mean Balanced Acc: {gscv_final.best_score_:.3f}")

# %%
