""" 
Created on 2025.04.25
Updated on 2026.02.17
Author: Lina Jeantin

Plots for mean values of EEG markers of complexity across sleep stages

Requirements:
   df_full: dataframe containing the results of a given marker of EEG complexity across sleep stages
            (see extract_results_df.py)

"""

# %%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import sem, t

# %%


df_full = pd.read_csv("/path/to/save/dataframes/df_full.csv")


marker_name = "spectral_entropy"  # ou "pe_theta", etc.
sleep_stages = ["W", "N2", "N3", "REM",]


x_labels = ['W', 'N2', 'N3', 'REM',]
boxcolors = {'W' : '#e64b35', 
             'N2' : '#4dbbd5', 
             'N3' : '#3c5488', 
             'REM' : '#f39b7f',}


# Compute means according to database hierarchy : mean per recording --> mean per patient --> mean across patients
def compute_group_mean_hierarchy(df, marker, sleep_stage):
    if "marker" in df.columns:
        df_sub = df[(df["marker"] == marker) & (df["sleep_stage"] == sleep_stage)].copy()
    else:
        df_sub = df[df["sleep_stage"] == sleep_stage].copy()

    if df_sub.empty:
        return np.nan, np.nan, np.nan, np.nan, 0, 0

    recording_means = df_sub.groupby('recording_id')['value'].mean() #

    # → Associates each patient to an ID, drop duplicates
    rec_to_patient = df_sub.drop_duplicates('recording_id').set_index('recording_id')['patient_id']
    # → Create dataframe containing patient id and mean of recording
    df_rec = pd.DataFrame({
        'recording_mean': recording_means,
        'patient_id': rec_to_patient
    }).dropna()

    # Mean of all the recordings of a single patient
    patient_means = df_rec.groupby('patient_id')['recording_mean'].mean()

    # → Global arithmetic mean (over all patients) and 95CI from patient_means.
    mean_global = patient_means.mean()
    std_global = patient_means.std()
    n_patients = len(patient_means)
    n_recordings = len(df_rec)
    if n_patients > 1:
        ci95 = sem(patient_means) * t.ppf(0.975, df=n_patients - 1)
        return mean_global, mean_global - ci95, mean_global + ci95, std_global, n_patients, n_recordings
    else:
        return mean_global, mean_global, mean_global, std_global, n_patients, n_recordings

    return mean_global, mean_global - ci95, mean_global + ci95, std_global, n_patients, n_recordings 


means, lower, upper, std, n_pat, n_rec = [], [], [], [], [], []

# Mean and CI for each group and sleep stage
for stage in x_labels:
    m, l, u, s, npat, nrec = compute_group_mean_hierarchy(df=df_full, marker=marker_name, sleep_stage=stage)
    means.append(m)
    lower.append(l)
    upper.append(u)
    std.append(s)
    n_pat.append(npat)
    n_rec.append(nrec)

#### Plot
fig, ax = plt.subplots(figsize=(5, 4))
x = np.arange(len(x_labels))
width = 0.35

for i, stage in enumerate(x_labels):
    ax.errorbar(x[i], means[i],  
                yerr = np.array([[means[i] - lower[i]],
                 [upper[i] - means[i]]]),
                fmt='s', capsize=5, 
                color=boxcolors.get(stage, 'gray'))

y_base = ax.get_ylim()[0] - 0.20 * (ax.get_ylim()[1] - ax.get_ylim()[0])

for i, stage in enumerate(x_labels):
    text = f"µ={means[i]:.3f} \n σ={std[i]:.3f} \n{n_rec[i]} recs\n{n_pat[i]} subj"
    ax.text(i, y_base, text,
            ha='center', va='top', fontsize=8, color='black')

ax.set_xticks(x)
ax.set_xticklabels(x_labels, fontsize=11)
ax.set_ylabel(f"{marker_name} (mean ± CI95)")
ax.set_title(f'{marker_name}')
ax.grid(True, linestyle='--', alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(0.5)
ax.spines['bottom'].set_linewidth(0.5)
plt.tight_layout()
plt.show()

# %%
