""" 
Created on 2025.04.25
Updated on 2026.02.17
Author: Lina Jeantin

Extract analysis results from the results dict in a dataframe for plots, linear mixed models & classifiers.

Requirements:
   results: dict containing the results of spectral and complexity analysis (see get_complexity_results.py)
            Organised as: results[sleep_stage][lobe][subregion][marker_name][recording_name]

"""

# %%

import os
import warnings
from collections import defaultdict
import numpy as np
import pandas as pd
import pickle

from get_recordings_info import extract_channel_id_from_recording

# %%


def extract_patient_id(recording_name):
    """Extracts the patient identifier from the recording name."""
    cleaned = recording_name.strip("'").strip('"')
    if len(cleaned) >= 5 and cleaned[2:5].isdigit():
        return cleaned[2:5]
    else:
        warnings.warn(f"Recording name '{recording_name}' does not match expected format for SUBJECT id extraction.",
                RuntimeWarning,)
        return None

def extract_marker_epochs_to_dataframe(results_dict, marker, sleep_stage):
    """  Extracts all values per epoch for a given marker and sleep stage. """
    data = []

    if sleep_stage not in results_dict:
        return pd.DataFrame()
    
    patient_counter = defaultdict(int)

    for lobe in results_dict[sleep_stage]:
        for subregion in results_dict[sleep_stage][lobe]:
            if marker not in results_dict[sleep_stage][lobe][subregion]:
                continue
            for recording, value in results_dict[sleep_stage][lobe][subregion][marker].items():
                patient_id = extract_patient_id(recording)
                if patient_id is None:
                    continue

                try:
                    channel_id = extract_channel_id_from_recording(recording)
                except Exception:
                    channel_id = None
                arr = np.asarray(value).ravel()
                for val in arr:
                    if np.isfinite(val):
                        patient_counter[patient_id] += 1
                        data.append({
                            "value": float(val),
                            "marker": marker,
                            "patient_id": patient_id,
                            "channel_id": channel_id,
                            "lobe": lobe,
                            "subregion": subregion,
                            "recording_id": recording,
                            "sleep_stage": sleep_stage,
                        })
    print(f"Epoch counts per patient:")
    for pid, count in sorted(patient_counter.items()):
        print(f"{pid}: {count} epochs")

    return pd.DataFrame(data)


# %%

with open("path/to/results.pkl", "rb") as f:
    results = pickle.load(f)

sleep_stages = ['W', 'N2', 'N3', 'REM',]
marker = 'komp'
save_dir = '/path/to/save/dataframes/'
os.makedirs(save_dir, exist_ok=True)

df_all_stages = []

for stage in sleep_stages:
    print(f"Computing state : {stage}")

    df_stage = extract_marker_epochs_to_dataframe(
        results_dict=results, # dict containing the results --> see get_complexity_results.py
        marker=marker,
        sleep_stage=stage,
    )  

    stage_filename = os.path.join(save_dir, f'df_{marker}_{stage}.csv')
    df_stage.to_csv(stage_filename, index=False)
    df_all_stages.append(df_stage)

df_full = pd.concat(df_all_stages, ignore_index=True)
final_filename = os.path.join(save_dir, f'df_{marker}_allstages.csv')
df_full.to_csv(final_filename, index=False)
print(f"Job done : {final_filename}")

# %%
