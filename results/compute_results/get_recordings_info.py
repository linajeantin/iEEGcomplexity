"""
Created on 2025.04.23
Updated on 2026.02.16
Author: Lina Jeantin

Code used to get useful information for plots and statistics, including:
   - Patient IDs (extracted from recording names)
   - Channel coordinates in the MNI space

Requirements:
   ChannelInformation.csv : a csv file containing information about channels, available with the MNI-Open dataset. 
                            Columns: 'Channel name', 'Electrode type', 'Patient', 'Hemisphere', 'Region', 'x', 'y', 'z']
   results: a dict containing results from the computation of spectral and complexity markers (see get_complexity_results.py)
                           Organised as: results[sleep_stage][lobe][subregion][marker_name][recording_name]

"""


import warnings
import numpy as np
import pandas as pd
import pickle


# %%

# =======================================================================================
# ==================================       HELPERS        ===============================
# =======================================================================================

def extract_patient_id(recording_name):
    """Extracts the patient identifier from the recording name."""
    cleaned = recording_name.strip("'").strip('"')
    if len(cleaned) >= 5 and cleaned[2:5].isdigit():
        return cleaned[2:5]
    else:
        warnings.warn(f"Recording name '{recording_name}' does not match expected format for SUBJECT id extraction.",
                RuntimeWarning,)
        return None

def extract_channel_id_from_recording(recording_name):
    """Extracts the channel identifier from a recording name."""
    cleaned = recording_name.strip("'").strip('"')
    return cleaned[:-1]

def extract_channel_id_from_csv(channel_name):
    """Cleans up the channel name from the ChannelInformation CSV file."""
    return channel_name.strip("'").strip('"')


def _norm_hemi(x):
    """Normalize hemisphere labels to 'lh'/'rh'."""
    s = str(x).strip().strip("'").lower()
    mapping = {"l": "lh", "left": "lh", "lh": "lh",
               "r": "rh", "right": "rh", "rh": "rh"}
    out = mapping.get(s, None)
    if out is None:
        warnings.warn(f"Unknown Hemisphere value '{x}' in CSV.", RuntimeWarning)
    return out


# %%

# =======================================================================================
# ==================================  GET CHANNEL COORDS  ===============================
# =======================================================================================

with open("path/to/results.pkl", "rb") as f:
    results = pickle.load(f)

channel_info = pd.read_csv("path/to/ChannelInformation.csv")

channel_info["Hemisphere"] = channel_info["Hemisphere"].apply(_norm_hemi)
channel_info["channel_id"] = channel_info["Channel name"].astype(str).apply(extract_channel_id_from_csv)
csv_channels = set(channel_info["channel_id"])

channel_coords = {
    row.channel_id: (float(row.x), float(row.y), float(row.z), row.Hemisphere)
    for row in channel_info.itertuples(index=False)
}

for state in results:
    results_channels = set()
    for lobe in results[state]:
        for region in results[state][lobe]:
            for marker in results[state][lobe][region]:
                for rec in results[state][lobe][region][marker]:
                    chan_id = extract_channel_id_from_recording(rec)
                    results_channels.add(chan_id)
    
    common = csv_channels & results_channels
    only_in_results = results_channels - csv_channels
    only_in_csv = csv_channels - results_channels
    
    print(f"\n Sleep stage : {state}")
    print(f"Total number of channels in results dict : {len(results_channels)}")
    print(f"Number of channels in ChannelInformation CSV file : {len(csv_channels)}")
    print(f"Common channels : {len(common)}")
    print(f"XX In results but not in CSV : {len(only_in_results)}")
    print(f"XX In CSV but not in results : {len(only_in_csv)}")


# Save
with open("path/to/save/channel_coords.pkl", "wb") as f:
    pickle.dump(channel_coords, f)

# %%
