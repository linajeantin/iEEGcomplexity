# %%

from pathlib import Path
import gc
import warnings
from tqdm.auto import tqdm
import numpy as np
import mne
import pickle

from methods_complexity import get_invalid_epochs, compute_complexity


# %%

# =======================================================================================
# ==================================      VARIABLES       ===============================
# =======================================================================================

# Filters
highpass = 0.1
lowpass = 80

# Epochs
epoch_length = 2.0
overlap_dur = 0.0

## Spectral analysis
param_psds = {
    "n_fft": 200,
    "n_overlap": None,
    "n_per_seg": 100,
}

param_freq = { # tau now given in ms, converted in samples within the function
    "delta": {"min": highpass, "max": 4, "tau": 64}, # tau now given in ms, converted in samples within the function
    "theta": {"min": 4, "max": 8, "tau": 32},
    "alpha": {"min": 8, "max": 12, "tau": 16},
    "beta": {"min": 12, "max": 30, "tau": 8},
    "gamma": {"min": 30, "max": lowpass, "tau": 4},
}

# Permutation entropy
kernel_pe = 3  # Size of sub-vectors (order of permutation)

# Kolmogorov complexity
n_bins_komp=32 # number of bins used for the symbolic transform


# %%

# =======================================================================================
# ==================================       HELPERS        ===============================
# =======================================================================================


def annotate_flat_segments(raw, threshold=1e-12, min_duration=1.0, descr_annot='del'):
    """
    Detects flat EEG segments in the MNE Raw object and adds a custom annotation on the flat portion of the Raw object (default: 'del')
    Parameters:
        raw : mne.io.Raw. EEG data (MNE Raw object)
        threshold : float. Maximum standard deviation in a window to consider the segment flat.
        min_duration : float. Minimum duration (in seconds) for a flat segment to be annotated.
    """
    sfreq = raw.info['sfreq']
    win_samples = int(0.2 * sfreq)  # 200 ms window 
    step = int(0.1 * sfreq)  # 100 ms overlap
    data, times = raw.get_data(return_times=True)
    flat_annotations = []

    for ch_idx in range(data.shape[0]):
        flat_mask = np.zeros(data.shape[1], dtype=bool)
        for i in range(0, data.shape[1] - win_samples, step):
            window = data[ch_idx, i:i + win_samples]
            if np.std(window) < threshold:
                flat_mask[i:i + win_samples] = True

        # Look for long, flat sequences
        start_idx = None
        for i in range(len(flat_mask)):
            if flat_mask[i] and start_idx is None:
                start_idx = i
            elif not flat_mask[i] and start_idx is not None:
                end_idx = i
                duration = (end_idx - start_idx) / sfreq
                if duration >= min_duration:
                    onset = times[start_idx]
                    flat_annotations.append((onset, duration, descr_annot))
                start_idx = None
        if start_idx is not None:
            duration = (len(flat_mask) - start_idx) / sfreq
            if duration >= min_duration:
                onset = times[start_idx]
                flat_annotations.append((onset, duration, descr_annot))

    # Merge close annotations (<0.5 s apart)
    merged_annotations = []
    flat_annotations.sort()
    for onset, duration, desc in flat_annotations:
        if not merged_annotations:
            merged_annotations.append([onset, duration])
        else:
            last_onset, last_duration = merged_annotations[-1]
            if onset <= last_onset + last_duration + 0.5:  
                end_time = max(last_onset + last_duration, onset + duration)
                merged_annotations[-1][1] = end_time - last_onset
            else:
                merged_annotations.append([onset, duration])

    # Apply annotations to raw
    if merged_annotations:
        onsets, durations = zip(*merged_annotations)
        annotations = mne.Annotations(onset=onsets, duration=durations,
                                       description=[descr_annot] * len(onsets))
        raw.set_annotations(annotations)


# %%

# =======================================================================================
# ========  COMPUTE SPECTRAL AND COMPLEXITY METRICS -- LOOP ON  EEG .edf FILES ==========
# =======================================================================================


# EEG .edf files were stored in subfolders organised as: path_to_data/sleep_stage/region/hemisphere/edf_file
# Regions start with a region prefix: ['front', 'par', 'occ', 'ins', 'temp']
# e.g.: path_to_data/N2/front_med_sup/GD008Lf_2W.edf
# Adapt loop to your own settings


base_path = Path("path/to/data")
hemisph = 'both' # left, right, or both hemispheres
sleep_stages = ['W', 'N2', 'N3', 'REM']
region_prefixes = ['front', 'par', 'occ', 'ins', 'temp']


results = {} # init dict for results

for sleep_state in sleep_stages:
    print(f"\n========= Sleep stage: {sleep_state} =========")

    results[sleep_state] = {}

    for region_prefix in region_prefixes:
        print(f"\n--- Lobe prefix: {region_prefix} ---")

        # Init dict for current region_prefix
        results[sleep_state][region_prefix] = {}

        state_path = base_path / sleep_state
        if not state_path.exists():
            print(f"!! Missing folder: {state_path}")
            continue

        # List matching regions (e.g. 'front_1', 'front_2', etc.)
        region_list = sorted([
            p for p in state_path.iterdir()
            if p.is_dir() and p.name.startswith(region_prefix) and (p / hemisph).is_dir()
        ], key=lambda p: p.name)

        for region_dir in region_list:
            region = region_dir.name
            print(f"\n → Subregion: {region}")
            region_path = region_dir / hemisph
            edf_files = sorted([p for p in region_path.glob("*.edf") if not p.name.startswith("._")],
                               key=lambda p: p.name)

            region_results = {}


            for file_path in edf_files:
                filename = file_path.stem
                print(f"   ↪ Processing : {filename}")

                try:
                    raw = mne.io.read_raw_edf(str(file_path), preload=True, encoding="latin1", verbose="ERROR")
                except Exception as e:
                    print(f"!! Failed to read {file_path.name}: {e}")
                    continue

                picks_filt = mne.pick_types(raw.info, eeg=True, seeg=True, exclude="bads")
                if len(picks_filt) == 0:
                    warnings.warn("No EEG/SEEG picks found; filtering all channels.", RuntimeWarning)
                    picks_filt = None

                raw.filter(l_freq=highpass, h_freq=lowpass, picks=picks_filt)
                raw.set_annotations(mne.Annotations([], [], []))

                annotate_flat_segments(raw, threshold=1e-12, min_duration=1.0, descr_annot='del')

                epochs = mne.make_fixed_length_epochs(
                    raw, duration=epoch_length, preload=True,
                    reject_by_annotation=False, overlap=overlap_dur
                )

                invalid_epochs = get_invalid_epochs(raw, epochs, bad_labels=['del',])

                markers = compute_complexity(epochs, 
                        invalid_epochs,
                        compute_power =         True,
                        compute_spect_entrop =  True, 
                        compute_perm_entrop =   True,
                        compute_komp =          True,
                        highpass=highpass, 
                        lowpass=lowpass,
                        f_bands = param_freq,
                        param_psds = param_psds,
                        kernel_pe=3,
                        tmin=None, 
                        tmax=None, 
                        backend_pe='python',
                        n_bins_komp=32,
                        backend_komp='python',
                        )

                # Store results: marker → filename → values
                for marker_name, marker_values in markers.items():
                    if marker_name not in region_results:
                        region_results[marker_name] = {}
                    region_results[marker_name][filename] = marker_values

                print(f"   ↪ {filename} processed.")

                del raw, epochs, markers
                gc.collect()

            # Store in full results dict
            results[sleep_state][region_prefix][region] = region_results

print("\n >>> All stages and regions processed.")

# Save
with open("path/to/results.pkl", "wb") as f:
    pickle.dump(results, f)

# %%
