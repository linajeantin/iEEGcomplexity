""" 
Created on 2025.04.23
Updated on 2026.02.16
Author: Lina Jeantin

Code used for plotting 3D Brain surfaces using mne.viz Brain objects.

Requirements:
    subjects_dir: path to fsaverage data for 3D plotting
    results: dict containing the results of spectral and complexity analysis (see get_complexity_results.py)
            Organised as: results[sleep_stage][lobe][subregion][marker_name][recording_name]
    channel_coords: a dict containing channel coordinates (see get_recordings_info.py)
    extract_channel_id_from_recording(): from get_recordings_info.py

"""

from pathlib import Path
import warnings
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.colors import Normalize
from scipy.spatial import cKDTree
from scipy.stats import mannwhitneyu, fligner
from statsmodels.stats.multitest import multipletests
from itertools import combinations
from collections import defaultdict
import mne
from mne.viz import Brain
from mne.surface import read_surface
import pickle

from get_recordings_info import extract_channel_id_from_recording

# %%

# =======================================================================================
# ==================================       3D PLOTS       ===============================
# =======================================================================================

subjects_dir = "/path/to/MNE-fsaverage-data"
subject = "fsaverage"

# load results & channel coords
with open("path/to/results.pkl", "rb") as f:
    results = pickle.load(f)

with open("path/to/save/channel_coords.pkl", "rb") as f:
    channel_coords = pickle.load(f)


# ============ Params

marker_3Dplot = 'komp' # choose marker from result dict
state_3Dplot = 'W' # choose sleep stage from result dict
lobes_to_plot = ['par', 'front', 'occ', 'ins', 'temp']
value_to_plot = 'mean' # 'var'
cutoff_percent = 100  # % top value (eg 50 for top 50% values of marker)
show_all = True # If set to False, only plots channels above cutoff_percent
surface_plot = 'pial' # e.g., "pial", "inflated"
cortex_color = "darkgrey" 
cortex_alpha=0.3
vmin_scale = 0.0
vmax_scale = 0.5
size_ch_points=0.40
ch_colormap= 'turbo'
ch_below_thr_color='lightgrey'
hemi_3Dplot="lh" # 'lh' 'rh'
viewtype="lateral"

# --- Extract coordinates & values
coords_raw, values_raw, hemi_raw = [], [], []

for state in results:
    if state != state_3Dplot:
        continue
    for lobe in results[state]:
        if lobes_to_plot is not None and not any(lobe.startswith(l) for l in lobes_to_plot):
            continue  
        for region in results[state][lobe]:
            for rec in results[state][lobe][region].get(marker_3Dplot, {}):
                chan = extract_channel_id_from_recording(rec)
                if chan not in channel_coords:
                    continue
                val = results[state][lobe][region][marker_3Dplot][rec]
                if value_to_plot == 'mean':
                    val = np.nanmean(val) if isinstance(val, np.ndarray) else val
                elif value_to_plot == 'var':
                    val = np.nanvar(val) if isinstance(val, np.ndarray) else val
                else:
                    print('Invalid value to plot - not mean/var')
                if np.isnan(val):
                    continue
                x, y, z, hemi = channel_coords[chan]
                coords_raw.append((x, y, z))
                values_raw.append(val)
                hemi_raw.append(hemi)

print(f"Total number of valid coordinates extracted for state {state_3Dplot} : {len(coords_raw)}")

# --- Compute cutoff value (top X%) ---
values_arr = np.asarray(values_raw, dtype=float)
if values_arr.size == 0:
    raise RuntimeError(
        f"No valid values extracted for state={state_3Dplot}, marker={marker_3Dplot}. "
        "Check channel_coords mapping and results content."
    )
threshold = np.nanpercentile(values_arr, 100 - cutoff_percent)
print(f"Cutoff ({cutoff_percent}% top) = {threshold:.3f}")

# --- Load the surface used for plotting (pial/inflated/white, etc.) ---
subjects_dir = Path(subjects_dir)
surf_name = surface_plot 
lh_surf, _ = read_surface(str(subjects_dir / subject / "surf" / f"lh.{surf_name}"))
rh_surf, _ = read_surface(str(subjects_dir / subject / "surf" / f"rh.{surf_name}"))
tree_lh = cKDTree(lh_surf)
tree_rh = cKDTree(rh_surf)

# --- Assign to vertices ---
vertices_lh, colors_lh = [], []
vertices_rh, colors_rh = [], []

norm = Normalize(vmin=vmin_scale, vmax=vmax_scale)
try:
    cmap = matplotlib.colormaps[ch_colormap]
except KeyError:
    warnings.warn(f"Unknown colormap '{ch_colormap}', falling back to 'viridis'.")
    cmap = matplotlib.colormaps["viridis"]

for (x, y, z), val, hemi in zip(coords_raw, values_raw, hemi_raw):
    is_above_cutoff = val >= threshold
    if not show_all and not is_above_cutoff:
        continue
    
    color = cmap(norm(val))[:3] if is_above_cutoff else ch_below_thr_color  

    if hemi == "lh":
        dist, idx = tree_lh.query([x, y, z])
        vertices_lh.append((idx, color))
    elif hemi == "rh":
        dist, idx = tree_rh.query([x, y, z])
        vertices_rh.append((idx, color))


# --- Init Brain object ---
brain = Brain(
    subject="fsaverage",
    hemi=hemi_3Dplot, 
    surf=surface_plot,  
    background="white",
    cortex= cortex_color,
    alpha=cortex_alpha,
    subjects_dir=subjects_dir,
    size=800,
    show=True
)

# --- Add contacts ---
if brain._hemi in ("lh", "both"):
    for idx, color in vertices_lh:
        brain.add_foci(
            idx, coords_as_verts=True, hemi="lh",
            color=color, scale_factor=size_ch_points
        )

if brain._hemi in ("rh", "both"):
    for idx, color in vertices_rh:
        brain.add_foci(
            idx, coords_as_verts=True, hemi="rh",
            color=color, scale_factor=size_ch_points
        )

# --- Colorbar ---
fig, ax = plt.subplots(figsize=(6, 1))
cb = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax, orientation='horizontal')
if cutoff_percent == 100 :
    cb.set_label(f"{marker_3Dplot}", fontsize=14)
else:
    cb.set_label(f"Mean {marker_3Dplot} values (Top {cutoff_percent}%)", fontsize=14)
cb.ax.tick_params(labelsize=14)
plt.show()

# --- 3D Plot ---
brain.show_view(viewtype)
out_path = Path(f"/path/to/save/fig/{marker_3Dplot}_{viewtype}_{state_3Dplot}.png")
out_path.parent.mkdir(parents=True, exist_ok=True)
brain.save_image(str(out_path))
brain.show()





# %%

## ===== CHECK DATA for comparisons between sleep stages

channels_per_state = defaultdict(set)

for st in results:
    for lb in results[st]:
        for rg in results[st][lb]:
            region_dict = results[st][lb][rg]  # marker -> {rec: values}
            if not region_dict:
                continue
            first_marker_recs = next(iter(region_dict.values()), {})
            for rec in first_marker_recs.keys():
                chan = extract_channel_id_from_recording(rec)
                if chan:
                    channels_per_state[st].add(chan)

sleep_pairs = list(combinations(["W", "N2", "N3", "REM"], 2))
print("Number of shared channels bewteen stages:")
for s1, s2 in sleep_pairs:
    common = channels_per_state[s1] & channels_per_state[s2]
    print(f"→ {len(common):4d} chans present in {s1} and {s2}")


# %%

## === MannWhitney or Flinger-Killeen

marker = "pe_theta" # choose name
state1 = "N2"
state2 = "REM"
alpha = 0.05
lobes_to_plot = ['par', 'front', 'occ', 'ins', 'temp']
surface_plot = 'pial' #'inflated' # "pial"
test_name = 'mannwhitneyu' # 'mannwhitneyu' # 'fligner'
vmin_scale = -0.1 
vmax_scale = 0.1 
size_ch_points=0.5
ch_colormap= 'plasma' 
ch_non_sign_color='gold'
cortex_color= 'darkgrey'
cortex_alpha=0.3
hemi_3Dplot="lh" # 'lh' 'rh'
viewtype="lateral"
mult_test_method="fdr_bh" # 'bonferroni'

# === Collect vals per patient/channel
channel_vals = defaultdict(lambda: defaultdict(list))  # channel -> state -> list of values

for state in results:
    if state not in [state1, state2]:
        continue
    for lobe in results[state]:
        if lobes_to_plot is not None and not any(lobe.startswith(l) for l in lobes_to_plot):
            continue 
        for region in results[state][lobe]:
            for rec, vals in results[state][lobe][region].get(marker, {}).items():
                chan = extract_channel_id_from_recording(rec)
                if chan not in channel_coords:
                    continue
                vals = np.asarray(vals)
                vals = vals[~np.isnan(vals)]
                if len(vals) == 0:
                    continue
                channel_vals[chan][state].append(vals)

# === Test
all_tests = []  # (chan, mean_diff, p_value)
nb_excl_nan, nb_excl_size = 0, 0

for chan, states in channel_vals.items():
    if state1 in states and state2 in states:
        v1 = np.concatenate(states[state1])
        v2 = np.concatenate(states[state2])
        if len(v1) == 0 or len(v2) == 0:
            nb_excl_nan += 1
            continue
        if len(v1) < 5 or len(v2) < 5:
            nb_excl_size += 1
            print(f"!! Insufficient length for chan: {chan}")
            continue
        try:
            if test_name == "mannwhitneyu":
                stat, p = mannwhitneyu(v1, v2, alternative="two-sided")
                diff = np.nanmean(v1) - np.nanmean(v2)
            elif test_name == "fligner":
                stat, p = fligner(v1, v2)
                diff = np.nanvar(v1) - np.nanvar(v2)
            else:
                raise ValueError("!! test_name must be 'mannwhitneyu' or 'fligner'")

            all_tests.append((chan, diff, p))
        except ValueError:
            continue
print(f"\u2192 Exclusions for NaN : {nb_excl_nan}")
print(f"Number of tests : {len(all_tests)}")

if len(all_tests) == 0:
    raise RuntimeError(
        f"No tests to run: no channel had usable data in both {state1} and {state2} "
        f"for marker='{marker}'."
    )

# === Multiple tests correction
pvals = [t[2] for t in all_tests]
rejected, pvals_corr, _, _ = multipletests(pvals, alpha=alpha, method=mult_test_method)

significant_diffs = {
    chan: mean_diff for (chan, mean_diff, _), keep in zip(all_tests, rejected) if keep
}

print(f"Number of significant tests before correction: {(np.array(pvals) < alpha).sum()}")
print(f"Number of significant tests after correction: {len(significant_diffs)}")


# === Coordinates
coords_raw_sig, diffs_raw_sig, hemi_raw_sig = [], [], []
coords_raw_nonsig, hemi_raw_nonsig = [], []

for (chan, mean_diff, pval), keep in zip(all_tests, rejected):
    x, y, z, hemi = channel_coords[chan]
    if keep: 
        coords_raw_sig.append((x, y, z))
        diffs_raw_sig.append(mean_diff)
        hemi_raw_sig.append(hemi)
    else: 
        coords_raw_nonsig.append((x, y, z))
        hemi_raw_nonsig.append(hemi)

# === Project on cortical surface
lh_surf, _ = read_surface(f"{subjects_dir}/fsaverage/surf/lh.{surface_plot}")
rh_surf, _ = read_surface(f"{subjects_dir}/fsaverage/surf/rh.{surface_plot}")
tree_lh = cKDTree(lh_surf)
tree_rh = cKDTree(rh_surf)
vertices_lh_sig, values_lh_sig = [], []
vertices_rh_sig, values_rh_sig = [], []

for (x, y, z), val, hemi in zip(coords_raw_sig, diffs_raw_sig, hemi_raw_sig):
    if hemi == "lh":
        dist, idx = tree_lh.query([x, y, z])
        vertices_lh_sig.append(idx)
        values_lh_sig.append(val)
    elif hemi == "rh":
        dist, idx = tree_rh.query([x, y, z])
        vertices_rh_sig.append(idx)
        values_rh_sig.append(val)

# Non significant chans
vertices_lh_nonsig, vertices_rh_nonsig = [], []

for (x, y, z), hemi in zip(coords_raw_nonsig, hemi_raw_nonsig):
    if hemi == "lh":
        dist, idx = tree_lh.query([x, y, z])
        vertices_lh_nonsig.append(idx)
    elif hemi == "rh":
        dist, idx = tree_rh.query([x, y, z])
        vertices_rh_nonsig.append(idx)

# === Plot
brain = Brain(
    subject="fsaverage",
    hemi=hemi_3Dplot,
    surf=surface_plot,
    background="white",
    cortex=cortex_color,
    alpha=cortex_alpha,
    subjects_dir=subjects_dir,
    size=800,
    show=True
)
norm = colors.TwoSlopeNorm(vmin=vmin_scale, vcenter=0, vmax=vmax_scale)
cmap = plt.get_cmap(ch_colormap) if isinstance(ch_colormap, str) else ch_colormap

# Significant
if brain._hemi in ("lh", "both"):
    for idx, val in zip(vertices_lh_sig, values_lh_sig):
        brain.add_foci(
            idx, coords_as_verts=True, hemi="lh",
            color=cmap(norm(val))[:3], scale_factor=size_ch_points
        )

if brain._hemi in ("rh", "both"):
    for idx, val in zip(vertices_rh_sig, values_rh_sig):
        brain.add_foci(
            idx, coords_as_verts=True, hemi="rh",
            color=cmap(norm(val))[:3], scale_factor=size_ch_points
        )

# Non significant
if brain._hemi in ("lh", "both"):
    for idx in vertices_lh_nonsig:
        brain.add_foci(
            idx, coords_as_verts=True, hemi="lh",
            color=ch_non_sign_color, scale_factor=size_ch_points
        )

if brain._hemi in ("rh", "both"):
    for idx in vertices_rh_nonsig:
        brain.add_foci(
            idx, coords_as_verts=True, hemi="rh",
            color=ch_non_sign_color, scale_factor=size_ch_points
        )

# === Colorbar
fig, ax = plt.subplots(figsize=(6, 1))
cb = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax, orientation="horizontal")
cb.set_label(f"Δ in Permutation Entropy (theta)", fontsize=14)
plt.show()

# === Plot
brain.show_view(viewtype)
out_path = Path(f"/path/to/save/fig/{marker_3Dplot}_{viewtype}_{state1}_{state2}_{test_name}.png")
out_path.parent.mkdir(parents=True, exist_ok=True)
brain.save_image(str(out_path))
brain.show()

# %%
