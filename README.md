# iEEGcomplexity

**Indexing intracranial EEG (iEEG) signal complexity**

This repository contains code and resources to compute, aggregate, visualize, and model intracranial EEG signal complexity markers, with applications to the **MNI-Open iEEG Atlas** dataset.

## Repository structure

### `app/`
Streamlit application for interactive data exploration.

- `app.py` — Streamlit app for data visualization  
- `metrics_by_parcel.csv` — Aggregated complexity metrics (by parcel) used by the app

### `results/`
Scripts used to compute markers and reproduce analyses/figures.

#### `results/compute_results/`
Core computation and data extraction utilities.

- `methods_complexity.py` — Methods to compute EEG complexity markers  
  *(spectrum measures, spectral entropy, permutation entropy, Kolmogorov complexity)*
- `get_complexity_results.py` — Computes iEEG complexity markers on the **MNI-Open iEEG Atlas** dataset  
  https://mni-open-ieegatlas.research.mcgill.ca/
- `get_recordings_info.py` — Utilities for metadata extraction:
  - patient IDs (parsed from recording names)
  - channel coordinates in MNI space
- `extract_results_df.py` — Converts the nested `results` dictionary into dataframes for plotting, statistics, and ML

#### `results/plots_stats_models/`
Analysis and visualization scripts.

- `3D_plots.py` — 3D brain visualizations using `mne.viz.Brain`
- `linear_mixed_models_sleep.py` — Linear mixed-effects models assessing marker variations across sleep stages  
  *(patient ID as random effect; accounts for varying numbers of recordings per patient)*
- `plot_means_sleepstages.py` — Plots of mean marker values across sleep stages (with summary statistics)
- `sleep_stage_classifier.py` — Random Forest classifier predicting sleep stages from iEEG complexity markers

### `source_data/`
Source data files contaning values of EEG complexity markers with different epoch lengths, for each recording.

---

## Notes

- Several methods in `methods_complexity.py` were adapted from **NICE tools v0.1.dev1** (GNU AGPL v3):  
  https://nice-tools.github.io/nice/  
  Please cite:  
  Engemann D.A.*, Raimondo F.*, King J.R., et al. *Robust EEG-based cross-site and cross-protocol classification of states of consciousness.* **Brain** 141(11), 3160–3178. doi:10.1093/brain/awy251

      
## About
Author:
Lina Jeantin, Paris Brain Institute, Pitié Salpétrière University Hospital, Paris, France. lina.jeantin@proton.me

