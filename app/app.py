# =========================
# App - Visualizing markers of EEG compexity
# Lina Jeantin, Nov. 2025
# lina.jeantin@proton.me
# =========================

# app.py
import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import mne

# =========================
# Config
# =========================
st.set_page_config(page_title="Atlas of iEEG Complexity", page_icon="🧠", layout="centered")


st.markdown("""
<style>
.main .block-container { max-width: 980px; }

.metric-title { text-align:center; }
.metric-title .region { font-size: 1.05rem; font-weight: 700; }
.metric-title .lobe   { font-size: 1rem;  font-style: italic; opacity: .85; margin-top: .15rem; }
.metric-subcap        { font-size: 1rem;  opacity: .9;  margin-top: .35rem; }

.metric-card { display:flex; flex-direction:column; align-items:center; margin: .6rem auto 1.1rem auto; }
.metric-card .metric-label { font-size: 1.05rem; opacity: .85; margin-bottom: .2rem; }
.metric-card .metric-value { font-size: 2.15rem; font-weight: 700; line-height: 1.1; }
.metric-card .metric-std   { font-size: 1rem;  opacity: .85; margin-top: .2rem; }

.metric-meta { text-align:center; opacity:.85; font-size: 1rem; }
[data-testid="stCaptionContainer"] p,
[data-testid="stCaptionContainer"] {
  font-size: 1rem;           /* default ~0.8rem */
  line-height: 1.35;
}
</style>
""", unsafe_allow_html=True)

# =========================
# Data & keys
# =========================

METRICS_PATH = "metrics_by_parcel_sample.csv"  

# == Mapping CSV parcel_key -> approximate (aparc.a2009s) (only for visual rendering)
REGION_TO_ANNOT = {
    "front_ant_cing": ["g_and_s_cingul-ant", "g_subcallosal"],
    "front_mid_cing": ["g_and_s_cingul-mid-ant", "g_and_s_cingul-mid-post"], 
    "front_med": ["g_front_sup"],                                                                       ###
    "front_med_prec": ["g_and_s_paracentral"],
    "front_med_sup": ["g_front_sup", "s_front_sup", "g_and_s_paracentral"],                             ###
    "front_sup_mot": ["g_and_s_paracentral"],                                    
    "front_rect_orb": ["g_rectus","g_orbital","s_orbital_lateral","s_orbital_med-olfact", "s_orbital-h_shapped","s_suborbital"],
    "front_op_inf": ["g_front_inf-opercular", "s_front_inf"],
    "front_orb_inf": ["g_front_inf-orbital"],
    "front_triang_inf": ["g_front_inf-triangul"],
    "front_opercul": [
        "g_front_inf-opercular",
        "lat_fissure-ant-horizont","lat_fissure-ant-vertical"
    ],
    "front_sup_pole": ["g_front_sup","s_front_sup","g_and_s_transv_frontopol","g_and_s_frontomargin"],
    "front_mid_front": ["g_front_middle", "s_front_middle"],
    "front_precentr": ["g_precentral","s_precentral-inf-part","s_precentral-sup-part","s_central"],
    "front_centr_op": ["g_and_s_subcentral"],
    "ins_ant": ["G_insular_short", "S_circular_insula_ant", "S_circular_insula_sup"],
    "ins_post": ["G_Ins_lg_and_S_cent_ins", "G_Ins_lg_and_S_cent_ins", "S_circular_insula_inf"],
    "occ_calc": ["s_calcarine"],
    "occ_cuneus": ["g_cuneus"],
    "occ_inf_pole": ["g_and_s_occipital_inf","pole_occipital"],
    "occ_ling_fusif": ["g_oc-temp_med-lingual", "s_collat_transv_post"],
    "occ_sup_middle": ["g_occipital_sup","g_occipital_middle","s_oc_middle_and_lunatus","S_oc_sup_and_transversal","s_occipital_ant"], 
    "par_ang": ["g_pariet_inf-angular","s_intrapariet_and_p_trans"], ### 
    "par_postcentr": ["g_postcentral","s_postcentral"],
    "par_post_cing": ["g_cingul-post-dorsal","g_cingul-post-ventral"],
    "par_precuneus": ["g_precuneus","s_subparietal","s_parieto_occipital"],
    "par_opercul": ["lat_fis-post","g_pariet_inf-supramar"],                           ###                   
    "par_supramarginal": ["g_pariet_inf-supramar","s_intrapariet_and_p_trans"], ### 
    "par_sup_lob": ["g_parietal_sup"],
    "temp_fus_parahip": ["g_oc-temp_lat-fusif","G_oc-temp_med-Parahip","s_collat_transv_ant"],
    "temp_inf_temp": ["g_temporal_inf","s_temporal_inf"],
    "temp_mtg": ["g_temporal_middle"],
    "temp_pole_plan_pol": ["pole_temporal","g_temp_sup-plan_polar"],
    "temp_plan_temp": ["g_temp_sup-plan_tempo"],
    "temp_sup": ["g_temporal_sup-lateral","s_temporal_sup"], ###
    "temp_transv": ["g_temp_sup-g_t_transv", "s_temporal_transverse"],
    "temp_hip": ["G_oc-temp_med-Parahip"],
    #"temp_amygd" : ["G_oc-temp_med-Parahip"]
}

# == Metrics
METRIC_MAP = {
    "Spectral entropy": ("spectral_entropy", "spectral_entropy_std"),
    "Kolmogorov complexity": ("komp", "komp_std"),
    "Alpha/Delta ratio": ("adr", "adr_std"),
    "Total power": ("psd_tot", "psd_tot_std"),

    # PSD raw
    "PSD delta (raw)": ("psd_raw_delta", "psd_raw_delta_std"),
    "PSD theta (raw)": ("psd_raw_theta", "psd_raw_theta_std"),
    "PSD alpha (raw)": ("psd_raw_alpha", "psd_raw_alpha_std"),
    "PSD beta (raw)": ("psd_raw_beta", "psd_raw_beta_std"),
    "PSD gamma (raw)": ("psd_raw_gamma", "psd_raw_gamma_std"),

    # PSD norm
    "PSD delta (norm)": ("psd_norm_delta", "psd_norm_delta_std"),
    "PSD theta (norm)": ("psd_norm_theta", "psd_norm_theta_std"),
    "PSD alpha (norm)": ("psd_norm_alpha", "psd_norm_alpha_std"),
    "PSD beta (norm)": ("psd_norm_beta", "psd_norm_beta_std"),
    "PSD gamma (norm)": ("psd_norm_gamma", "psd_norm_gamma_std"),

    # Permutation entropy
    "Permutation entropy (delta)": ("pe_delta", "pe_delta_std"),
    "Permutation entropy (theta)": ("pe_theta", "pe_theta_std"),
    "Permutation entropy (alpha)": ("pe_alpha", "pe_alpha_std"),
    "Permutation entropy (beta)": ("pe_beta", "pe_beta_std"),
    "Permutation entropy (gamma)": ("pe_gamma", "pe_gamma_std"),
}

# =========================
# Load
# =========================
@st.cache_data
def load_metrics_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "epoch_length" in df.columns:
        df["epoch_length"] = pd.to_numeric(df["epoch_length"], errors="coerce").astype(float)
    for col_num in ["n_recordings", "n_subj", "n_patients"]:
        if col_num in df.columns:
            df[col_num] = pd.to_numeric(df[col_num], errors="coerce")
    for c in ["sleep_stage","cortex","parcel_name","parcel_key","lobe"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    return df

df = load_metrics_csv(METRICS_PATH)

ordered_stages = ["W","N2","N3","REM","propofol"]
stages_all = sorted(df["sleep_stage"].dropna().unique().tolist(), key=lambda x: ordered_stages.index(x) if x in ordered_stages else 999)
epochs_all = sorted(df["epoch_length"].dropna().unique().tolist()) if "epoch_length" in df.columns else []
epileptic_all = sorted(df["cortex"].dropna().unique().tolist()) if "cortex" in df.columns else ["healthy","epileptic"]
lobes_all = sorted(df["lobe"].dropna().unique().tolist())


# =========================
# Helpers 
# =========================
@st.cache_resource
def get_fs_paths():
    fsaverage_dir = mne.datasets.fetch_fsaverage(verbose=False)   # .../MNE-fsaverage-data/fsaverage
    subjects_dir = os.path.dirname(fsaverage_dir)                 # .../MNE-fsaverage-data
    return fsaverage_dir, subjects_dir

@st.cache_resource
def load_surfaces(fsaverage_dir: str):
    lh_verts, lh_faces = mne.read_surface(os.path.join(fsaverage_dir,"surf","lh.pial"))
    rh_verts, rh_faces = mne.read_surface(os.path.join(fsaverage_dir,"surf","rh.pial"))
    return (lh_verts, lh_faces), (rh_verts, rh_faces)

def build_mesh(verts, faces, opacity=0.12, color="#888"):
    x, y, z = verts.T
    i, j, k = faces.T
    return go.Mesh3d(
        x=x, y=y, z=z, i=i, j=j, k=k,
        color=color, opacity=opacity, showscale=False,
        lighting=dict(ambient=0.5, diffuse=0.6, specular=0.2),
    )

def submesh_from_vertices(verts, faces, vert_mask, color="#E69F00", name="region"):
    fmask = vert_mask[faces].all(axis=1)
    if not fmask.any():
        return None
    f_sub = faces[fmask]
    uniq = np.unique(f_sub)
    remap = {old: new for new, old in enumerate(uniq)}
    f_re = np.vectorize(remap.get)(f_sub)
    v_sub = verts[uniq]
    x, y, z = v_sub.T
    i, j, k = f_re.T
    return go.Mesh3d(
        x=x, y=y, z=z, i=i, j=j, k=k,
        color=color, opacity=0.85, name=name, showscale=False,
        lighting=dict(ambient=0.5, diffuse=0.7, specular=0.2),
    )

def get_region_vertices(subjects_dir: str, parc_name: str, parcel_key: str):
    frags = [f.lower() for f in REGION_TO_ANNOT.get(parcel_key, [])]
    if not frags:
        return {}
    out = {}
    for hemi in ("lh","rh"):
        labels = mne.read_labels_from_annot(
            subject="fsaverage", parc=parc_name, hemi=hemi, subjects_dir=subjects_dir, verbose=False
        )
        verts = []
        for lab in labels:
            nm = lab.name.lower()
            if any(frag in nm for frag in frags):
                verts.append(lab.vertices)
        if verts:
            out[hemi] = np.unique(np.concatenate(verts))
    return out

# =========================
# Selectors
# =========================
st.title("🧠 Indexing iEEG Complexity")
st.caption("Select a cortical region and a sleep stage to retrieve the average metric of iEEG spectrum or complexity.")

c1, c2 = st.columns(2)

def _on_lobe_change():
    regions = sorted(df.loc[df["lobe"] == st.session_state.lobe, "parcel_name"].dropna().unique().tolist())
    if not regions:
        st.session_state.region = None
    elif st.session_state.get("region") not in regions:
        st.session_state.region = regions[0]

if "lobe" not in st.session_state:
    st.session_state.lobe = lobes_all[0] if lobes_all else None
if "region" not in st.session_state:
    init_regions = sorted(df.loc[df["lobe"] == st.session_state.lobe, "parcel_name"].dropna().unique().tolist()) if st.session_state.lobe else []
    st.session_state.region = init_regions[0] if init_regions else None

lobe = c1.selectbox("Lobe", options=lobes_all,
    index=(lobes_all.index(st.session_state.lobe) if st.session_state.lobe in lobes_all else 0) if lobes_all else None,
    key="lobe", on_change=_on_lobe_change)

regions_for_lobe = sorted(df.loc[df["lobe"] == lobe, "parcel_name"].dropna().unique().tolist()) if lobe else []
region = c2.selectbox("Cortical region", options=regions_for_lobe,
    index=(regions_for_lobe.index(st.session_state.region) if st.session_state.region in regions_for_lobe else 0) if regions_for_lobe else None,
    key="region")

with st.form("query_rest"):
    c3, c4 = st.columns(2)
    stage = c3.selectbox("Sleep stage", options=stages_all, index=stages_all.index("W") if "W" in stages_all else 0)
    cortex = c4.selectbox("Cortex", options=epileptic_all, index=epileptic_all.index("healthy") if "healthy" in epileptic_all else 0)

    wanted_epochs = [1,2,5,10]
    epochs_all_num = sorted({float(e) for e in epochs_all}) if epochs_all else []
    epoch_options = [float(e) for e in wanted_epochs if float(e) in epochs_all_num]
    options = epoch_options if epoch_options else epochs_all_num

    if not options:
        epoch = None
        st.info("No epoch length in source data file file.")
    elif len(options) == 1:
        epoch = st.selectbox("Epoch length (s)", options=options, index=0)
    else:
        epoch = st.select_slider("Epoch length (s)", options=options, value=options[0])

    metric_label = st.selectbox("Metric", options=list(METRIC_MAP.keys()), index=0)
    submitted = st.form_submit_button("Query")

# =========================
# Rendering
# =========================
if submitted:
    if not (lobe and region and stage and cortex):
        st.warning("Please select a lobe, cortical region, sleep stage and cortex nature.")
    else:
        mask = (
            (df["lobe"] == lobe) &
            (df["parcel_name"] == region) &
            (df["sleep_stage"] == stage) &
            (df["cortex"] == cortex)
        )
        if epoch is not None and "epoch_length" in df.columns:
            mask &= (df["epoch_length"] == float(epoch))

        hit = df.loc[mask]

        if hit.empty:
            st.warning("No statistics available for this combination.")
            with st.expander("See available combinations for this region"):
                cols = ["sleep_stage","epoch_length","cortex"]
                cols = [c for c in cols if c in df.columns]
                sub = df.loc[(df["lobe"] == lobe) & (df["parcel_name"] == region), cols].drop_duplicates()
                st.dataframe(sub.reset_index(drop=True))
        else:
            row = hit.iloc[0]
            col_mean, col_std = METRIC_MAP[metric_label]
            val_mean = row.get(col_mean, np.nan)
            val_std  = row.get(col_std, np.nan)

            # ----- Metrics -----
            st.subheader("📊 Metric")
            st.markdown(
                f"""
                <div class="metric-title">
                    <div class="region">{region}</div>
                    <div class="lobe">{lobe}</div>
                    <div class="metric-subcap">
                        Stage: <strong>{stage}</strong> • Cortex: <strong>{cortex}</strong>{f" • Epoch: <strong>{epoch:.0f}s</strong>" if epoch is not None else ""}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">{metric_label}</div>
                    <div class="metric-value">{'NA' if pd.isna(val_mean) else f'{val_mean:.3f}'}</div>
                    <div class="metric-std">{'' if pd.isna(val_std) else f'± {val_std:.3f} (SD)'}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            n_rec, n_sub = None, None
            if "n_recordings" in row and pd.notna(row["n_recordings"]):
                try: n_rec = int(row["n_recordings"])
                except Exception: pass
            if "n_subj" in row and pd.notna(row["n_subj"]):
                try: n_sub = int(row["n_subj"])
                except Exception: pass

            if n_rec is not None and n_sub is not None:
                meta_line = f"Metric computed with <strong>{n_rec}</strong> recording(s) in <strong>{n_sub}</strong> subject(s)"
            elif n_rec is not None:
                meta_line = f"Metric computed with <strong>{n_rec}</strong> recording(s)"
            elif n_sub is not None:
                meta_line = f"Metric computed in <strong>{n_sub}</strong> subject(s)"
            else:
                meta_line = ""

            if meta_line:
                st.markdown(f"<div class='metric-meta'>{meta_line}</div>", unsafe_allow_html=True)

            # ----- 3D 
            try:
                fsaverage_dir, subjects_dir = get_fs_paths()
                (lh_verts, lh_faces), (rh_verts, rh_faces) = load_surfaces(fsaverage_dir)

                fig = go.Figure()
                fig.add_trace(go.Mesh3d(
                    x=lh_verts[:,0], y=lh_verts[:,1], z=lh_verts[:,2],
                    i=lh_faces[:,0], j=lh_faces[:,1], k=lh_faces[:,2],
                    color="#888", opacity=0.12, showscale=False,
                    lighting=dict(ambient=0.5, diffuse=0.6, specular=0.2),
                ))
                fig.add_trace(go.Mesh3d(
                    x=rh_verts[:,0], y=rh_verts[:,1], z=rh_verts[:,2],
                    i=rh_faces[:,0], j=rh_faces[:,1], k=rh_faces[:,2],
                    color="#888", opacity=0.12, showscale=False,
                    lighting=dict(ambient=0.5, diffuse=0.6, specular=0.2),
                ))

                parc_name = "aparc.a2009s"
                key_candidates = df.loc[
                    (df["lobe"] == lobe) & (df["parcel_name"] == region), "parcel_key"
                ].dropna().unique().tolist()
                region_key = key_candidates[0] if key_candidates else None

                if region_key:
                    region_verts = get_region_vertices(subjects_dir, parc_name, region_key)
                    color_region = "#E69F00"
                    if "lh" in region_verts and region_verts["lh"].size:
                        lh_mask = np.zeros(lh_verts.shape[0], dtype=bool); lh_mask[region_verts["lh"]] = True
                        t = submesh_from_vertices(lh_verts, lh_faces, lh_mask, color=color_region, name="LH region")
                        if t: fig.add_trace(t)
                    if "rh" in region_verts and region_verts["rh"].size:
                        rh_mask = np.zeros(rh_verts.shape[0], dtype=bool); rh_mask[region_verts["rh"]] = True
                        t = submesh_from_vertices(rh_verts, rh_faces, rh_mask, color=color_region, name="RH region")
                        if t: fig.add_trace(t)
                    if not region_verts:
                        st.caption("⚠️ No label found for this region in the aparc.a2009s parcellation for 3D rendering.")
                else:
                    st.caption("⚠️ `parcel_key` not found in source data file.")

                fig.update_scenes(aspectmode="data")
                fig.update_layout(
                    height=620, margin=dict(l=0, r=0, t=0, b=0), showlegend=False,
                    paper_bgcolor="#000", plot_bgcolor="#000",
                    scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False))
                )
                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"3D view unavailable : {e}")

# =========================
# About
# =========================
with st.expander("About / Reproducibility"):
    st.markdown("""
**Resources**:
- **Cortical parcellations used for quantitative EEG analysis**: MICCAI 2012 Neuromorphometrics. https://www.neuromorphometrics.com/2012_MICCAI_Challenge_Data.html
- **Source data files for all metrics, cortical regions and sleep stages**: https://github.com/linajeantin/iEEGcomplexity/tree/main
- **3D rendering** of the selected cortical region is based on fsaverage templates and the aparc.a2009s parcellation (Destrieux et al., 2010 https://surfer.nmr.mgh.harvard.edu/fswiki/CorticalParcellation)

**To use this work please cite**: <Temporary: Lina Jeantin, Lionel Naccache, Paris Brain Institute, Pitié Salpétrière University Hospital, Paris, France. lina.jeantin@proton.me>

This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version. **Please note that this program is provided with no warranty of any kind.**
""")

