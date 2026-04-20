import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import torch

from utils import (
    CONTINUOUS_COLS,
    BINARY_COLS,
    CATEGORICAL_COLS,
    prepare_input_dataframe,
    load_ttvae,
    load_feature_names,
    load_cluster_model,
    load_pseudotime_bounds,
    load_ood_threshold,
    compute_latent,
    compute_pseudotime,
    assign_cluster,
    batched_reconstruction_error,
)

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="TB Risk Profiling System",
    page_icon="🫁",
    layout="wide",
)

# ============================================================
# GLOBAL DARK UI STYLES
# ============================================================
st.markdown(
    """
    <style>
    body {
        background-color: #0b1020;
        color: #e5e7eb;
    }

    .block-container {
        padding-top: 2.5rem;
    }

    .card {
        background: #121a33;
        border-radius: 14px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0px 10px 30px rgba(0,0,0,0.4);
    }

    .hero {
        background: linear-gradient(135deg, #0f172a, #1e3a8a);
        padding: 2.5rem;
        border-radius: 18px;
        margin-bottom: 2rem;
    }

    .hero h1 {
        color: white;
        font-size: 2.8rem;
    }

    .hero p {
        color: #c7d2fe;
        max-width: 900px;
        font-size: 1.05rem;
    }

    .pill {
        display: inline-block;
        background: #1e293b;
        color: #93c5fd;
        padding: 0.4rem 0.75rem;
        border-radius: 999px;
        font-size: 0.75rem;
        margin-right: 0.5rem;
    }

    .section-title {
        font-size: 1.6rem;
        margin-bottom: 0.3rem;
    }

    .muted {
        color: #9ca3af;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# HERO SECTION
# ============================================================
st.markdown(
    """
    <div class="hero">
        <span class="pill">AI‑POWERED</span>
        <span class="pill">PRIVACY‑PRESERVING</span>
        <span class="pill">RESEARCH‑GRADE</span>
        <h1>🫁 TB Risk Profiling System</h1>
        <p>
        Advanced unsupervised representation learning and generative modeling
        for tuberculosis risk staging, latent progression sequencing, and
        phenotype discovery.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# LOAD ARTIFACTS
# ============================================================
@st.cache_resource
def load_preprocessor():
    try:
        return joblib.load("models/preprocessor.joblib")
    except Exception:
        return joblib.load("models/preprocessor.pkl")

@st.cache_resource
def load_all_artifacts():
    feature_names = load_feature_names()
    model = load_ttvae(input_dim=len(feature_names))
    kmeans = load_cluster_model()
    preprocessor = load_preprocessor()
    pt_bounds = load_pseudotime_bounds()
    ood_info = load_ood_threshold()
    return feature_names, model, kmeans, preprocessor, pt_bounds, ood_info

feature_names, model, kmeans, preprocessor, PT_BOUNDS, OOD_INFO = load_all_artifacts()
OOD_THRESHOLD = float(OOD_INFO.get("threshold", 0.0))

# ============================================================
# UPLOAD CARD
# ============================================================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("<div class='section-title'>📂 Upload Patient Cohort</div>", unsafe_allow_html=True)
st.markdown("<p class='muted'>Upload a CSV containing patient‑level TB records.</p>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("", type=["csv"])
analyze = st.button("🚀 Analyze Cohort", type="primary")

st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# COHORT ANALYSIS
# ============================================================
results = None

if uploaded_file and analyze:
    df_raw = pd.read_csv(uploaded_file)
    X = preprocessor.transform(prepare_input_dataframe(df_raw)).astype(np.float32)

    latents = compute_latent(model, X)
    pseudotime_norm = compute_pseudotime(latents, bounds=PT_BOUNDS)
    clusters = assign_cluster(kmeans, latents)
    rec_error = batched_reconstruction_error(model, X)
    ood_flags = rec_error > OOD_THRESHOLD

    results = pd.DataFrame({
        "Cluster": clusters,
        "Pseudotime": pseudotime_norm,
        "Reconstruction Error": rec_error,
    })

    st.success("✅ Cohort processed successfully.")

# ============================================================
# RESULTS SUMMARY (DASHBOARD STYLE)
# ============================================================
if results is not None:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)

    c1.metric("Total Patients", len(results))
    c2.metric("Avg Pseudotime", f"{results['Pseudotime'].mean():.2f}")
    c3.metric("OOD Warnings", int(ood_flags.sum()))

    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# SYNTHETIC GENERATION CARD
# ============================================================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("<div class='section-title'>🧬 Synthetic Patient Generation</div>", unsafe_allow_html=True)
st.markdown(
    "<p class='muted'>Generate statistically plausible TB patient profiles from the learned latent space.</p>",
    unsafe_allow_html=True,
)

num_samples = st.slider("Number of synthetic patients", 10, 200, 50)

if st.button("✨ Generate Synthetic Patients"):
    z = torch.randn(num_samples, 16)
    with torch.no_grad():
        syn = model.decode(z).cpu().numpy()

    syn_df = pd.DataFrame(syn, columns=feature_names)
    st.dataframe(syn_df.head(10), use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# FOOTER
# ============================================================
st.caption(
    "This system is intended for research and analytical use only. "
    "Synthetic data is not real patient data."
)
