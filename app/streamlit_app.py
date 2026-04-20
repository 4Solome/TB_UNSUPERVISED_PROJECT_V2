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

st.title("TB Risk Profiling System")
st.caption(
    "Unsupervised representation learning and generative modeling for latent "
    "tuberculosis risk sequencing and phenotype discovery."
)

# ============================================================
# CLUSTER DEFINITIONS
# ============================================================
CLUSTER_INFO = {
    3: {
        "name": "High-Risk / Active TB Phenotype",
        "stage": "Earliest / Most severe risk stage",
        "risk": "High Risk",
        "summary": "Strong cough, chest pain, sputum and other symptomatic TB signals.",
        "key_features": ["cough", "chest_pain", "sputum", "fever", "weight_loss"],
    },
    1: {
        "name": "Symptomatic TB Phenotype",
        "stage": "Early risk stage",
        "risk": "High Risk",
        "summary": "Moderate cough and chest pain with clinically relevant symptom burden.",
        "key_features": ["cough", "chest_pain", "fever", "weight_loss"],
    },
    4: {
        "name": "Transitional TB Risk Phenotype",
        "stage": "Intermediate / transition stage",
        "risk": "Moderate Risk",
        "summary": "Milder but emerging symptom profile.",
        "key_features": ["chest_pain", "cough", "weight_loss"],
    },
    2: {
        "name": "Low-Symptom Phenotype",
        "stage": "Later / stable stage",
        "risk": "Low Risk",
        "summary": "Very low symptom burden.",
        "key_features": ["minimal symptoms"],
    },
    0: {
        "name": "Very Low-Risk / Stable Phenotype",
        "stage": "Most stable stage",
        "risk": "Very Low Risk",
        "summary": "Minimal symptoms and stability.",
        "key_features": ["stable profile"],
    },
}

CLUSTER_ORDER = [3, 1, 4, 2, 0]

MAX_ROWS = 30000
BATCH_SIZE = 1024
LATENT_DIM = 16

# ============================================================
# LOADERS
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
OOD_PERCENTILE = int(OOD_INFO.get("percentile", 95))

# ============================================================
# SYNTHETIC DECODER (FIXED ✅)
# ============================================================
def decode_synthetic_from_transformed(syn_df: pd.DataFrame):
    """
    Properly decode synthetic samples from transformed space into
    human-readable clinical-like values.
    """

    decoded = pd.DataFrame(index=syn_df.index)

    def rescale(series, max_val, min_val=0):
        return (series.clip(0, 1) * (max_val - min_val) + min_val).round().astype(int)

    # Explicit continuous schema (CRITICAL)
    CONT_SCHEMA = {
        "age_census": (100, 0),
        "cough_d": (30, 0),
        "fever_d": (30, 0),
        "wloss_d": (365, 0),
        "sputum_d": (30, 0),
        "tbhist_y": (35, 1990),
        "tbtreat_w": (52, 0),
    }

    # Continuous
    for col, (mx, mn) in CONT_SCHEMA.items():
        tcol = f"cont__{col}"
        if tcol in syn_df.columns:
            decoded[col] = rescale(syn_df[tcol], mx, mn)

    # Binary
    for col in BINARY_COLS:
        tcol = f"bin__{col}"
        if tcol in syn_df.columns:
            decoded[col] = (syn_df[tcol] >= 0.5).astype(int)

    # Categorical
    for col in CATEGORICAL_COLS:
        prefix = f"cat__{col}_"
        matches = [c for c in syn_df.columns if c.startswith(prefix)]
        if matches:
            vals = (
                syn_df[matches]
                .idxmax(axis=1)
                .str.replace(prefix, "", regex=False)
                .str.replace(".0", "", regex=False)
            )
            decoded[col] = vals

    return decoded

# ============================================================
# SYNTHETIC DATA GENERATION
# ============================================================
st.divider()
st.header("Synthetic Patient Generation")

num_samples = st.slider("Number of synthetic patients", 10, 200, 50)

if st.button("Generate Synthetic Patients"):
    z = torch.randn(num_samples, LATENT_DIM)

    with torch.no_grad():
        syn_array = model.decode(z).cpu().numpy()

    syn_df = pd.DataFrame(syn_array, columns=feature_names)
    decoded = decode_synthetic_from_transformed(syn_df)

    st.success(f"Generated {num_samples} readable synthetic patient records.")
    st.dataframe(decoded.head(10), use_container_width=True)

    st.download_button(
        "Download Synthetic Dataset",
        decoded.to_csv(index=False),
        file_name="synthetic_tb_patients.csv",
        mime="text/csv",
    )

st.caption(
    "Synthetic records are statistically plausible but not real patients. "
    "They must not be used directly for clinical decision-making."
)
