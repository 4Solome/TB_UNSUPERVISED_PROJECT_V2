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
# INTERPRETABLE CLUSTER DEFINITIONS
# Ordered by risk progression from your saved cluster outputs:
# 3 -> 1 -> 4 -> 2 -> 0
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
        "summary": "Milder but emerging symptom profile, suggesting transition along risk progression.",
        "key_features": ["chest_pain", "cough", "weight_loss"],
    },
    2: {
        "name": "Low-Symptom Phenotype",
        "stage": "Later / relatively stable stage",
        "risk": "Low Risk",
        "summary": "Very low symptom burden and weak clinical activity.",
        "key_features": ["minimal symptoms", "stable radiology"],
    },
    0: {
        "name": "Very Low-Risk / Stable Phenotype",
        "stage": "Latest / most stable stage",
        "risk": "Very Low Risk",
        "summary": "Minimal symptoms and stable overall profile.",
        "key_features": ["minimal symptoms", "stable profile"],
    },
}

CLUSTER_ORDER = [3, 1, 4, 2, 0]

MAX_ROWS = 30000
BATCH_SIZE = 1024
LATENT_DIM = 16

ALL_COLS = CONTINUOUS_COLS + BINARY_COLS + CATEGORICAL_COLS


# ============================================================
# CACHED LOADERS
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

OOD_THRESHOLD = float(
    OOD_INFO.get("threshold", OOD_INFO.get("ood_threshold", 0.0))
)
OOD_PERCENTILE = int(OOD_INFO.get("percentile", 95))


# ============================================================
# HELPERS
# ============================================================
def transform_uploaded_data(df_raw: pd.DataFrame):
    df_clean = prepare_input_dataframe(df_raw)
    X = preprocessor.transform(df_clean)

    X_df = pd.DataFrame(X, columns=preprocessor.get_feature_names_out())
    X_df = X_df.reindex(columns=feature_names, fill_value=0.0)

    return df_clean, X_df.values.astype(np.float32)


def progression_position_label(pt_norm: float) -> str:
    if pt_norm < 0.20:
        return "Earliest / Highest-Risk Position"
    if pt_norm < 0.45:
        return "Early Progression Position"
    if pt_norm < 0.70:
        return "Intermediate Progression Position"
    if pt_norm < 0.90:
        return "Late Progression Position"
    return "Most Stable / Lowest-Risk Position"


def risk_bucket_from_cluster(cluster_id: int) -> str:
    return CLUSTER_INFO.get(cluster_id, {}).get("risk", "Unknown")


def reliability_label(flag: bool) -> str:
    return "⚠️ OOD Warning" if flag else "✅ In Distribution"


def build_patient_results(latents, pseudotime_norm, clusters, rec_error, ood_flags):
    rows = []

    for i in range(len(clusters)):
        cid = int(clusters[i])
        info = CLUSTER_INFO.get(cid, {})

        rows.append(
            {
                "Cluster": cid,
                "Phenotype": info.get("name", f"Cluster {cid}"),
                "Risk Category": risk_bucket_from_cluster(cid),
                "Progression Position": progression_position_label(float(pseudotime_norm[i])),
                "Pseudotime (0-1)": round(float(pseudotime_norm[i]), 3),
                "Reconstruction Error": round(float(rec_error[i]), 6),
                "Reliability": reliability_label(bool(ood_flags[i])),
            }
        )

    return pd.DataFrame(rows)


def plot_latent_by_cluster(latents, clusters):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    for cid in sorted(np.unique(clusters)):
        mask = clusters == cid
        label = CLUSTER_INFO.get(int(cid), {}).get("name", f"Cluster {cid}")
        ax.scatter(
            latents[mask, 0],
            latents[mask, 1],
            label=label,
            alpha=0.65,
            s=18,
        )

    ax.set_xlabel("z1")
    ax.set_ylabel("z2")
    ax.set_title("Latent Space Colored by Cluster")
    ax.legend(fontsize=7)
    return fig


def plot_latent_by_pseudotime(latents, pseudotime_norm):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sc = ax.scatter(
        latents[:, 0],
        latents[:, 1],
        c=pseudotime_norm,
        cmap="viridis",
        alpha=0.7,
        s=18,
    )
    ax.set_xlabel("z1")
    ax.set_ylabel("z2")
    ax.set_title("Latent Space Colored by Pseudotime")
    plt.colorbar(sc, ax=ax, label="Normalized Pseudotime")
    return fig


def plot_cluster_distribution(results_df):
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ordered = (
        results_df["Phenotype"]
        .value_counts()
        .reindex(results_df["Phenotype"].unique(), fill_value=0)
    )
    ordered.plot(kind="bar", ax=ax)
    ax.set_ylabel("Count")
    ax.set_title("Phenotype Distribution")
    return fig


def plot_pseudotime_distribution(results_df):
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.hist(results_df["Pseudotime (0-1)"], bins=20)
    ax.set_xlabel("Normalized Pseudotime")
    ax.set_ylabel("Count")
    ax.set_title("Pseudotime Distribution")
    return fig


def build_cluster_summary(results_df):
    summary = (
        results_df.groupby(["Cluster", "Phenotype", "Risk Category"], as_index=False)
        .agg(
            Count=("Pseudotime (0-1)", "count"),
            Mean_Pseudotime=("Pseudotime (0-1)", "mean"),
            Mean_Reconstruction_Error=("Reconstruction Error", "mean"),
        )
        .sort_values("Cluster", key=lambda s: s.map({c: i for i, c in enumerate(CLUSTER_ORDER)}))
    )

    summary["Mean_Pseudotime"] = summary["Mean_Pseudotime"].round(3)
    summary["Mean_Reconstruction_Error"] = summary["Mean_Reconstruction_Error"].round(6)
    return summary


def build_cluster_feature_profiles(df_clean, clusters):
    prof = df_clean.copy()
    prof["Cluster"] = clusters
    profile_means = prof.groupby("Cluster").mean(numeric_only=True).reset_index()
    profile_means["Phenotype"] = profile_means["Cluster"].map(
        lambda x: CLUSTER_INFO.get(int(x), {}).get("name", f"Cluster {x}")
    )

    ordered_cols = ["Cluster", "Phenotype"] + [
        c for c in profile_means.columns if c not in ["Cluster", "Phenotype"]
    ]
    profile_means = profile_means[ordered_cols]
    return profile_means


def decode_synthetic_from_transformed(syn_df: pd.DataFrame):
    decoded = pd.DataFrame()

    # Continuous
    for col in CONTINUOUS_COLS:
        tcol = f"cont__{col}"
        if tcol in syn_df.columns:
            decoded[col] = syn_df[tcol].clip(0, 1)

    # Binary
    for col in BINARY_COLS:
        tcol = f"bin__{col}"
        if tcol in syn_df.columns:
            decoded[col] = (syn_df[tcol] >= 0.5).astype(int)

    # Categorical
    for col in CATEGORICAL_COLS:
        prefix = f"cat__{col}_"
        matching = [c for c in syn_df.columns if c.startswith(prefix)]
        if matching:
            decoded[col] = (
                syn_df[matching]
                .idxmax(axis=1)
                .str.replace(prefix, "", regex=False)
            )

    return decoded


# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.header("Model Artifacts")
st.sidebar.write(f"Features: {len(feature_names)}")
st.sidebar.write(f"OOD percentile: {OOD_PERCENTILE}")
st.sidebar.write(f"OOD threshold: {OOD_THRESHOLD:.6f}")
st.sidebar.write(
    f"Pseudotime bounds: [{PT_BOUNDS['min']:.3f}, {PT_BOUNDS['max']:.3f}]"
)

st.sidebar.header("Expected Input Columns")
st.sidebar.caption("The app will auto-fill missing columns, but these are the trained schema fields.")
st.sidebar.write("Continuous:", CONTINUOUS_COLS)
st.sidebar.write("Binary:", BINARY_COLS)
st.sidebar.write("Categorical:", CATEGORICAL_COLS)


# ============================================================
# UPLOAD + ANALYSIS
# ============================================================
st.header("Upload Patient Cohort")
uploaded_file = st.file_uploader(
    "Upload a CSV file containing TB patient records",
    type=["csv"],
)
analyze = st.button("Analyze Cohort", type="primary")

results = None
df_clean = None
latents = None
pseudotime_norm = None
clusters = None
rec_error = None
ood_flags = None

if uploaded_file and analyze:
    try:
        df_raw = pd.read_csv(uploaded_file)

        if len(df_raw) > MAX_ROWS:
            st.error(
                f"The uploaded dataset contains {len(df_raw):,} rows. "
                f"This deployment supports up to {MAX_ROWS:,} rows per run."
            )
            st.stop()

        df_clean, X = transform_uploaded_data(df_raw)

        latents = compute_latent(model, X)
        pseudotime_norm = compute_pseudotime(latents, bounds=PT_BOUNDS)
        clusters = assign_cluster(kmeans, latents)

        rec_error = batched_reconstruction_error(model, X, batch_size=BATCH_SIZE)
        ood_flags = rec_error > OOD_THRESHOLD

        results = build_patient_results(
            latents=latents,
            pseudotime_norm=pseudotime_norm,
            clusters=clusters,
            rec_error=rec_error,
            ood_flags=ood_flags,
        )

        st.success("Cohort processed successfully.")

    except Exception as e:
        st.error(
            "The uploaded data could not be processed. "
            "Please confirm the CSV structure is compatible with the trained schema."
        )
        st.exception(e)
        st.stop()


# ============================================================
# RESULTS
# ============================================================
if results is not None:
    st.header("Cohort Summary")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Patients", f"{len(results):,}")
    c2.metric("Average Pseudotime", f"{results['Pseudotime (0-1)'].mean():.2f}")
    c3.metric("OOD Warnings", int(ood_flags.sum()))
    c4.metric("Phenotypes Detected", int(results["Phenotype"].nunique()))

    st.header("Patient-Level Results")
    st.dataframe(results, use_container_width=True)

    st.header("Latent Space Interpretation")
    col1, col2 = st.columns(2)

    with col1:
        st.pyplot(plot_latent_by_cluster(latents, clusters), use_container_width=True)

    with col2:
        st.pyplot(plot_latent_by_pseudotime(latents, pseudotime_norm), use_container_width=True)

    st.header("Distribution Views")
    col3, col4 = st.columns(2)

    with col3:
        st.pyplot(plot_cluster_distribution(results), use_container_width=True)

    with col4:
        st.pyplot(plot_pseudotime_distribution(results), use_container_width=True)

    st.header("Detailed Interpretation")

    with st.expander("Cluster-Level Summary", expanded=True):
        summary = build_cluster_summary(results)
        st.dataframe(summary, use_container_width=True)

    with st.expander("Phenotype Definitions", expanded=False):
        for cid in CLUSTER_ORDER:
            info = CLUSTER_INFO[cid]
            st.markdown(
                f"**Cluster {cid}: {info['name']}**  \n"
                f"- Stage: {info['stage']}  \n"
                f"- Risk: {info['risk']}  \n"
                f"- Description: {info['summary']}  \n"
                f"- Key features: {', '.join(info['key_features'])}"
            )

    with st.expander("Cluster Feature Profiles", expanded=False):
        profile_means = build_cluster_feature_profiles(df_clean, clusters)
        st.dataframe(profile_means, use_container_width=True)
        st.download_button(
            "Download Cluster Feature Profiles",
            profile_means.to_csv(index=False),
            file_name="cluster_feature_profiles.csv",
            mime="text/csv",
        )

    with st.expander("Uploaded Data Preview", expanded=False):
        st.dataframe(df_clean.head(200), use_container_width=True)

    st.download_button(
        "Download Patient-Level Results",
        results.to_csv(index=False),
        file_name="tb_risk_results.csv",
        mime="text/csv",
    )

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
