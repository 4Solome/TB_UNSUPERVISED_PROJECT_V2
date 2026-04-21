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
    page_icon="📊🫁",
    layout="wide",
)

# ============================================================
# CUSTOM UI STYLING
# ============================================================
st.markdown(
    """
    <style>
    :root {
        --bg: #060b17;
        --card: rgba(13, 20, 38, 0.88);
        --card-2: rgba(12, 18, 34, 0.96);
        --border: rgba(114, 137, 218, 0.22);
        --text: #eef2ff;
        --muted: #a8b2d1;
        --pink: #ff4d8d;
        --purple: #7c4dff;
        --cyan: #2dd4bf;
        --blue: #3b82f6;
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(76, 29, 149, 0.20), transparent 25%),
            radial-gradient(circle at top right, rgba(37, 99, 235, 0.14), transparent 25%),
            linear-gradient(180deg, #040915 0%, #050b17 50%, #06101d 100%);
        color: var(--text);
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1320px;
    }

    h1, h2, h3, h4, h5, h6, p, label, div {
        color: var(--text);
    }

    .hero-wrap {
        padding: 0.8rem 0 1.2rem 0;
    }

    .hero-badge {
        display: inline-block;
        padding: 0.42rem 0.9rem;
        border-radius: 999px;
        background: linear-gradient(90deg, rgba(124,77,255,0.25), rgba(59,130,246,0.18));
        border: 1px solid rgba(124,77,255,0.25);
        color: #d9d7ff;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        margin-bottom: 1rem;
    }

    .hero-title {
        font-size: 4rem;
        line-height: 1.02;
        font-weight: 800;
        margin: 0 0 0.85rem 0;
        letter-spacing: -0.03em;
    }

    .hero-subtitle {
        font-size: 1.35rem;
        line-height: 1.6;
        color: var(--muted);
        max-width: 760px;
        margin-bottom: 1.5rem;
    }

    .feature-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.9rem;
        margin-top: 0.8rem;
    }

    .feature-card {
        background: rgba(13, 20, 38, 0.70);
        border: 1px solid rgba(114, 137, 218, 0.18);
        border-radius: 18px;
        padding: 1rem 1rem 0.95rem 1rem;
        min-height: 120px;
        box-shadow: 0 8px 30px rgba(0,0,0,0.18);
    }

    .feature-title {
        font-size: 1.02rem;
        font-weight: 700;
        margin-bottom: 0.35rem;
    }

    .feature-text {
        color: var(--muted);
        font-size: 0.95rem;
        line-height: 1.5;
    }

    .visual-card {
        background:
            radial-gradient(circle at center, rgba(124,77,255,0.26), transparent 45%),
            linear-gradient(180deg, rgba(15,22,40,0.95), rgba(9,14,28,0.95));
        border: 1px solid rgba(114, 137, 218, 0.18);
        border-radius: 28px;
        min-height: 370px;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: inset 0 0 60px rgba(62, 90, 255, 0.08);
        position: relative;
        overflow: hidden;
    }

    .visual-card:before,
    .visual-card:after {
        content: "";
        position: absolute;
        width: 180px;
        height: 180px;
        border: 1px solid rgba(124,77,255,0.14);
        border-radius: 24px;
        transform: rotate(24deg);
    }

    .visual-card:before {
        top: 18px;
        right: 42px;
    }

    .visual-card:after {
        bottom: 22px;
        left: 40px;
    }

    .lung {
        font-size: 10rem;
        filter: drop-shadow(0 0 25px rgba(124,77,255,0.50));
    }

    .section-card {
        background: linear-gradient(180deg, rgba(13,20,38,0.96), rgba(8,14,28,0.98));
        border: 1px solid rgba(114, 137, 218, 0.20);
        border-radius: 24px;
        padding: 1.15rem 1.15rem 1.25rem 1.15rem;
        box-shadow: 0 18px 40px rgba(0, 0, 0, 0.22);
        margin-top: 1rem;
        margin-bottom: 1rem;
    }

    .section-head {
        display: flex;
        gap: 0.9rem;
        align-items: flex-start;
        margin-bottom: 0.8rem;
    }

    .section-icon {
        width: 56px;
        height: 56px;
        border-radius: 18px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.6rem;
        background: linear-gradient(135deg, rgba(124,77,255,0.28), rgba(59,130,246,0.18));
        border: 1px solid rgba(124,77,255,0.22);
        flex-shrink: 0;
    }

    .section-title {
        font-size: 1.75rem;
        font-weight: 800;
        margin: 0;
    }

    .section-subtitle {
        color: var(--muted);
        margin-top: 0.2rem;
        font-size: 1rem;
    }

    .soft-note {
        margin-top: 0.85rem;
        padding: 0.9rem 1rem;
        border-radius: 16px;
        background: rgba(8, 13, 26, 0.72);
        border: 1px solid rgba(114, 137, 218, 0.16);
        color: var(--muted);
        font-size: 0.96rem;
    }

    .footer-note {
        margin-top: 1rem;
        padding: 0.9rem 1rem;
        border-radius: 16px;
        background: rgba(8, 13, 26, 0.60);
        border: 1px solid rgba(114, 137, 218, 0.18);
        color: var(--muted);
        font-size: 0.95rem;
    }

    .metric-card {
        background: linear-gradient(180deg, rgba(14, 22, 41, 0.96), rgba(10, 15, 28, 0.96));
        border: 1px solid rgba(114, 137, 218, 0.18);
        border-radius: 18px;
        padding: 0.35rem 0.35rem 0.2rem 0.35rem;
        box-shadow: 0 10px 24px rgba(0,0,0,0.16);
    }

    div[data-testid="stMetric"] {
        background: transparent;
        border: none;
        padding: 0.65rem 0.8rem;
        border-radius: 16px;
    }

    div[data-testid="stMetricLabel"] {
        color: var(--muted);
        font-weight: 600;
    }

    div[data-testid="stMetricValue"] {
        color: white;
        font-weight: 800;
    }

    .stButton > button {
        border: none !important;
        border-radius: 14px !important;
        padding: 0.78rem 1.35rem !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        color: white !important;
        background: linear-gradient(90deg, var(--pink), var(--purple)) !important;
        box-shadow: 0 10px 25px rgba(124,77,255,0.26) !important;
    }

    .stButton > button:hover {
        filter: brightness(1.06);
        transform: translateY(-1px);
    }

    div[data-testid="stFileUploader"] {
        background: rgba(10, 16, 30, 0.72);
        border: 1px solid rgba(114, 137, 218, 0.18);
        border-radius: 18px;
        padding: 0.9rem;
    }

    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    .stSlider {
        background: transparent;
    }

    .stDataFrame, .stTable, div[data-testid="stExpander"] {
        border-radius: 18px;
        overflow: hidden;
    }

    div[data-testid="stExpander"] {
        border: 1px solid rgba(114, 137, 218, 0.16);
        background: rgba(11, 16, 30, 0.75);
    }

    hr {
        border-color: rgba(114, 137, 218, 0.12);
    }

    @media (max-width: 1100px) {
        .hero-title {
            font-size: 2.8rem;
        }
        .feature-grid {
            grid-template-columns: 1fr;
        }
        .visual-card {
            min-height: 240px;
        }
        .lung {
            font-size: 6rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
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
    fig.patch.set_facecolor("#0b1324")
    ax.set_facecolor("#0b1324")

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

    ax.set_xlabel("z1", color="white")
    ax.set_ylabel("z2", color="white")
    ax.set_title("Latent Space Colored by Cluster", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#2a3550")
    leg = ax.legend(fontsize=7)
    if leg:
        for text in leg.get_texts():
            text.set_color("white")
        leg.get_frame().set_facecolor("#0b1324")
        leg.get_frame().set_edgecolor("#2a3550")
    return fig


def plot_latent_by_pseudotime(latents, pseudotime_norm):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    fig.patch.set_facecolor("#0b1324")
    ax.set_facecolor("#0b1324")

    sc = ax.scatter(
        latents[:, 0],
        latents[:, 1],
        c=pseudotime_norm,
        cmap="viridis",
        alpha=0.7,
        s=18,
    )
    ax.set_xlabel("z1", color="white")
    ax.set_ylabel("z2", color="white")
    ax.set_title("Latent Space Colored by Pseudotime", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#2a3550")
    cbar = plt.colorbar(sc, ax=ax, label="Normalized Pseudotime")
    cbar.ax.yaxis.label.set_color("white")
    cbar.ax.tick_params(colors="white")
    cbar.outline.set_edgecolor("#2a3550")
    return fig


def plot_cluster_distribution(results_df):
    fig, ax = plt.subplots(figsize=(5, 3.5))
    fig.patch.set_facecolor("#0b1324")
    ax.set_facecolor("#0b1324")

    ordered = (
        results_df["Phenotype"]
        .value_counts()
        .reindex(results_df["Phenotype"].unique(), fill_value=0)
    )
    ordered.plot(kind="bar", ax=ax)
    ax.set_ylabel("Count", color="white")
    ax.set_title("Phenotype Distribution", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#2a3550")
    return fig


def plot_pseudotime_distribution(results_df):
    fig, ax = plt.subplots(figsize=(5, 3.5))
    fig.patch.set_facecolor("#0b1324")
    ax.set_facecolor("#0b1324")

    ax.hist(results_df["Pseudotime (0-1)"], bins=20)
    ax.set_xlabel("Normalized Pseudotime", color="white")
    ax.set_ylabel("Count", color="white")
    ax.set_title("Pseudotime Distribution", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#2a3550")
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
# HERO SECTION (BETTER ALIGNED STREAMLIT VERSION)
# ============================================================
left_col, right_col = st.columns([1.6, 1], gap="large")

with left_col:
    st.markdown("### ✦ AI-POWERED")
    st.markdown("# TB Risk Profiling System")
    st.markdown(
        """
        Unsupervised representation learning and generative modeling for latent
        tuberculosis risk sequencing and phenotype discovery.
        """
    )

    st.markdown("### Key Capabilities")

    st.markdown(
        """
        **🛡 Privacy Preserving**  
        Generative synthetic data without exposing real patients
        """
    )

    st.markdown(
        """
        **✦ Advanced AI**  
        Latent representation learning and progression modeling
        """
    )

    st.markdown(
        """
        **📊 Actionable Insights**  
        Identify high-risk patterns and patient phenotypes
        """
    )

with right_col:
    st.markdown("<div style='height: 70px;'></div>", unsafe_allow_html=True)
    st.markdown(
        "<div style='text-align: center; font-size: 120px;'>🫁</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<h3 style='text-align: center; margin-top: -10px;'>TB Risk Visualization</h3>",
        unsafe_allow_html=True,
    )

# ============================================================
# UPLOAD + ANALYSIS
# ============================================================
st.markdown(
    """
    <div class="section-card">
        <div class="section-head">
            <div class="section-icon">☁️</div>
            <div>
                <div class="section-title">Upload Patient Cohort</div>
                <div class="section-subtitle">Upload a CSV file containing TB patient records.</div>
            </div>
        </div>
    """,
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Upload a CSV file containing TB patient records",
    type=["csv"],
    label_visibility="collapsed",
)

st.markdown(
    """
    <div class="soft-note">
        Our system will auto-fill missing columns based on the trained schema.
    </div>
    """,
    unsafe_allow_html=True,
)

analyze = st.button("Analyze Cohort", type="primary")

st.markdown("</div>", unsafe_allow_html=True)

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
    st.markdown("### Cohort Summary")

    m1, m2, m3, m4 = st.columns(4, gap="medium")
    with m1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Patients", f"{len(results):,}")
        st.markdown("</div>", unsafe_allow_html=True)
    with m2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Average Pseudotime", f"{results['Pseudotime (0-1)'].mean():.2f}")
        st.markdown("</div>", unsafe_allow_html=True)
    with m3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("OOD Warnings", int(ood_flags.sum()))
        st.markdown("</div>", unsafe_allow_html=True)
    with m4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Phenotypes Detected", int(results["Phenotype"].nunique()))
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("### Patient-Level Results")
    st.dataframe(results, use_container_width=True)

    st.markdown("### Latent Space Interpretation")
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.pyplot(plot_latent_by_cluster(latents, clusters), use_container_width=True)

    with col2:
        st.pyplot(plot_latent_by_pseudotime(latents, pseudotime_norm), use_container_width=True)

    st.markdown("### Distribution Views")
    col3, col4 = st.columns(2, gap="large")

    with col3:
        st.pyplot(plot_cluster_distribution(results), use_container_width=True)

    with col4:
        st.pyplot(plot_pseudotime_distribution(results), use_container_width=True)

    st.markdown("### Detailed Interpretation")

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
