from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

from frontend.components.image_viewer import render_image_viewer
from frontend.components.map_viewer import render_map
from frontend.components.metrics_cards import render_metrics
from frontend.components.report_panel import render_report_panel
from frontend.components.uploader import render_uploader


API_URL = os.getenv("IRW_API_URL", "http://localhost:8000")
ROOT = Path(__file__).resolve().parents[1]
st.set_page_config(page_title="IR Warriors", layout="wide", initial_sidebar_state="auto")
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; letter-spacing: 0; }
    .stApp { background: #f5f8fa; color: #102a3a; }
    [data-testid="stSidebar"] { background: #071d2e; }
    [data-testid="stSidebar"] * { color: #eaf7fb; }
    [data-testid="stHeader"] { background: rgba(7,29,46,.96); }
    div[data-testid="stMetric"] { background: white; border: 1px solid #d7e4ea; border-radius: 6px; padding: 16px; box-shadow: 0 3px 12px rgba(7,29,46,.06); }
    .irw-header { background:#071d2e; color:white; padding:18px 24px; border-left:5px solid #00b8d9; margin:-1rem 0 1.2rem; box-sizing:border-box; }
    .irw-header h1 { font-size:24px; margin:0; letter-spacing:0; }
    .irw-header p { color:#9edcea; margin:5px 0 0; font-size:13px; overflow-wrap:anywhere; }
    .status { display:inline-block; border-radius:4px; padding:3px 8px; font-size:12px; font-weight:600; background:#dff7ea; color:#117744; }
    .planned { background:#edf1f4; color:#526674; }
    .stButton button, .stDownloadButton button { border-radius:6px; }
    </style>
    """,
    unsafe_allow_html=True,
)


def api_health() -> dict:
    try:
        response = requests.get(f"{API_URL}/health", timeout=3)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return {"status": "offline", "cuda_available": False, "model_version": "unavailable"}


health = api_health()
st.markdown(
    f'<div class="irw-header"><h1>IR WARRIORS | Satellite Interpretation Console</h1>'
    f'<p>IR enhancement and RGB translation · API {health["status"]} · GPU {"available" if health["cuda_available"] else "CPU mode"} · {health["model_version"]}</p></div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    logo = ROOT / "frontend" / "assets" / "logo.png"
    if logo.exists():
        st.image(str(logo), width=150)
    page = st.radio(
        "Workspace",
        ["Dashboard", "Upload Dataset", "Preprocessing", "AI Inference", "Evaluation", "GIS Viewer", "Reports", "Settings"],
    )

job = st.session_state.get("job")
if job and job.get("status") in {"preprocessing", "inference"}:
    try:
        job = requests.get(f"{API_URL}/status/{job['job_id']}", timeout=5).json()
        st.session_state.job = job
    except requests.RequestException as exc:
        st.warning(f"Job status could not be refreshed: {exc}")

if page == "Dashboard":
    st.subheader("Mission Dashboard")
    metrics = job.get("metrics", {}) if job else {}
    manifest_path = ROOT / "data/patches/manifest.csv"
    patch_count = len(pd.read_csv(manifest_path)) if manifest_path.exists() else 0
    cards = st.columns(6)
    cards[0].metric("Total Scenes", 1 if job else 0)
    cards[1].metric("Processed Patches", patch_count)
    cards[2].metric("Model Status", "Ready" if (ROOT / "checkpoints/best_generator.pth").exists() else "Required")
    cards[3].metric("Average PSNR", f"{metrics['psnr']:.3f} dB" if isinstance(metrics.get("psnr"), (int, float)) else "N/A")
    cards[4].metric("Average SSIM", f"{metrics['ssim']:.4f}" if isinstance(metrics.get("ssim"), (int, float)) else "N/A")
    cards[5].metric("Inference Time", f"{metrics['inference_time_ms']:.1f} ms" if isinstance(metrics.get("inference_time_ms"), (int, float)) else "N/A")
    st.subheader("Recent Job")
    st.dataframe(pd.DataFrame([job]) if job else pd.DataFrame(columns=["job_id", "status"]), use_container_width=True)
    if job:
        st.progress(1.0 if job["status"] == "complete" else 0.55 if job["status"] in {"preprocessing", "inference"} else 0.2)
    st.caption(f"Storage: {ROOT / 'data'} | API: {API_URL}")

elif page == "Upload Dataset":
    st.subheader("Dataset Intake")
    render_uploader(API_URL)

elif page == "Preprocessing":
    st.subheader("Landsat Preprocessing")
    left, right = st.columns(2)
    with left:
        clip = st.toggle("Clip to Pune district", value=False)
        boundary_path = st.text_input("Maharashtra district boundary path", value="data/boundaries/maharashtra_district.shp", disabled=not clip)
        cloud = st.toggle("Apply QA_PIXEL cloud mask", value=True)
        patch_size = st.number_input("Patch size", 64, 1024, 256, step=32)
        stride = st.number_input("Stride", 32, 1024, 192, step=32)
    with right:
        valid = st.slider("Minimum valid fraction", 0.0, 1.0, 0.75, 0.05)
        minimum, maximum = st.slider("Thermal range (Kelvin)", 200.0, 380.0, (250.0, 330.0))
        gamma = st.slider("Gamma correction", 0.1, 1.5, 1.0 / 2.2, 0.01)
    st.session_state.preprocess_options = {
        "clip_to_pune": clip,
        "boundary_path": boundary_path if clip else None,
        "cloud_mask": cloud,
        "patch_size": patch_size,
        "stride": stride,
        "min_valid_fraction": valid,
        "thermal_min_kelvin": minimum,
        "thermal_max_kelvin": maximum,
        "gamma": gamma,
    }
    if job:
        st.info(f"Pipeline status: {job['status']}")
        render_image_viewer(job)
    else:
        st.info("Upload and validate a scene to begin preprocessing.")

elif page == "AI Inference":
    st.subheader("AI Inference")
    labels = st.columns(3)
    labels[0].markdown('<span class="status">Pix2PixHD · Implemented</span>', unsafe_allow_html=True)
    labels[1].markdown('<span class="status planned">SwinIR · Planned Enhancement</span>', unsafe_allow_html=True)
    labels[2].markdown('<span class="status planned">SegFormer · Planned Enhancement</span>', unsafe_allow_html=True)
    checkpoint = st.text_input("Checkpoint path (optional)", placeholder="Uses checkpoints/best_generator.pth")
    if st.button("Run Inference", type="primary", disabled=not job or job.get("status") not in {"preprocessed", "complete", "inference_failed"}):
        response = requests.post(
            f"{API_URL}/predict", json={"job_id": job["job_id"], "checkpoint_path": checkpoint or None}, timeout=30
        )
        if response.ok:
            st.session_state.job = response.json()
            st.rerun()
        else:
            st.error(response.json().get("detail", response.text))
    if job:
        render_image_viewer(job)

elif page == "Evaluation":
    st.subheader("Held-Out Evaluation")
    render_metrics(job.get("metrics", {}) if job else {})
    st.warning("Metrics reflect the current prototype dataset and should not be interpreted as nationwide operational performance.")
    history_path = ROOT / "checkpoints" / "training_history.csv"
    if history_path.exists():
        history = pd.read_csv(history_path)
        st.line_chart(history.set_index("epoch")[["generator_loss", "discriminator_loss", "validation_l1"]])
    else:
        st.info("Training curves will appear after model training.")

elif page == "GIS Viewer":
    st.subheader("Geospatial Result Viewer")
    render_map(job or {})

elif page == "Reports":
    st.subheader("Reports and Artifacts")
    if job:
        render_report_panel(API_URL, job)
    else:
        st.info("Complete a job before generating reports.")

else:
    st.subheader("Settings")
    st.text_input("Backend API", value=API_URL, disabled=True)
    st.text_input("Model version", value=health["model_version"], disabled=True)
    st.checkbox("CPU fallback enabled", value=True, disabled=True)
