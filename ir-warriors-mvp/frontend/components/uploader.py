from __future__ import annotations

import requests
import streamlit as st


def render_uploader(api_url: str) -> None:
    uploaded = st.file_uploader("Landsat scene or thermal input", type=["zip", "tif", "tiff", "png"])
    left, middle, right = st.columns(3)
    if left.button("Validate Dataset", type="primary", use_container_width=True, disabled=uploaded is None):
        with st.spinner("Validating scene structure..."):
            try:
                response = requests.post(
                    f"{api_url}/upload-scene",
                    files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type or "application/octet-stream")},
                    timeout=300,
                )
                response.raise_for_status()
                st.session_state.job = response.json()
                st.success("Dataset accepted and validated.")
            except requests.RequestException as exc:
                detail = exc.response.json().get("detail", str(exc)) if exc.response is not None else str(exc)
                st.error(detail)
    if middle.button("Start Preprocessing", use_container_width=True, disabled="job" not in st.session_state):
        payload = {"job_id": st.session_state.job["job_id"], **st.session_state.get("preprocess_options", {})}
        try:
            response = requests.post(f"{api_url}/preprocess", json=payload, timeout=30)
            response.raise_for_status()
            st.session_state.job = response.json()
            st.info("Preprocessing started.")
        except requests.RequestException as exc:
            st.error(str(exc))
    if right.button("Clear", use_container_width=True):
        st.session_state.pop("job", None)
        st.rerun()

    job = st.session_state.get("job")
    if job:
        metadata = job.get("input_metadata", {})
        st.subheader("Validation Results")
        columns = st.columns(4)
        fields = [
            ("Satellite", metadata.get("satellite", "Pending preprocessing")),
            ("Scene ID", metadata.get("scene_id", "Unavailable")),
            ("Thermal Band", metadata.get("thermal_band", "Unavailable")),
            ("QA Band", metadata.get("qa_band", "Unavailable")),
            ("CRS", metadata.get("crs", "Pending")),
            ("Resolution", metadata.get("resolution", "Pending")),
            ("RGB Bands", metadata.get("rgb_bands", "Pending")),
            ("Cloud Mask", metadata.get("cloud_mask_available", "Pending")),
        ]
        for index, (label, value) in enumerate(fields):
            columns[index % 4].metric(label, str(value))
