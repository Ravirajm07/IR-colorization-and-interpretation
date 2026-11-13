import requests
import streamlit as st


def render_report_panel(api_url: str, job: dict) -> None:
    job_id = job["job_id"]
    if st.button("Generate PDF Report", type="primary"):
        try:
            response = requests.post(f"{api_url}/generate-report", json={"job_id": job_id}, timeout=60)
            response.raise_for_status()
            st.success("Report generated.")
        except requests.RequestException as exc:
            st.error(str(exc))
    artifacts = [
        ("Generated RGB PNG", "preview"),
        ("Generated RGB GeoTIFF", "geotiff"),
        ("Comparison Image", "comparison"),
        ("Metrics CSV", "metrics_csv"),
        ("PDF Report", "report"),
        ("Model ZIP", "model_zip"),
        ("Project Configuration", "configuration"),
    ]
    for label, file_type in artifacts:
        st.link_button(label, f"{api_url}/download/{job_id}/{file_type}", use_container_width=True)
