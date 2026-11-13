from __future__ import annotations

import streamlit as st


def render_metrics(metrics: dict) -> None:
    columns = st.columns(7)
    fields = [
        ("PSNR", "psnr", " dB"),
        ("SSIM", "ssim", ""),
        ("FID", "fid", ""),
        ("LPIPS", "lpips", ""),
        ("MAE", "mae", ""),
        ("RMSE", "rmse", ""),
        ("Inference", "inference_time_ms", " ms"),
    ]
    for column, (label, key, suffix) in zip(columns, fields):
        value = metrics.get(key, "Metric unavailable")
        formatted = f"{value:.4f}{suffix}" if isinstance(value, (int, float)) else str(value)
        column.metric(label, formatted)
