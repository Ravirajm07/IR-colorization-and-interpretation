from __future__ import annotations

from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image


def render_image_viewer(job: dict) -> None:
    paths = job.get("output_paths", {})
    columns = st.columns(4)
    thermal_path = paths.get("thermal")
    rgb_path = paths.get("rgb")
    preview_path = paths.get("preview")
    if thermal_path and Path(thermal_path).exists():
        thermal = np.load(thermal_path)
        columns[0].image(thermal, caption="Original Thermal IR", clamp=True)
    else:
        columns[0].info("Thermal preview unavailable")
    if rgb_path and Path(rgb_path).exists():
        target = np.load(rgb_path)
        columns[1].image(np.moveaxis(target, 0, -1), caption="Ground-Truth RGB", clamp=True)
    else:
        columns[1].info("Ground truth unavailable")
    if preview_path and Path(preview_path).exists():
        generated = np.asarray(Image.open(preview_path))
        columns[2].image(generated, caption="Generated RGB")
        if rgb_path and Path(rgb_path).exists():
            target = np.moveaxis(np.load(rgb_path), 0, -1)
            difference = np.mean(np.abs(generated.astype(np.float32) / 255.0 - target), axis=2)
            columns[3].image(difference, caption="Absolute Difference", clamp=True)
        else:
            columns[3].info("Difference requires ground truth")
    else:
        columns[2].info("Run inference to generate RGB")
        columns[3].info("Difference unavailable")
