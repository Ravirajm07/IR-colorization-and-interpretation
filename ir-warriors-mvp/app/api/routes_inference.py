from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from fastapi import APIRouter, BackgroundTasks, HTTPException
from PIL import Image
import matplotlib.pyplot as plt

from app.models.model_loader import load_generator
from app.schemas.requests import PredictRequest, PreprocessRequest
from app.schemas.responses import JobResponse
from app.services.geospatial_service import export_rgb_geotiff
from app.services.inference_service import predict_array
from app.services.job_service import jobs
from app.services.metrics_service import image_metrics
from app.services.preprocessing_service import clip_scene_to_pune, preprocess_landsat_scene, preprocess_png, save_preprocessed_scene, scene_summary


router = APIRouter(tags=["processing"])


@router.post("/preprocess", response_model=JobResponse)
def preprocess(request: PreprocessRequest, background: BackgroundTasks) -> dict:
    try:
        job = jobs.get(request.job_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    if job["status"] not in {"uploaded", "preprocessing_failed"}:
        raise HTTPException(409, f"Job cannot be preprocessed from status '{job['status']}'.")
    jobs.update(request.job_id, status="preprocessing")
    background.add_task(_preprocess_job, request)
    return jobs.get(request.job_id)


def _preprocess_job(request: PreprocessRequest) -> None:
    try:
        job = jobs.get(request.job_id)
        source = Path(job["source_path"])
        if source.suffix.lower() == ".png":
            scene = preprocess_png(source)
        else:
            scene = preprocess_landsat_scene(
                source,
                require_rgb=False,
                apply_cloud_mask=request.cloud_mask,
                thermal_range=(request.thermal_min_kelvin, request.thermal_max_kelvin),
                gamma=request.gamma,
            )
        if request.clip_to_pune:
            if not request.boundary_path:
                raise ValueError("Pune clipping was requested but no Maharashtra district boundary path was provided.")
            scene = clip_scene_to_pune(scene, request.boundary_path)
        paths = save_preprocessed_scene(scene, jobs.directory(request.job_id) / "processed")
        jobs.update(
            request.job_id,
            status="preprocessed",
            input_metadata=scene_summary(scene),
            output_paths={key: str(value) for key, value in paths.items()},
        )
    except Exception as exc:
        jobs.update(request.job_id, status="preprocessing_failed", warnings=[str(exc)])


@router.post("/predict", response_model=JobResponse)
def predict(request: PredictRequest, background: BackgroundTasks) -> dict:
    try:
        job = jobs.get(request.job_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    if job["status"] not in {"preprocessed", "inference_failed", "complete"}:
        raise HTTPException(409, f"Job cannot run inference from status '{job['status']}'.")
    jobs.update(request.job_id, status="inference")
    background.add_task(_predict_job, request)
    return jobs.get(request.job_id)


def _predict_job(request: PredictRequest) -> None:
    try:
        job = jobs.get(request.job_id)
        paths = job["output_paths"]
        thermal = np.load(paths["thermal"])
        valid = np.load(paths["mask"]).astype(bool)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_generator(request.checkpoint_path, device)
        prediction, elapsed = predict_array(model, thermal, device)
        prediction[:, ~valid] = 0
        output_dir = jobs.directory(request.job_id) / "outputs"
        output_dir.mkdir(exist_ok=True)
        preview = output_dir / "generated_rgb.png"
        Image.fromarray((np.moveaxis(prediction, 0, -1) * 255).astype(np.uint8)).save(preview)
        paths["preview"] = str(preview)
        metadata = json.loads(Path(paths["metadata"]).read_text(encoding="utf-8"))
        if metadata.get("crs"):
            geotiff = export_rgb_geotiff(prediction, output_dir / "generated_rgb_geotiff.tif", metadata["crs"], metadata["transform"], valid)
            paths["geotiff"] = str(geotiff)
        metrics: dict[str, float | str] = {"inference_time_ms": elapsed}
        warnings = list(job.get("warnings", []))
        if request.compute_metrics and "rgb" in paths:
            target = np.load(paths["rgb"])
            metrics.update(image_metrics(prediction, target, valid))
            metrics_csv = output_dir / "metrics.csv"
            pd.DataFrame([metrics]).to_csv(metrics_csv, index=False)
            paths["metrics_csv"] = str(metrics_csv)
            comparison = output_dir / "comparison.png"
            figure, axes = plt.subplots(1, 4, figsize=(14, 4))
            values = (thermal, np.moveaxis(target, 0, -1), np.moveaxis(prediction, 0, -1), np.mean(np.abs(prediction - target), axis=0))
            titles = ("Thermal IR", "Reference RGB", "Generated RGB", "Difference")
            for axis, value, title in zip(axes, values, titles):
                axis.imshow(value, cmap="magma" if value.ndim == 2 else None)
                axis.set_title(title)
                axis.axis("off")
            figure.tight_layout()
            figure.savefig(comparison, dpi=160)
            plt.close(figure)
            paths["comparison"] = str(comparison)
        elif request.compute_metrics:
            warnings.append("Reference RGB is unavailable; full-reference metrics were not computed.")
        jobs.update(request.job_id, status="complete", output_paths=paths, metrics=metrics, warnings=warnings)
    except torch.cuda.OutOfMemoryError:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        jobs.update(request.job_id, status="inference_failed", warnings=["GPU out of memory. Use a smaller image or CPU mode."])
    except Exception as exc:
        jobs.update(request.job_id, status="inference_failed", warnings=[str(exc)])


@router.get("/status/{job_id}", response_model=JobResponse)
def status(job_id: str) -> dict:
    try:
        return jobs.get(job_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
