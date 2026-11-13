from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from skimage.metrics import structural_similarity


def image_metrics(prediction: np.ndarray, target: np.ndarray, valid_mask: np.ndarray) -> dict[str, float]:
    if prediction.shape != target.shape or prediction.shape[0] != 3:
        raise ValueError("Prediction and target must both have shape [3, H, W].")
    if valid_mask.shape != prediction.shape[1:] or not valid_mask.any():
        raise ValueError("A non-empty validity mask matching the image is required.")
    differences = prediction[:, valid_mask] - target[:, valid_mask]
    mse = float(np.mean(differences**2))
    mae = float(np.mean(np.abs(differences)))
    rmse = math.sqrt(mse)
    psnr = float("inf") if mse == 0 else 10.0 * math.log10(1.0 / mse)
    _, ssim_map = structural_similarity(
        target,
        prediction,
        data_range=1.0,
        channel_axis=0,
        gaussian_weights=True,
        full=True,
    )
    if ssim_map.ndim == 3:
        ssim_map = ssim_map.mean(axis=0)
    return {"psnr": psnr, "ssim": float(ssim_map[valid_mask].mean()), "mae": mae, "rmse": rmse}


def optional_perceptual_metrics(
    predictions: list[torch.Tensor], targets: list[torch.Tensor], device: torch.device
) -> tuple[dict[str, float | str], list[str]]:
    metrics: dict[str, float | str] = {"lpips": "Metric unavailable", "fid": "Metric unavailable"}
    warnings: list[str] = []
    if not predictions:
        return metrics, ["LPIPS and FID require test predictions."]
    try:
        import lpips

        loss_fn = lpips.LPIPS(net="alex").to(device)
        with torch.inference_mode():
            values = [loss_fn(p.to(device) * 2 - 1, t.to(device) * 2 - 1).item() for p, t in zip(predictions, targets)]
        metrics["lpips"] = float(np.mean(values))
    except Exception as exc:
        warnings.append(f"LPIPS unavailable: {exc}")
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance

        if len(predictions) < 2:
            raise ValueError("FID requires at least two test samples.")
        fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
        for prediction, target in zip(predictions, targets):
            fid.update(target.to(device), real=True)
            fid.update(prediction.to(device), real=False)
        metrics["fid"] = float(fid.compute().item())
    except Exception as exc:
        warnings.append(f"FID unavailable: {exc}")
    return metrics, warnings


def aggregate_metrics(rows: list[dict[str, Any]]) -> dict[str, float | str]:
    summary: dict[str, float | str] = {}
    for key in ("psnr", "ssim", "mae", "rmse", "inference_time_ms"):
        values = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float)) and np.isfinite(row[key])]
        summary[key] = float(np.mean(values)) if values else "Metric unavailable"
    return summary
