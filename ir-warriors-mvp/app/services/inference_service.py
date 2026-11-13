from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

from app.models.model_loader import load_generator
from app.services.geospatial_service import export_rgb_geotiff
from app.services.metrics_service import aggregate_metrics, image_metrics, optional_perceptual_metrics
from app.services.report_service import generate_evaluation_pdf
from app.utils.config import get_settings


def predict_array(
    model: torch.nn.Module,
    thermal: np.ndarray,
    device: torch.device,
    tile_size: int = 512,
    overlap: int = 64,
) -> tuple[np.ndarray, float]:
    if thermal.ndim != 2:
        raise ValueError("Thermal input must have shape [H, W].")
    if tile_size <= 0 or overlap < 0 or overlap >= tile_size:
        raise ValueError("Tile size must be positive and overlap must be smaller than the tile.")
    height, width = thermal.shape
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    accumulator = np.zeros((3, height, width), dtype=np.float32)
    weights = np.zeros((height, width), dtype=np.float32)
    stride = tile_size - overlap
    y_positions = _tile_positions(height, tile_size, stride)
    x_positions = _tile_positions(width, tile_size, stride)
    with torch.inference_mode():
        for y in y_positions:
            for x in x_positions:
                tile = thermal[y : min(y + tile_size, height), x : min(x + tile_size, width)]
                predicted_tile = _predict_tile(model, tile, device)
                tile_height, tile_width = tile.shape
                accumulator[:, y : y + tile_height, x : x + tile_width] += predicted_tile
                weights[y : y + tile_height, x : x + tile_width] += 1.0
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - started) * 1000.0
    result = np.divide(accumulator, weights[None], out=np.zeros_like(accumulator), where=weights[None] > 0)
    return result, elapsed


def _tile_positions(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= tile_size:
        return [0]
    positions = list(range(0, length - tile_size + 1, stride))
    final = length - tile_size
    if positions[-1] != final:
        positions.append(final)
    return positions


def _predict_tile(model: torch.nn.Module, tile: np.ndarray, device: torch.device) -> np.ndarray:
    height, width = tile.shape
    pad_h, pad_w = (-height) % 8, (-width) % 8
    tensor = torch.from_numpy(tile.astype(np.float32))[None, None].to(device) * 2.0 - 1.0
    if pad_h or pad_w:
        mode = "reflect" if height > 1 and width > 1 else "replicate"
        tensor = torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h), mode=mode)
    prediction = model(tensor)
    return ((prediction[0, :, :height, :width] + 1.0) / 2.0).clamp(0, 1).cpu().numpy()


def infer_manifest(
    manifest_path: str | Path,
    checkpoint_path: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, object]:
    settings = get_settings()
    output = Path(output_dir or settings.output_dir)
    prediction_dir = output / "predictions"
    comparison_dir = output / "comparisons"
    metrics_dir = output / "metrics"
    for path in (prediction_dir, comparison_dir, metrics_dir):
        path.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(manifest_path)
    frame = frame[frame["split"] == "test"].reset_index(drop=True)
    if frame.empty:
        raise ValueError("No held-out test patches are available for evaluation.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_generator(checkpoint_path, device)
    rows, predictions, targets = [], [], []
    preview_items = []
    for _, row in frame.iterrows():
        thermal = np.load(row.thermal_path)
        target = np.load(row.rgb_path)
        valid = np.load(row.mask_path).astype(bool)
        prediction, elapsed = predict_array(model, thermal, device)
        metrics = image_metrics(prediction, target, valid)
        metrics.update({"patch_name": row.patch_name, "scene_name": row.scene_name, "inference_time_ms": elapsed})
        rows.append(metrics)
        np.save(prediction_dir / f"{row.patch_name}_generated.npy", prediction)
        Image.fromarray((np.moveaxis(prediction, 0, -1) * 255).astype(np.uint8)).save(
            prediction_dir / f"{row.patch_name}_generated.png"
        )
        Image.fromarray((thermal * 255).astype(np.uint8)).save(prediction_dir / f"{row.patch_name}_ir.png")
        Image.fromarray((np.moveaxis(target, 0, -1) * 255).astype(np.uint8)).save(prediction_dir / f"{row.patch_name}_real.png")
        plt.imsave(prediction_dir / f"{row.patch_name}_difference.png", np.mean(np.abs(prediction - target), axis=0), cmap="magma")
        predictions.append(torch.from_numpy(prediction)[None])
        targets.append(torch.from_numpy(target)[None])
        if len(preview_items) < 6:
            preview_items.append((thermal, target, prediction, np.mean(np.abs(prediction - target), axis=0)))

    summary = aggregate_metrics(rows)
    perceptual, warnings = optional_perceptual_metrics(predictions, targets, device)
    summary.update(perceptual)
    pd.DataFrame(rows).to_csv(metrics_dir / "test_metrics.csv", index=False)
    (metrics_dir / "test_metrics_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    comparison_path = comparison_dir / "comparison_grid.png"
    _save_comparison_grid(preview_items, comparison_path)
    history_path = settings.checkpoint_dir / "training_history.csv"
    if history_path.exists():
        history = pd.read_csv(history_path)
        axes = history.plot(x="epoch", y=[column for column in ("generator_loss", "discriminator_loss", "validation_l1") if column in history])
        axes.figure.tight_layout()
        axes.figure.savefig(comparison_dir / "training_curves.png", dpi=160)
        plt.close(axes.figure)
    report_path = generate_evaluation_pdf(summary, comparison_path, output / "reports" / "evaluation_report.pdf")
    return {"metrics": summary, "warnings": warnings, "patch_metrics": rows, "report": str(report_path)}


def _save_comparison_grid(items: list[tuple[np.ndarray, ...]], output_path: Path) -> None:
    fig, axes = plt.subplots(len(items), 4, figsize=(14, 3.5 * len(items)), squeeze=False)
    titles = ("Thermal IR", "Ground Truth RGB", "Generated RGB", "Absolute Difference")
    for row_axes, values in zip(axes, items):
        for axis, value, title in zip(row_axes, values, titles):
            display = np.moveaxis(value, 0, -1) if value.ndim == 3 and value.shape[0] in (3, 4) else value
            axis.imshow(display, cmap="inferno" if display.ndim == 2 else None)
            axis.set_title(title)
            axis.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def reconstruct_scene_mosaic(
    manifest_path: str | Path,
    prediction_dir: str | Path,
    scene_name: str,
    output_path: str | Path,
    boundary_path: str | Path | None = None,
) -> Path:
    frame = pd.read_csv(manifest_path)
    frame = frame[(frame["split"] == "test") & (frame["scene_name"] == scene_name)]
    if frame.empty:
        raise ValueError(f"No test patches found for scene {scene_name}.")
    first = frame.iloc[0]
    height, width = int(first.scene_height), int(first.scene_width)
    accumulator = np.zeros((3, height, width), dtype=np.float64)
    weights = np.zeros((height, width), dtype=np.float64)
    target_accumulator = np.zeros_like(accumulator)
    for _, row in frame.iterrows():
        prediction = np.load(Path(prediction_dir) / f"{row.patch_name}_generated.npy")
        target = np.load(row.rgb_path)
        valid = np.load(row.mask_path).astype(bool)
        y, x = int(row.y_offset), int(row.x_offset)
        ph, pw = valid.shape
        accumulator[:, y : y + ph, x : x + pw] += prediction * valid
        target_accumulator[:, y : y + ph, x : x + pw] += target * valid
        weights[y : y + ph, x : x + pw] += valid
    covered = weights > 0
    generated = np.divide(accumulator, weights[None], out=np.zeros_like(accumulator), where=covered[None]).astype(np.float32)
    target = np.divide(target_accumulator, weights[None], out=np.zeros_like(accumulator), where=covered[None]).astype(np.float32)
    transform = json.loads(first.scene_transform)
    export_rgb_geotiff(generated, output_path, first.crs, transform, covered)
    output_path = Path(output_path)
    export_rgb_geotiff(target, output_path.with_name("ground_truth_rgb_geotiff.tif"), first.crs, transform, covered)
    Image.fromarray((np.moveaxis(generated, 0, -1) * 255).astype(np.uint8)).save(output_path.with_suffix(".png"))
    difference = np.mean(np.abs(generated - target), axis=0)
    plt.imsave(output_path.with_name("difference_mosaic.png"), difference, cmap="magma")
    covered_y, covered_x = np.where(covered)
    if covered_y.size:
        margin = 32
        y0, y1 = max(0, int(covered_y.min()) - margin), min(height, int(covered_y.max()) + margin + 1)
        x0, x1 = max(0, int(covered_x.min()) - margin), min(width, int(covered_x.max()) + margin + 1)
        zoom = generated[:, y0:y1, x0:x1]
        Image.fromarray((np.moveaxis(zoom, 0, -1) * 255).astype(np.uint8)).save(output_path.with_name("zoomed_area.png"))
    if boundary_path:
        from app.services.geospatial_service import load_pune_boundary, save_boundary_overlay

        boundary = load_pune_boundary(boundary_path, first.crs)
        save_boundary_overlay(generated, boundary, output_path.with_name("pune_boundary_overlay.png"), first.crs, transform)
    return output_path
