from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from affine import Affine

from app.services.preprocessing_service import preprocess_landsat_scene
from app.utils.config import get_settings


def assign_scene_splits(scene_names: list[str]) -> dict[str, str]:
    names = sorted(set(scene_names))
    if len(names) < 3:
        raise ValueError("At least three scenes are required for leakage-free train/validation/test splitting.")
    if len(names) == 3:
        counts = (1, 1, 1)
    else:
        val_count = max(1, round(len(names) * 0.2))
        test_count = max(1, round(len(names) * 0.2))
        counts = (len(names) - val_count - test_count, val_count, test_count)
    train_count, val_count, _ = counts
    return {
        name: "train" if i < train_count else "val" if i < train_count + val_count else "test"
        for i, name in enumerate(names)
    }


def generate_patches(
    scene_paths: list[str | Path],
    output_dir: str | Path,
    patch_size: int = 256,
    stride: int = 192,
    min_valid_fraction: float = 0.75,
) -> Path:
    if stride <= 0 or patch_size <= 0:
        raise ValueError("Patch size and stride must be positive.")
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    splits = assign_scene_splits([Path(path).name for path in scene_paths])
    records: list[dict[str, object]] = []
    for scene_path in scene_paths:
        scene = preprocess_landsat_scene(scene_path)
        split = splits[Path(scene_path).name]
        scene_output = output / split / scene.scene_id
        scene_output.mkdir(parents=True, exist_ok=True)
        height, width = scene.thermal.shape
        if height < patch_size or width < patch_size:
            continue
        for y in range(0, height - patch_size + 1, stride):
            for x in range(0, width - patch_size + 1, stride):
                window = np.s_[y : y + patch_size, x : x + patch_size]
                valid = scene.valid_mask[window]
                valid_fraction = float(valid.mean())
                if valid_fraction < min_valid_fraction:
                    continue
                name = f"{scene.scene_id}_y{y:06d}_x{x:06d}"
                thermal_path = scene_output / f"{name}_ir.npy"
                rgb_path = scene_output / f"{name}_rgb.npy"
                mask_path = scene_output / f"{name}_mask.npy"
                np.save(thermal_path, scene.thermal[window])
                np.save(rgb_path, scene.rgb[:, window[0], window[1]])
                np.save(mask_path, valid)
                patch_transform = tuple(Affine(*scene.metadata.transform) * Affine.translation(x, y))[:6]
                records.append(
                    {
                        "patch_name": name,
                        "scene_name": scene.scene_id,
                        "split": split,
                        "x_offset": x,
                        "y_offset": y,
                        "valid_fraction": valid_fraction,
                        "thermal_path": str(thermal_path),
                        "rgb_path": str(rgb_path),
                        "mask_path": str(mask_path),
                        "crs": scene.metadata.crs,
                        "transform": json.dumps(patch_transform),
                        "scene_transform": json.dumps(scene.metadata.transform),
                        "scene_width": width,
                        "scene_height": height,
                    }
                )
    if not records:
        raise ValueError("Training could not start because no valid paired patches were created.")
    manifest = output / "manifest.csv"
    pd.DataFrame(records).to_csv(manifest, index=False)
    return manifest


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Prepare aligned Landsat patches.")
    parser.add_argument("scenes", nargs="+", help="Extracted Landsat scene directories")
    parser.add_argument("--output", default=str(settings.data_dir / "patches"))
    parser.add_argument("--patch-size", type=int, default=settings.patch_size)
    parser.add_argument("--stride", type=int, default=settings.stride)
    parser.add_argument("--min-valid-fraction", type=float, default=settings.min_valid_fraction)
    args = parser.parse_args()
    print(generate_patches(args.scenes, args.output, args.patch_size, args.stride, args.min_valid_fraction))


if __name__ == "__main__":
    main()
