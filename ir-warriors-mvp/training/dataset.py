from __future__ import annotations

from pathlib import Path

import albumentations as A
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class LandsatPatchDataset(Dataset):
    def __init__(self, manifest: str | Path, split: str, augment: bool = False) -> None:
        frame = pd.read_csv(manifest)
        required = {"patch_name", "scene_name", "split", "thermal_path", "rgb_path", "mask_path"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Manifest is missing columns: {', '.join(sorted(missing))}")
        self.rows = frame[frame["split"] == split].reset_index(drop=True)
        if self.rows.empty:
            raise ValueError(f"No patches found for split '{split}'.")
        self.augmentation = (
            A.Compose([A.HorizontalFlip(0.5), A.VerticalFlip(0.5), A.RandomRotate90(0.5)]) if augment else None
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        row = self.rows.iloc[index]
        infrared = np.load(row.thermal_path).astype(np.float32)
        rgb = np.load(row.rgb_path).astype(np.float32)
        mask = np.load(row.mask_path).astype(np.float32)
        if infrared.ndim != 2 or rgb.shape != (3, *infrared.shape) or mask.shape != infrared.shape:
            raise ValueError(f"Invalid paired patch dimensions for {row.patch_name}.")
        if self.augmentation:
            paired = np.concatenate((infrared[..., None], np.moveaxis(rgb, 0, -1)), axis=-1)
            transformed = self.augmentation(image=paired, mask=mask)
            paired, mask = transformed["image"], transformed["mask"]
            infrared, rgb = paired[..., 0], np.moveaxis(paired[..., 1:], -1, 0)
        return {
            "ir": torch.from_numpy(np.ascontiguousarray(infrared[None] * 2.0 - 1.0)),
            "rgb": torch.from_numpy(np.ascontiguousarray(rgb * 2.0 - 1.0)),
            "mask": torch.from_numpy(np.ascontiguousarray(mask[None])),
            "name": str(row.patch_name),
            "scene": str(row.scene_name),
        }
