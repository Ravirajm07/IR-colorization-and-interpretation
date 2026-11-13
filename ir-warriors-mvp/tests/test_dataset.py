from pathlib import Path

from training.dataset import LandsatPatchDataset
from training.prepare_dataset import generate_patches


def test_patch_generation_uses_scene_level_splits(landsat_scene_factory, tmp_path):
    scenes = [landsat_scene_factory(f"LC08_L2SP_147047_2024010{i}_20240110_02_T1") for i in range(1, 4)]
    manifest = generate_patches(scenes, tmp_path / "patches", patch_size=16, stride=16, min_valid_fraction=0.5)
    train = LandsatPatchDataset(manifest, "train", augment=True)
    val = LandsatPatchDataset(manifest, "val")
    test = LandsatPatchDataset(manifest, "test")
    assert len(train) == len(val) == len(test) == 4
    sample = train[0]
    assert sample["ir"].shape == (1, 16, 16)
    assert sample["rgb"].shape == (3, 16, 16)
    assert sample["mask"].shape == (1, 16, 16)
    assert sample["ir"].min() >= -1 and sample["ir"].max() <= 1
    assert Path(manifest).exists()
