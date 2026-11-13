from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture
def landsat_scene_factory(tmp_path):
    def create(name: str = "LC08_L2SP_147047_20240101_20240110_02_T1", size: int = 32) -> Path:
        scene = tmp_path / name
        scene.mkdir()
        profile = {
            "driver": "GTiff",
            "height": size,
            "width": size,
            "count": 1,
            "dtype": "uint16",
            "crs": "EPSG:32643",
            "transform": from_origin(700000, 2100000, 30, 30),
            "nodata": 0,
        }
        arrays = {
            "ST_B10": np.full((size, size), 45000, dtype=np.uint16),
            "SR_B2": np.full((size, size), 10000, dtype=np.uint16),
            "SR_B3": np.full((size, size), 11000, dtype=np.uint16),
            "SR_B4": np.full((size, size), 12000, dtype=np.uint16),
            "QA_PIXEL": np.zeros((size, size), dtype=np.uint16),
        }
        arrays["QA_PIXEL"][0, 0] = 1 << 3
        for band, data in arrays.items():
            with rasterio.open(scene / f"{name}_{band}.TIF", "w", **profile) as destination:
                destination.write(data, 1)
        (scene / f"{name}_MTL.txt").write_text("GROUP = LANDSAT_METADATA_FILE\nEND\n", encoding="ascii")
        return scene

    return create
