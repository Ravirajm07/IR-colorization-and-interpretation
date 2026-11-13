from __future__ import annotations

from pathlib import Path
from typing import Iterable

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from affine import Affine
from rasterio.mask import mask as raster_mask


DISTRICT_FIELDS = ("district", "DISTRICT", "DIST_NAME", "NAME", "dtname")


def load_pune_boundary(path: str | Path, target_crs: str | None = None) -> gpd.GeoDataFrame:
    frame = gpd.read_file(path)
    field = next((name for name in DISTRICT_FIELDS if name in frame.columns), None)
    if field is None:
        raise ValueError(f"District field not found. Expected one of: {', '.join(DISTRICT_FIELDS)}")
    pune = frame[frame[field].astype(str).str.strip().str.casefold() == "pune"].copy()
    if pune.empty:
        raise ValueError("Pune district was not found in the boundary file.")
    if target_crs:
        if pune.crs is None:
            raise ValueError("District boundary has no CRS.")
        pune = pune.to_crs(target_crs)
    return pune


def clip_raster_to_boundary(raster_path: str | Path, boundary: gpd.GeoDataFrame, output_path: str | Path) -> Path:
    with rasterio.open(raster_path) as source:
        projected = boundary.to_crs(source.crs)
        data, transform = raster_mask(source, projected.geometry, crop=True, nodata=source.nodata)
        profile = source.profile.copy()
        profile.update(height=data.shape[1], width=data.shape[2], transform=transform)
        with rasterio.open(output_path, "w", **profile) as destination:
            destination.write(data)
    return Path(output_path)


def export_rgb_geotiff(
    rgb: np.ndarray,
    output_path: str | Path,
    crs: str,
    transform: Iterable[float] | Affine,
    valid_mask: np.ndarray | None = None,
) -> Path:
    if rgb.ndim != 3 or rgb.shape[0] != 3:
        raise ValueError("RGB output must have shape [3, H, W].")
    affine = transform if isinstance(transform, Affine) else Affine(*list(transform)[:6])
    data = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    if valid_mask is not None:
        data[:, ~valid_mask] = 0
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[1],
        width=data.shape[2],
        count=3,
        dtype="uint8",
        crs=crs,
        transform=affine,
        nodata=0,
        compress="deflate",
        photometric="RGB",
    ) as destination:
        destination.write(data)
        destination.set_band_description(1, "Red")
        destination.set_band_description(2, "Green")
        destination.set_band_description(3, "Blue")
    return path


def save_boundary_overlay(rgb: np.ndarray, boundary: gpd.GeoDataFrame, output_path: str | Path, crs: str, transform: Iterable[float]) -> Path:
    affine = Affine(*list(transform)[:6])
    height, width = rgb.shape[1:]
    left, top = affine * (0, 0)
    right, bottom = affine * (width, height)
    projected = boundary.to_crs(crs)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(np.moveaxis(np.clip(rgb, 0, 1), 0, -1), extent=[left, right, bottom, top])
    projected.boundary.plot(ax=ax, color="#00e5ff", linewidth=1.5)
    ax.set_title("Generated RGB with Pune District Boundary")
    ax.set_xlabel("Easting")
    ax.set_ylabel("Northing")
    fig.tight_layout()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path
