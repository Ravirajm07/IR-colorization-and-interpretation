from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from affine import Affine
from PIL import Image
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.windows import Window, from_bounds, transform as window_transform
from rasterio.warp import reproject

from app.utils.config import get_settings
from app.utils.validators import SceneFiles, SceneValidationError, discover_scene


REFLECTANCE_SCALE = 0.0000275
REFLECTANCE_OFFSET = -0.2
TEMPERATURE_SCALE = 0.00341802
TEMPERATURE_OFFSET = 149.0
QA_INVALID_BITS = (0, 1, 2, 3, 4, 5)  # fill, dilated cloud, cirrus, cloud, shadow, snow


@dataclass
class RasterMetadata:
    crs: str | None
    transform: tuple[float, float, float, float, float, float]
    width: int
    height: int
    resolution: tuple[float, float]
    nodata: float | None
    dtype: str


@dataclass
class PreprocessedScene:
    scene_id: str
    thermal: np.ndarray
    rgb: np.ndarray | None
    valid_mask: np.ndarray
    metadata: RasterMetadata
    source_files: SceneFiles


def scale_reflectance(dn: np.ndarray) -> np.ndarray:
    return dn.astype(np.float32) * REFLECTANCE_SCALE + REFLECTANCE_OFFSET


def scale_temperature(dn: np.ndarray) -> np.ndarray:
    return dn.astype(np.float32) * TEMPERATURE_SCALE + TEMPERATURE_OFFSET


def qa_valid_mask(qa: np.ndarray) -> np.ndarray:
    qa_uint = qa.astype(np.uint16, copy=False)
    invalid = np.zeros(qa_uint.shape, dtype=bool)
    for bit in QA_INVALID_BITS:
        invalid |= (qa_uint & (1 << bit)) != 0
    return ~invalid


def normalize_thermal(kelvin: np.ndarray, minimum: float, maximum: float) -> np.ndarray:
    if maximum <= minimum:
        raise ValueError("Thermal maximum must be greater than minimum.")
    return np.clip((kelvin - minimum) / (maximum - minimum), 0.0, 1.0).astype(np.float32)


def normalize_rgb(reflectance: np.ndarray, minimum: float, maximum: float, gamma: float) -> np.ndarray:
    if maximum <= minimum:
        raise ValueError("Reflectance maximum must be greater than minimum.")
    scaled = np.clip((reflectance - minimum) / (maximum - minimum), 0.0, 1.0)
    return np.power(scaled, gamma).astype(np.float32)


def _aligned_read(path: Path, reference: rasterio.io.DatasetReader, resampling: Resampling) -> np.ndarray:
    with rasterio.open(path) as source:
        if source.crs is None or reference.crs is None:
            raise SceneValidationError("Unsupported CRS: every GeoTIFF must define a CRS.")
        same_grid = (
            source.crs == reference.crs
            and source.transform.almost_equals(reference.transform)
            and source.width == reference.width
            and source.height == reference.height
        )
        if same_grid:
            return source.read(1)
        destination = np.empty((reference.height, reference.width), dtype=source.dtypes[0])
        reproject(
            source=source.read(1),
            destination=destination,
            src_transform=source.transform,
            src_crs=source.crs,
            src_nodata=source.nodata,
            dst_transform=reference.transform,
            dst_crs=reference.crs,
            dst_nodata=source.nodata,
            resampling=resampling,
        )
        return destination


def _metadata_from_dataset(dataset: rasterio.io.DatasetReader) -> RasterMetadata:
    return RasterMetadata(
        crs=dataset.crs.to_string() if dataset.crs else None,
        transform=tuple(dataset.transform)[:6],
        width=dataset.width,
        height=dataset.height,
        resolution=tuple(map(abs, dataset.res)),
        nodata=dataset.nodata,
        dtype=dataset.dtypes[0],
    )


def preprocess_landsat_scene(
    scene_path: str | Path,
    *,
    require_rgb: bool = True,
    apply_cloud_mask: bool = True,
    thermal_range: tuple[float, float] | None = None,
    reflectance_range: tuple[float, float] | None = None,
    gamma: float | None = None,
) -> PreprocessedScene:
    settings = get_settings()
    files = discover_scene(scene_path, require_rgb=require_rgb, require_qa=apply_cloud_mask)
    thermal_range = thermal_range or (settings.thermal_min_kelvin, settings.thermal_max_kelvin)
    reflectance_range = reflectance_range or (settings.reflectance_min, settings.reflectance_max)
    gamma = settings.gamma if gamma is None else gamma

    try:
        with rasterio.open(files.thermal) as reference:
            if reference.count != 1:
                raise SceneValidationError("ST_B10 must be a single-band GeoTIFF.")
            metadata = _metadata_from_dataset(reference)
            thermal_dn = reference.read(1)
            valid = np.ones(thermal_dn.shape, dtype=bool)
            if reference.nodata is not None:
                valid &= thermal_dn != reference.nodata
            valid &= thermal_dn > 0

            if apply_cloud_mask and files.qa:
                valid &= qa_valid_mask(_aligned_read(files.qa, reference, Resampling.nearest))

            rgb = None
            if files.has_rgb:
                channels = [
                    _aligned_read(files.red, reference, Resampling.bilinear),
                    _aligned_read(files.green, reference, Resampling.bilinear),
                    _aligned_read(files.blue, reference, Resampling.bilinear),
                ]
                for channel in channels:
                    valid &= channel > 0
                reflectance = np.stack([scale_reflectance(channel) for channel in channels])
                rgb = normalize_rgb(reflectance, *reflectance_range, gamma)

            thermal = normalize_thermal(scale_temperature(thermal_dn), *thermal_range)
            thermal[~valid] = 0.0
            if rgb is not None:
                rgb[:, ~valid] = 0.0
    except rasterio.errors.RasterioIOError as exc:
        raise SceneValidationError(f"Unreadable GeoTIFF: {exc}") from exc

    return PreprocessedScene(files.scene_id, thermal, rgb, valid, metadata, files)


def preprocess_png(path: str | Path) -> PreprocessedScene:
    image = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
    identity = tuple(Affine.identity())[:6]
    metadata = RasterMetadata(None, identity, image.shape[1], image.shape[0], (1.0, 1.0), None, "uint8")
    files = SceneFiles(Path(path).parent, Path(path).stem, Path(path))
    return PreprocessedScene(Path(path).stem, image, None, np.ones_like(image, dtype=bool), metadata, files)


def clip_scene_to_pune(scene: PreprocessedScene, boundary_path: str | Path) -> PreprocessedScene:
    if scene.metadata.crs is None:
        raise SceneValidationError("Pune clipping requires a georeferenced raster CRS.")
    from app.services.geospatial_service import load_pune_boundary

    boundary = load_pune_boundary(boundary_path, scene.metadata.crs)
    affine = Affine(*scene.metadata.transform)
    raw_window = from_bounds(*boundary.total_bounds, transform=affine)
    window = raw_window.round_offsets().round_lengths().intersection(Window(0, 0, scene.metadata.width, scene.metadata.height))
    row_start, row_stop = int(window.row_off), int(window.row_off + window.height)
    col_start, col_stop = int(window.col_off), int(window.col_off + window.width)
    if row_stop <= row_start or col_stop <= col_start:
        raise SceneValidationError("Pune boundary does not intersect the satellite scene.")
    cropped_transform = window_transform(window, affine)
    shape = (row_stop - row_start, col_stop - col_start)
    inside = geometry_mask(boundary.geometry, out_shape=shape, transform=cropped_transform, invert=True)
    thermal = scene.thermal[row_start:row_stop, col_start:col_stop].copy()
    valid = scene.valid_mask[row_start:row_stop, col_start:col_stop].copy() & inside
    thermal[~valid] = 0
    rgb = None
    if scene.rgb is not None:
        rgb = scene.rgb[:, row_start:row_stop, col_start:col_stop].copy()
        rgb[:, ~valid] = 0
    metadata = RasterMetadata(
        crs=scene.metadata.crs,
        transform=tuple(cropped_transform)[:6],
        width=shape[1],
        height=shape[0],
        resolution=scene.metadata.resolution,
        nodata=scene.metadata.nodata,
        dtype=scene.metadata.dtype,
    )
    return PreprocessedScene(scene.scene_id, thermal, rgb, valid, metadata, scene.source_files)


def scene_summary(scene: PreprocessedScene) -> dict[str, Any]:
    acquired = None
    parts = scene.scene_id.split("_")
    if len(parts) >= 4 and len(parts[3]) == 8:
        acquired = f"{parts[3][:4]}-{parts[3][4:6]}-{parts[3][6:]}"
    return {
        "satellite": "Landsat 8/9" if scene.scene_id.startswith(("LC08", "LC09")) else "Unknown",
        "scene_id": scene.scene_id,
        "acquisition_date": acquired,
        "thermal_band": scene.source_files.thermal.name,
        "rgb_bands": [p.name for p in (scene.source_files.red, scene.source_files.green, scene.source_files.blue) if p],
        "qa_band": scene.source_files.qa.name if scene.source_files.qa else None,
        "crs": scene.metadata.crs,
        "resolution": scene.metadata.resolution,
        "cloud_mask_available": scene.source_files.qa is not None,
        "valid_fraction": float(scene.valid_mask.mean()),
    }


def save_preprocessed_scene(scene: PreprocessedScene, output_dir: str | Path) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    thermal_path, mask_path = output / "thermal.npy", output / "valid_mask.npy"
    np.save(thermal_path, scene.thermal)
    np.save(mask_path, scene.valid_mask)
    paths = {"thermal": thermal_path, "mask": mask_path}
    if scene.rgb is not None:
        rgb_path = output / "rgb.npy"
        np.save(rgb_path, scene.rgb)
        paths["rgb"] = rgb_path
    metadata_path = output / "metadata.json"
    metadata_path.write_text(json.dumps({"scene_id": scene.scene_id, **asdict(scene.metadata)}, indent=2), encoding="utf-8")
    paths["metadata"] = metadata_path
    return paths
