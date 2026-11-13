from __future__ import annotations

import re
import zipfile
from dataclasses import dataclass
from pathlib import Path


class SceneValidationError(ValueError):
    """Raised when an input cannot form a scientifically valid Landsat pair."""


LANDSAT_PATTERNS = {
    "thermal": re.compile(r"_ST_B10\.TIF$", re.IGNORECASE),
    "blue": re.compile(r"_SR_B2\.TIF$", re.IGNORECASE),
    "green": re.compile(r"_SR_B3\.TIF$", re.IGNORECASE),
    "red": re.compile(r"_SR_B4\.TIF$", re.IGNORECASE),
    "nir": re.compile(r"_SR_B5\.TIF$", re.IGNORECASE),
    "swir": re.compile(r"_SR_B6\.TIF$", re.IGNORECASE),
    "qa": re.compile(r"_QA_PIXEL\.TIF$", re.IGNORECASE),
    "mtl_txt": re.compile(r"_MTL\.txt$", re.IGNORECASE),
    "mtl_json": re.compile(r"_MTL\.json$", re.IGNORECASE),
}


@dataclass(frozen=True)
class SceneFiles:
    root: Path
    scene_id: str
    thermal: Path
    red: Path | None = None
    green: Path | None = None
    blue: Path | None = None
    qa: Path | None = None
    metadata: Path | None = None

    @property
    def has_rgb(self) -> bool:
        return all((self.red, self.green, self.blue))


def discover_scene(path: str | Path, require_rgb: bool = True, require_qa: bool = True) -> SceneFiles:
    root = Path(path)
    if root.is_file() and root.suffix.lower() in {".tif", ".tiff"}:
        return SceneFiles(root.parent, root.stem, root)
    if not root.is_dir():
        raise SceneValidationError(f"Scene path does not exist or is not a directory: {root}")

    found: dict[str, Path] = {}
    for item in root.rglob("*"):
        if not item.is_file():
            continue
        for name, pattern in LANDSAT_PATTERNS.items():
            if name not in found and pattern.search(item.name):
                found[name] = item

    if "thermal" not in found:
        raise SceneValidationError("ST_B10 thermal band not found.")
    missing_rgb = [name for name in ("red", "green", "blue") if name not in found]
    if require_rgb and missing_rgb:
        bands = {"red": "SR_B4", "green": "SR_B3", "blue": "SR_B2"}
        raise SceneValidationError(f"Required RGB bands not found: {', '.join(bands[x] for x in missing_rgb)}.")
    if require_qa and "qa" not in found:
        raise SceneValidationError("QA_PIXEL quality band not found.")

    scene_id = found["thermal"].name.upper().rsplit("_ST_B10", 1)[0]
    return SceneFiles(
        root=root,
        scene_id=scene_id,
        thermal=found["thermal"],
        red=found.get("red"),
        green=found.get("green"),
        blue=found.get("blue"),
        qa=found.get("qa"),
        metadata=found.get("mtl_json") or found.get("mtl_txt"),
    )


def safe_extract_zip(zip_path: str | Path, destination: str | Path) -> Path:
    zip_path, destination = Path(zip_path), Path(destination)
    if not zipfile.is_zipfile(zip_path):
        raise SceneValidationError("Invalid ZIP file.")
    destination.mkdir(parents=True, exist_ok=True)
    target_root = destination.resolve()
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if target_root != target and target_root not in target.parents:
                raise SceneValidationError("ZIP contains an unsafe path.")
        archive.extractall(destination)
    return destination


def validate_upload_name(filename: str) -> None:
    if not filename or Path(filename).name != filename:
        raise SceneValidationError("Invalid upload filename.")
    if Path(filename).suffix.lower() not in {".zip", ".tif", ".tiff", ".png"}:
        raise SceneValidationError("Supported files are Landsat ZIP, GeoTIFF, and PNG.")
