from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.schemas.responses import JobResponse
from app.services.job_service import jobs
from app.utils.config import get_settings
from app.utils.validators import SceneValidationError, discover_scene, safe_extract_zip, validate_upload_name


router = APIRouter(tags=["data"])


@router.post("/upload-scene", response_model=JobResponse)
async def upload_scene(file: UploadFile = File(...)) -> dict:
    settings = get_settings()
    try:
        validate_upload_name(file.filename or "")
    except SceneValidationError as exc:
        raise HTTPException(400, str(exc)) from exc
    job = jobs.create()
    upload_dir = jobs.directory(job["job_id"]) / "upload"
    upload_dir.mkdir()
    destination = upload_dir / Path(file.filename).name
    written = 0
    try:
        with destination.open("wb") as output:
            while chunk := await file.read(1024 * 1024):
                written += len(chunk)
                if written > settings.max_upload_mb * 1024 * 1024:
                    raise HTTPException(413, f"Upload exceeds {settings.max_upload_mb} MB.")
                output.write(chunk)
        source_path = destination
        if destination.suffix.lower() == ".zip":
            source_path = safe_extract_zip(destination, upload_dir / "scene")
            scene = discover_scene(source_path)
            metadata = {"scene_id": scene.scene_id, "thermal_band": scene.thermal.name, "qa_band": scene.qa.name if scene.qa else None}
        elif destination.suffix.lower() in {".tif", ".tiff"}:
            scene = discover_scene(destination, require_rgb=False, require_qa=False)
            metadata = {"scene_id": scene.scene_id, "thermal_band": scene.thermal.name, "input_type": "GeoTIFF"}
        else:
            metadata = {"scene_id": destination.stem, "input_type": "PNG quick demo"}
        return jobs.update(job["job_id"], status="uploaded", source_path=str(source_path), input_metadata=metadata)
    except HTTPException:
        shutil.rmtree(jobs.directory(job["job_id"]), ignore_errors=True)
        raise
    except (SceneValidationError, OSError) as exc:
        jobs.update(job["job_id"], status="failed", warnings=[str(exc)])
        raise HTTPException(400, str(exc)) from exc
