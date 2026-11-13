from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.schemas.requests import ReportRequest
from app.services.job_service import jobs
from app.services.report_service import create_model_bundle, generate_pdf_report


router = APIRouter(tags=["reports"])


@router.post("/generate-report")
def generate_report(request: ReportRequest) -> dict:
    try:
        job = jobs.get(request.job_id)
        report = generate_pdf_report(job, jobs.directory(request.job_id) / "outputs" / "evaluation_report.pdf")
        paths = job.get("output_paths", {})
        paths["report"] = str(report)
        jobs.update(request.job_id, output_paths=paths)
        return {"job_id": request.job_id, "report_path": str(report)}
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.get("/download/{job_id}/{file_type}")
def download(job_id: str, file_type: str) -> FileResponse:
    try:
        job = jobs.get(job_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    if file_type == "model_zip":
        try:
            path = create_model_bundle(jobs.directory(job_id) / "outputs" / "model.zip")
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
    elif file_type == "configuration":
        path = jobs.directory(job_id) / "job.json"
    else:
        path_value = job.get("output_paths", {}).get(file_type)
        if not path_value:
            raise HTTPException(404, f"Artifact '{file_type}' is unavailable for this job.")
        path = Path(path_value).resolve()
        if jobs.directory(job_id).resolve() not in path.parents:
            raise HTTPException(403, "Artifact path is outside the job directory.")
    if not path.exists():
        raise HTTPException(404, "Artifact file no longer exists.")
    return FileResponse(path, filename=path.name)
