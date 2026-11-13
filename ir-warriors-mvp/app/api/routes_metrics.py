from fastapi import APIRouter, HTTPException

from app.services.job_service import jobs


router = APIRouter(tags=["evaluation"])


@router.get("/metrics/{job_id}")
def metrics(job_id: str) -> dict:
    try:
        job = jobs.get(job_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    return {"job_id": job_id, "status": job["status"], "metrics": job.get("metrics", {}), "warnings": job.get("warnings", [])}
