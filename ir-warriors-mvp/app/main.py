import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes_inference import router as inference_router
from app.api.routes_metrics import router as metrics_router
from app.api.routes_reports import router as reports_router
from app.api.routes_upload import router as upload_router
from app.schemas.responses import HealthResponse
from app.utils.config import get_settings
from app.utils.logging import configure_logging


configure_logging()
settings = get_settings()
app = FastAPI(
    title="IR Warriors API",
    description="Landsat 8/9 IR enhancement and RGB translation service",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
app.include_router(upload_router)
app.include_router(inference_router)
app.include_router(metrics_router)
app.include_router(reports_router)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="healthy", model_version=settings.model_version, cuda_available=torch.cuda.is_available())
