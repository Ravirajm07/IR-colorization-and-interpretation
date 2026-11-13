from pydantic import BaseModel, Field


class PreprocessRequest(BaseModel):
    job_id: str
    clip_to_pune: bool = False
    boundary_path: str | None = None
    cloud_mask: bool = True
    patch_size: int = Field(256, ge=64, le=1024)
    stride: int = Field(192, ge=32, le=1024)
    min_valid_fraction: float = Field(0.75, ge=0.0, le=1.0)
    thermal_min_kelvin: float = 250.0
    thermal_max_kelvin: float = 330.0
    gamma: float = Field(1.0 / 2.2, gt=0.0, le=2.0)


class PredictRequest(BaseModel):
    job_id: str
    checkpoint_path: str | None = None
    compute_metrics: bool = True


class ReportRequest(BaseModel):
    job_id: str
