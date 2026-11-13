from typing import Any

from pydantic import BaseModel, Field


class JobResponse(BaseModel):
    job_id: str
    status: str
    input_metadata: dict[str, Any] = Field(default_factory=dict)
    output_paths: dict[str, str] = Field(default_factory=dict)
    metrics: dict[str, float | str | None] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    model_version: str
    cuda_available: bool
