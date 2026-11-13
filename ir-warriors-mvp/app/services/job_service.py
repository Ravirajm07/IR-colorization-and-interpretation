from __future__ import annotations

import json
import threading
import uuid
from pathlib import Path
from typing import Any

from app.utils.config import get_settings


class JobStore:
    def __init__(self) -> None:
        self.root = get_settings().temp_dir / "jobs"
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def create(self) -> dict[str, Any]:
        job_id = uuid.uuid4().hex
        job = {"job_id": job_id, "status": "created", "input_metadata": {}, "output_paths": {}, "metrics": {}, "warnings": []}
        (self.root / job_id).mkdir(parents=True)
        self.save(job)
        return job

    def directory(self, job_id: str) -> Path:
        path = (self.root / job_id).resolve()
        if self.root.resolve() not in path.parents:
            raise KeyError("Invalid job ID.")
        return path

    def save(self, job: dict[str, Any]) -> None:
        with self._lock:
            path = self.directory(job["job_id"]) / "job.json"
            path.write_text(json.dumps(job, indent=2, default=str), encoding="utf-8")

    def get(self, job_id: str) -> dict[str, Any]:
        path = self.directory(job_id) / "job.json"
        if not path.exists():
            raise KeyError(f"Job not found: {job_id}")
        return json.loads(path.read_text(encoding="utf-8"))

    def update(self, job_id: str, **values: Any) -> dict[str, Any]:
        job = self.get(job_id)
        job.update(values)
        self.save(job)
        return job


jobs = JobStore()
