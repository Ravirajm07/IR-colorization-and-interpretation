import json

import numpy as np
import torch
from fastapi.testclient import TestClient

from app.main import app
from app.services.job_service import jobs


client = TestClient(app)


class ConstantGenerator(torch.nn.Module):
    def forward(self, infrared):
        return infrared.repeat(1, 3, 1, 1)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_inference_response(monkeypatch):
    job = jobs.create()
    processed = jobs.directory(job["job_id"]) / "processed"
    processed.mkdir()
    thermal = processed / "thermal.npy"
    mask = processed / "mask.npy"
    metadata = processed / "metadata.json"
    np.save(thermal, np.full((16, 16), 0.5, dtype=np.float32))
    np.save(mask, np.ones((16, 16), dtype=bool))
    metadata.write_text(json.dumps({"crs": None, "transform": [1, 0, 0, 0, 1, 0]}), encoding="utf-8")
    jobs.update(
        job["job_id"],
        status="preprocessed",
        output_paths={"thermal": str(thermal), "mask": str(mask), "metadata": str(metadata)},
    )
    monkeypatch.setattr("app.api.routes_inference.load_generator", lambda checkpoint, device: ConstantGenerator().to(device).eval())
    response = client.post("/predict", json={"job_id": job["job_id"], "compute_metrics": True})
    assert response.status_code == 200
    completed = client.get(f"/status/{job['job_id']}").json()
    assert completed["status"] == "complete"
    assert completed["metrics"]["inference_time_ms"] > 0
    assert any("Reference RGB is unavailable" in warning for warning in completed["warnings"])
