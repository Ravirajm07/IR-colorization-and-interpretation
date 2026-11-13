from __future__ import annotations

from pathlib import Path

import torch

from app.models.generator import GlobalGenerator
from app.utils.config import get_settings


def load_generator(checkpoint_path: str | Path | None = None, device: str | torch.device | None = None) -> GlobalGenerator:
    settings = get_settings()
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    path = Path(checkpoint_path or settings.checkpoint_dir / "best_generator.pth")
    if not path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {path}")
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
        config = payload.get("config", {}) if isinstance(payload, dict) else {}
        model = GlobalGenerator(
            base_filters=int(config.get("base_filters", settings.generator_filters)),
            residual_blocks=int(config.get("residual_blocks", settings.generator_res_blocks)),
        )
        state = payload.get("generator", payload.get("model_state_dict", payload)) if isinstance(payload, dict) else payload
        model.load_state_dict(state)
    except (RuntimeError, KeyError, TypeError) as exc:
        raise RuntimeError(f"Corrupt or incompatible model checkpoint: {exc}") from exc
    return model.to(device).eval()
