from __future__ import annotations

import torch

from app.models.losses import masked_l1


@torch.inference_mode()
def validate_generator(model: torch.nn.Module, loader: torch.utils.data.DataLoader, device: torch.device) -> float:
    model.eval()
    total, batches = 0.0, 0
    for batch in loader:
        infrared = batch["ir"].to(device)
        target = batch["rgb"].to(device)
        mask = batch["mask"].to(device)
        total += float(masked_l1(model(infrared), target, mask).item())
        batches += 1
    return total / max(batches, 1)
