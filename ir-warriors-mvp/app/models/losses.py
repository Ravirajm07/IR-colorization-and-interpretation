from __future__ import annotations

import warnings

import torch
from torch import nn
from torchvision.models import VGG19_Weights, vgg19


class LSGANLoss(nn.Module):
    def forward(self, predictions: list[list[torch.Tensor]], is_real: bool) -> torch.Tensor:
        losses = []
        target_value = 1.0 if is_real else 0.0
        for scale in predictions:
            losses.append(torch.mean((scale[-1] - target_value) ** 2))
        return torch.stack(losses).mean()


def feature_matching_loss(
    fake: list[list[torch.Tensor]], real: list[list[torch.Tensor]], mask: torch.Tensor | None = None
) -> torch.Tensor:
    del mask  # feature maps have scale-dependent support; input validity is handled by masked reconstruction.
    losses = []
    for fake_scale, real_scale in zip(fake, real):
        losses.extend(torch.mean(torch.abs(a - b.detach())) for a, b in zip(fake_scale[:-1], real_scale[:-1]))
    return torch.stack(losses).mean()


def masked_l1(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    expanded = mask.expand_as(prediction)
    denominator = expanded.sum().clamp_min(1.0)
    return (torch.abs(prediction - target) * expanded).sum() / denominator


class VGGPerceptualLoss(nn.Module):
    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = VGG19_Weights.IMAGENET1K_V1 if pretrained else None
        try:
            features = vgg19(weights=weights).features
        except Exception as exc:
            warnings.warn(f"Pretrained VGG19 unavailable ({exc}); VGG loss is disabled.", stacklevel=2)
            self.enabled = False
            self.blocks = nn.ModuleList()
            return
        self.enabled = True
        self.blocks = nn.ModuleList(
            [features[:4].eval(), features[4:9].eval(), features[9:18].eval(), features[18:27].eval()]
        )
        for parameter in self.parameters():
            parameter.requires_grad = False
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return prediction.new_zeros(())
        x = ((prediction + 1.0) / 2.0 - self.mean) / self.std
        y = ((target + 1.0) / 2.0 - self.mean) / self.std
        loss = prediction.new_zeros(())
        for block in self.blocks:
            x, y = block(x), block(y)
            loss = loss + torch.mean(torch.abs(x - y.detach()))
        return loss
