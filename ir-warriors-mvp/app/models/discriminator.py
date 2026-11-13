from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class PatchDiscriminator(nn.Module):
    def __init__(self, input_channels: int = 4, base_filters: int = 64, layers: int = 4) -> None:
        super().__init__()
        blocks: list[nn.Module] = [
            nn.Sequential(nn.Conv2d(input_channels, base_filters, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True))
        ]
        channels = base_filters
        for index in range(1, layers):
            next_channels = min(base_filters * (2**index), 512)
            stride = 1 if index == layers - 1 else 2
            blocks.append(
                nn.Sequential(
                    nn.Conv2d(channels, next_channels, 4, stride, 1, bias=False),
                    nn.InstanceNorm2d(next_channels),
                    nn.LeakyReLU(0.2, inplace=True),
                )
            )
            channels = next_channels
        blocks.append(nn.Sequential(nn.Conv2d(channels, 1, 4, 1, 1)))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, infrared: torch.Tensor, rgb: torch.Tensor) -> list[torch.Tensor]:
        x = torch.cat((infrared, rgb), dim=1)
        features = []
        for block in self.blocks:
            x = block(x)
            features.append(x)
        return features


class MultiScaleDiscriminator(nn.Module):
    def __init__(self, scales: int = 2, **kwargs: int) -> None:
        super().__init__()
        if scales < 2:
            raise ValueError("Multi-scale discriminator requires at least two scales.")
        self.discriminators = nn.ModuleList(PatchDiscriminator(**kwargs) for _ in range(scales))

    def forward(self, infrared: torch.Tensor, rgb: torch.Tensor) -> list[list[torch.Tensor]]:
        outputs = []
        for index, discriminator in enumerate(self.discriminators):
            if index:
                infrared = F.avg_pool2d(infrared, 3, 2, 1, count_include_pad=False)
                rgb = F.avg_pool2d(rgb, 3, 2, 1, count_include_pad=False)
            outputs.append(discriminator(infrared, rgb))
        return outputs
