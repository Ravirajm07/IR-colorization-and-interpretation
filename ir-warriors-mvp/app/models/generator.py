from __future__ import annotations

import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3, bias=False),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3, bias=False),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class GlobalGenerator(nn.Module):
    """Pix2PixHD-inspired residual generator for 1-channel IR to 3-channel RGB."""

    def __init__(
        self,
        input_channels: int = 1,
        output_channels: int = 3,
        base_filters: int = 64,
        downsample_layers: int = 3,
        residual_blocks: int = 6,
        conditioning_channels: int = 0,
    ) -> None:
        super().__init__()
        in_channels = input_channels + conditioning_channels
        layers: list[nn.Module] = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, base_filters, 7, bias=False),
            nn.InstanceNorm2d(base_filters),
            nn.ReLU(inplace=True),
        ]
        channels = base_filters
        for _ in range(downsample_layers):
            layers.extend(
                [
                    nn.Conv2d(channels, channels * 2, 3, stride=2, padding=1, bias=False),
                    nn.InstanceNorm2d(channels * 2),
                    nn.ReLU(inplace=True),
                ]
            )
            channels *= 2
        layers.extend(ResidualBlock(channels) for _ in range(residual_blocks))
        for _ in range(downsample_layers):
            layers.extend(
                [
                    nn.ConvTranspose2d(channels, channels // 2, 3, stride=2, padding=1, output_padding=1, bias=False),
                    nn.InstanceNorm2d(channels // 2),
                    nn.ReLU(inplace=True),
                ]
            )
            channels //= 2
        layers.extend([nn.ReflectionPad2d(3), nn.Conv2d(channels, output_channels, 7), nn.Tanh()])
        self.network = nn.Sequential(*layers)

    def forward(self, infrared: torch.Tensor, conditioning: torch.Tensor | None = None) -> torch.Tensor:
        if conditioning is not None:
            infrared = torch.cat((infrared, conditioning), dim=1)
        return self.network(infrared)


def initialize_weights(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(module.weight, 0.0, 0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
