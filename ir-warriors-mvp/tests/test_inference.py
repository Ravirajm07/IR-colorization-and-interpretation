from pathlib import Path

import numpy as np
import torch

from app.models.discriminator import MultiScaleDiscriminator
from app.models.generator import GlobalGenerator
from app.services.inference_service import _save_comparison_grid, predict_array
from app.services.metrics_service import image_metrics


def test_model_forward_shapes():
    generator = GlobalGenerator(base_filters=8, residual_blocks=1)
    discriminator = MultiScaleDiscriminator(scales=2, base_filters=8, layers=3)
    infrared = torch.randn(2, 1, 64, 64)
    generated = generator(infrared)
    outputs = discriminator(infrared, generated)
    assert generated.shape == (2, 3, 64, 64)
    assert len(outputs) == 2
    assert all(scale[-1].shape[1] == 1 for scale in outputs)


def test_prediction_and_metrics_are_computed():
    model = GlobalGenerator(base_filters=8, residual_blocks=1).eval()
    prediction, elapsed = predict_array(model, np.full((33, 35), 0.5, dtype=np.float32), torch.device("cpu"))
    target = np.zeros_like(prediction)
    metrics = image_metrics(prediction, target, np.ones((33, 35), dtype=bool))
    assert prediction.shape == (3, 33, 35)
    assert elapsed > 0
    assert set(metrics) == {"psnr", "ssim", "mae", "rmse"}


def test_comparison_grid_accepts_channel_first_rgb(tmp_path):
    output = tmp_path / "comparison.png"
    item = (
        np.zeros((16, 16), dtype=np.float32),
        np.zeros((3, 16, 16), dtype=np.float32),
        np.zeros((3, 16, 16), dtype=np.float32),
        np.zeros((16, 16), dtype=np.float32),
    )
    _save_comparison_grid([item], output)
    assert Path(output).stat().st_size > 0


def test_tiled_prediction_stitches_without_seams():
    class RepeatGenerator(torch.nn.Module):
        def forward(self, infrared):
            return infrared.repeat(1, 3, 1, 1)

    thermal = np.random.default_rng(7).random((301, 333), dtype=np.float32)
    prediction, _ = predict_array(RepeatGenerator(), thermal, torch.device("cpu"), tile_size=128, overlap=32)
    np.testing.assert_allclose(prediction, np.repeat(thermal[None], 3, axis=0), atol=1e-6)
