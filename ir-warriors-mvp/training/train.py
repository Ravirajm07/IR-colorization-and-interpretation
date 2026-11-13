from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image

from app.models.discriminator import MultiScaleDiscriminator
from app.models.generator import GlobalGenerator, initialize_weights
from app.models.losses import LSGANLoss, VGGPerceptualLoss, feature_matching_loss, masked_l1
from app.utils.config import get_settings
from training.dataset import LandsatPatchDataset
from training.validate import validate_generator


@dataclass
class TrainConfig:
    manifest: str
    epochs: int = 100
    batch_size: int = 4
    workers: int = 2
    learning_rate: float = 0.0002
    beta1: float = 0.5
    beta2: float = 0.999
    lambda_l1: float = 100.0
    lambda_feature: float = 10.0
    lambda_vgg: float = 5.0
    base_filters: int = 64
    residual_blocks: int = 6
    discriminator_filters: int = 64
    discriminator_layers: int = 4
    mixed_precision: bool = True
    early_stopping_patience: int = 0
    resume: str | None = None


def _scheduler(optimizer: torch.optim.Optimizer, epochs: int) -> torch.optim.lr_scheduler.LambdaLR:
    midpoint = max(1, epochs // 2)
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda epoch: 1.0 if epoch < midpoint else max(0.0, 1.0 - (epoch - midpoint) / max(1, epochs - midpoint))
    )


def train(config: TrainConfig, output_dir: str | Path | None = None) -> dict[str, object]:
    settings = get_settings()
    checkpoint_dir = Path(output_dir or settings.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = config.mixed_precision and device.type == "cuda"
    train_set = LandsatPatchDataset(config.manifest, "train", augment=True)
    val_set = LandsatPatchDataset(config.manifest, "val", augment=False)
    train_loader = DataLoader(train_set, config.batch_size, shuffle=True, num_workers=config.workers, pin_memory=use_amp)
    val_loader = DataLoader(val_set, config.batch_size, shuffle=False, num_workers=config.workers, pin_memory=use_amp)

    generator = GlobalGenerator(base_filters=config.base_filters, residual_blocks=config.residual_blocks).to(device)
    discriminator = MultiScaleDiscriminator(
        scales=2, base_filters=config.discriminator_filters, layers=config.discriminator_layers
    ).to(device)
    generator.apply(initialize_weights)
    discriminator.apply(initialize_weights)
    optimizer_g = torch.optim.Adam(generator.parameters(), config.learning_rate, betas=(config.beta1, config.beta2))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), config.learning_rate, betas=(config.beta1, config.beta2))
    scheduler_g, scheduler_d = _scheduler(optimizer_g, config.epochs), _scheduler(optimizer_d, config.epochs)
    scaler_g = torch.amp.GradScaler("cuda", enabled=use_amp)
    scaler_d = torch.amp.GradScaler("cuda", enabled=use_amp)
    adversarial = LSGANLoss()
    perceptual = VGGPerceptualLoss(pretrained=True).to(device) if config.lambda_vgg > 0 else None
    writer = SummaryWriter(checkpoint_dir / "tensorboard")
    start_epoch, best_validation, history = 0, float("inf"), []

    if config.resume:
        payload = torch.load(config.resume, map_location=device, weights_only=False)
        generator.load_state_dict(payload["generator"])
        discriminator.load_state_dict(payload["discriminator"])
        optimizer_g.load_state_dict(payload["optimizer_g"])
        optimizer_d.load_state_dict(payload["optimizer_d"])
        scheduler_g.load_state_dict(payload["scheduler_g"])
        scheduler_d.load_state_dict(payload["scheduler_d"])
        start_epoch = int(payload["epoch"]) + 1
        best_validation = float(payload["best_validation"])
        history = list(payload.get("history", []))

    stale_epochs = 0
    for epoch in range(start_epoch, config.epochs):
        generator.train()
        discriminator.train()
        sums = {"generator": 0.0, "discriminator": 0.0, "l1": 0.0}
        started = time.perf_counter()
        for batch in train_loader:
            infrared = batch["ir"].to(device, non_blocking=True)
            target = batch["rgb"].to(device, non_blocking=True)
            valid = batch["mask"].to(device, non_blocking=True)

            optimizer_d.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                generated = generator(infrared)
                real_outputs = discriminator(infrared, target)
                fake_outputs = discriminator(infrared, generated.detach())
                loss_d = 0.5 * (adversarial(real_outputs, True) + adversarial(fake_outputs, False))
            scaler_d.scale(loss_d).backward()
            scaler_d.step(optimizer_d)
            scaler_d.update()

            optimizer_g.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                fake_outputs = discriminator(infrared, generated)
                with torch.no_grad():
                    real_outputs = discriminator(infrared, target)
                loss_l1 = masked_l1(generated, target, valid)
                loss_g = (
                    adversarial(fake_outputs, True)
                    + config.lambda_l1 * loss_l1
                    + config.lambda_feature * feature_matching_loss(fake_outputs, real_outputs)
                    + (config.lambda_vgg * perceptual(generated, target) if perceptual is not None else generated.new_zeros(()))
                )
            scaler_g.scale(loss_g).backward()
            scaler_g.step(optimizer_g)
            scaler_g.update()
            sums["generator"] += float(loss_g.item())
            sums["discriminator"] += float(loss_d.item())
            sums["l1"] += float(loss_l1.item())

        batches = max(1, len(train_loader))
        validation = validate_generator(generator, val_loader, device)
        scheduler_g.step()
        scheduler_d.step()
        row = {
            "epoch": epoch + 1,
            "generator_loss": sums["generator"] / batches,
            "discriminator_loss": sums["discriminator"] / batches,
            "train_l1": sums["l1"] / batches,
            "validation_l1": validation,
            "learning_rate": optimizer_g.param_groups[0]["lr"],
            "epoch_seconds": time.perf_counter() - started,
        }
        history.append(row)
        for key, value in row.items():
            if key not in {"epoch", "epoch_seconds"}:
                writer.add_scalar(key, value, epoch + 1)
        sample = torch.cat((infrared[:2].repeat(1, 3, 1, 1), target[:2], generated[:2]), dim=0)
        save_image((sample + 1) / 2, checkpoint_dir / f"sample_epoch_{epoch + 1:04d}.png", nrow=2)

        checkpoint = {
            "epoch": epoch,
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
            "optimizer_g": optimizer_g.state_dict(),
            "optimizer_d": optimizer_d.state_dict(),
            "scheduler_g": scheduler_g.state_dict(),
            "scheduler_d": scheduler_d.state_dict(),
            "best_validation": min(best_validation, validation),
            "history": history,
            "config": asdict(config),
        }
        torch.save(checkpoint, checkpoint_dir / "latest_checkpoint.pth")
        if validation < best_validation:
            best_validation = validation
            stale_epochs = 0
            torch.save({"generator": generator.state_dict(), "config": asdict(config)}, checkpoint_dir / "best_generator.pth")
            torch.save({"discriminator": discriminator.state_dict(), "config": asdict(config)}, checkpoint_dir / "best_discriminator.pth")
        else:
            stale_epochs += 1
        pd.DataFrame(history).to_csv(checkpoint_dir / "training_history.csv", index=False)
        (checkpoint_dir / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
        if config.early_stopping_patience and stale_epochs >= config.early_stopping_patience:
            break
    writer.close()
    return {"best_validation_l1": best_validation, "epochs_completed": len(history), "device": str(device)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the IR-to-RGB generator.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--resume")
    parser.add_argument("--output")
    args = parser.parse_args()
    print(train(TrainConfig(args.manifest, args.epochs, args.batch_size, args.workers, resume=args.resume), args.output))


if __name__ == "__main__":
    main()
