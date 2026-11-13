import argparse
from pathlib import Path

import torch

from app.models.model_loader import load_generator


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the generator as TorchScript.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="checkpoints/generator.ts")
    args = parser.parse_args()
    device = torch.device("cpu")
    model = load_generator(args.checkpoint, device)
    traced = torch.jit.trace(model, torch.zeros(1, 1, 256, 256))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    traced.save(args.output)


if __name__ == "__main__":
    main()
