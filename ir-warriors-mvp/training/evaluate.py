import argparse
import json

from app.services.inference_service import infer_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint on held-out test patches.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--output")
    args = parser.parse_args()
    print(json.dumps(infer_manifest(args.manifest, args.checkpoint, args.output), indent=2, default=str))


if __name__ == "__main__":
    main()
