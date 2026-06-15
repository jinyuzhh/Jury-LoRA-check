"""Command-line entry point for Jury-LoRA federated fine-tuning."""

import argparse
from pathlib import Path
from typing import Sequence

import yaml

from src.utils.config import load_config
from src.utils.seed import set_seed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Initialize a Jury-LoRA federated fine-tuning run."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the YAML experiment configuration.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Load configuration and dispatch the selected federated method."""
    args = parse_args(argv)
    config = load_config(args.config)

    set_seed(config["seed"])

    output_dir = Path(config["output"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loaded configuration:")
    print(yaml.safe_dump(config, sort_keys=False).strip())
    print(f"Output directory: {output_dir}")

    method = config.get("method")
    if method == "fedavg_lora":
        from src.federated.trainer import run_fedavg_training

        run_fedavg_training(args.config)
    else:
        raise ValueError(f"Unsupported federated method: {method!r}.")


if __name__ == "__main__":
    main()
