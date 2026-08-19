from __future__ import annotations

import argparse
from pathlib import Path

from nhanes_semantic.config import load_config
from nhanes_semantic.pipeline import STAGES, run_stage
from nhanes_semantic.utils import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the GPU-first NHANES semantic regression experiment."
    )
    parser.add_argument(
        "--config", type=Path, default=Path("config/default.yaml"), help="YAML configuration"
    )
    parser.add_argument(
        "--stage", choices=["all", *STAGES], default="all", help="Restartable pipeline stage"
    )
    parser.add_argument(
        "--force", action="store_true", help="Rebuild outputs for the selected stage(s)"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    log_path = Path(config["_repo_root"]) / config["project"]["output_dir"] / "logs" / "run.log"
    setup_logging(log_path)
    run_stage(config, args.stage, force=args.force)


if __name__ == "__main__":
    main()

