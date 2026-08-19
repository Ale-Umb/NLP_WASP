from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .config import ensure_project_directories, public_config
from .embeddings import generate_embeddings
from .evaluation import evaluate_all
from .nhanes import download_nhanes, prepare_nhanes
from .preprocess import preprocess_nhanes
from .reporting import generate_report
from .task_factory import build_tasks
from .training import select_operator_rank, train_all_embeddings
from .utils import environment_summary, get_logger, write_json
from .wiki_adaptation import train_wikipedia_adapted_encoder


STAGES = [
    "download",
    "prepare",
    "tasks",
    "adapt",
    "embed",
    "rank",
    "train",
    "evaluate",
    "report",
]


def run_stage(config: dict[str, Any], stage: str, force: bool = False) -> None:
    logger = get_logger()
    project_paths = ensure_project_directories(config)
    write_json(project_paths["audit"] / "environment.json", environment_summary())
    resolved_config_path = project_paths["outputs"] / "resolved_config.yaml"
    resolved_config_path.write_text(
        yaml.safe_dump(public_config(config), sort_keys=False), encoding="utf-8"
    )
    stages = STAGES if stage == "all" else [stage]
    for current in stages:
        logger.info("Starting stage: %s", current)
        if current == "download":
            download_nhanes(config, force=force)
        elif current == "prepare":
            prepare_nhanes(config, force=force)
            preprocess_nhanes(config, force=force)
        elif current == "tasks":
            build_tasks(config, force=force)
        elif current == "adapt":
            train_wikipedia_adapted_encoder(config, force=force)
        elif current == "embed":
            generate_embeddings(config, force=force)
        elif current == "rank":
            select_operator_rank(config, force=force)
        elif current == "train":
            train_all_embeddings(config, force=force)
        elif current == "evaluate":
            evaluate_all(config, force=force)
        elif current == "report":
            generate_report(config, force=force)
        else:
            raise ValueError(f"Unknown stage: {current}")
        logger.info("Completed stage: %s", current)
