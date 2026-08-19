from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a mapping: {config_path}")
    config["_config_path"] = str(config_path)
    config["_repo_root"] = str(REPO_ROOT)
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    row_fractions = config["preprocessing"]["row_split_fractions"]
    target_fractions = config["task_factory"]["target_split_fractions"]
    for name, values in (
        ("row_split_fractions", row_fractions),
        ("target_split_fractions", target_fractions),
    ):
        total = sum(float(value) for value in values.values())
        if abs(total - 1.0) > 1e-8:
            raise ValueError(f"{name} must sum to 1.0, got {total}")
        if any(float(value) <= 0 for value in values.values()):
            raise ValueError(f"Every {name} entry must be positive")
    if config["training"]["aggregation"] not in {"sqrt", "mean"}:
        raise ValueError("training.aggregation must be 'sqrt' or 'mean'")
    if config["evaluation"].get("mode", "zero_shot") != "zero_shot":
        raise ValueError("evaluation.mode must be 'zero_shot'")
    discovery = config["nhanes"].get("discovery", {})
    repeated_record_policy = str(
        config["nhanes"].get("repeated_record_policy", "exclude")
    ).lower()
    if repeated_record_policy not in {"exclude", "error"}:
        raise ValueError("nhanes.repeated_record_policy must be 'exclude' or 'error'")
    non_participant_table_policy = str(
        config["nhanes"].get("non_participant_table_policy", "exclude")
    ).lower()
    if non_participant_table_policy not in {"exclude", "error"}:
        raise ValueError(
            "nhanes.non_participant_table_policy must be 'exclude' or 'error'"
        )
    xpt_text_encodings = config["nhanes"].get("xpt_text_encodings", [])
    if not isinstance(xpt_text_encodings, list) or not xpt_text_encodings:
        raise ValueError("nhanes.xpt_text_encodings must be a non-empty list")
    for encoding in xpt_text_encodings:
        try:
            "".encode(str(encoding))
        except LookupError as exc:
            raise ValueError(f"Unknown XPT text encoding: {encoding}") from exc
    if bool(discovery.get("enabled", False)):
        if not discovery.get("components"):
            raise ValueError("nhanes.discovery.components cannot be empty")
        if "{component}" not in str(discovery.get("data_page_url", "")):
            raise ValueError("nhanes.discovery.data_page_url must contain {component}")

    task_factory = config["task_factory"]
    if int(task_factory.get("min_feature_domains", 1)) < 1:
        raise ValueError("task_factory.min_feature_domains must be positive")
    if int(task_factory.get("min_feature_tables", 1)) < 1:
        raise ValueError("task_factory.min_feature_tables must be positive")

    wikipedia = config["wikipedia"]
    if bool(wikipedia.get("enabled", True)):
        if not str(wikipedia.get("zim_url", "")).strip():
            raise ValueError("wikipedia.zim_url is required")
        if not str(wikipedia.get("zim_path", "")).strip():
            raise ValueError("wikipedia.zim_path is required")
        if int(wikipedia.get("zim_expected_bytes", 0)) < 0:
            raise ValueError("wikipedia.zim_expected_bytes cannot be negative")

    adaptation = config.get("wikipedia_adaptation", {})
    if bool(adaptation.get("enabled", False)):
        if int(adaptation.get("max_pairs", 0)) < 100:
            raise ValueError("wikipedia_adaptation.max_pairs must be at least 100")
        fraction = float(adaptation.get("validation_fraction", 0.0))
        if not 0.0 < fraction < 0.5:
            raise ValueError("wikipedia_adaptation.validation_fraction must be in (0, 0.5)")
        if int(adaptation.get("mini_batch_size", 1)) > int(adaptation.get("batch_size", 1)):
            raise ValueError("wikipedia_adaptation.mini_batch_size cannot exceed batch_size")

    rank_selection = config.get("rank_selection", {})
    if bool(rank_selection.get("enabled", False)):
        candidates = [int(value) for value in rank_selection.get("candidates", [])]
        if not candidates or any(value <= 0 for value in candidates):
            raise ValueError("rank_selection.candidates must contain positive ranks")
        if int(rank_selection.get("repeats", 0)) < 1:
            raise ValueError("rank_selection.repeats must be positive")
    names = [str(spec["name"]) for spec in config["embeddings"]["models"]]
    if len(names) != len(set(names)):
        raise ValueError("Embedding model names must be unique")
    kinds = {str(spec["kind"]) for spec in config["embeddings"]["models"]}
    unknown = kinds - {"sentence_transformer", "adapted_sentence_transformer", "random"}
    if unknown:
        raise ValueError(f"Unknown embedding model kinds: {sorted(unknown)}")


def project_path(config: dict[str, Any], key: str) -> Path:
    raw = Path(config["project"][key])
    return raw if raw.is_absolute() else REPO_ROOT / raw


def paths(config: dict[str, Any]) -> dict[str, Path]:
    data_dir = project_path(config, "data_dir")
    output_dir = project_path(config, "output_dir")
    result = {
        "repo": REPO_ROOT,
        "data": data_dir,
        "raw": data_dir / "raw",
        "processed": data_dir / "processed",
        "outputs": output_dir,
        "audit": output_dir / "data_audit",
        "tasks": output_dir / "tasks",
        "embeddings": output_dir / "embeddings",
        "representations": output_dir / "representations",
        "checkpoints": output_dir / "checkpoints",
        "metrics": output_dir / "metrics",
        "figures": output_dir / "figures",
        "logs": output_dir / "logs",
    }
    return result


def ensure_project_directories(config: dict[str, Any]) -> dict[str, Path]:
    result = paths(config)
    for path in result.values():
        if path != REPO_ROOT:
            path.mkdir(parents=True, exist_ok=True)
    return result


def public_config(config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if not key.startswith("_")}


def config_digest(config: dict[str, Any]) -> str:
    payload = json.dumps(public_config(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
