from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import config_digest, ensure_project_directories
from .embeddings import load_embeddings
from .preprocess import normalize_target, normalize_values
from .task_factory import load_tasks
from .utils import file_sha256, get_logger, slugify


def task_order(tasks: pd.DataFrame) -> list[str]:
    return sorted(tasks["task_id"].astype(str).unique().tolist())


def target_embedding_bank(
    tasks: pd.DataFrame, embedding_lookup: dict[str, np.ndarray]
) -> tuple[list[str], list[str], np.ndarray]:
    ordered_ids = task_order(tasks)
    task_by_id = tasks.set_index("task_id")
    targets = [str(task_by_id.loc[task_id, "target"]) for task_id in ordered_ids]
    missing = sorted({target for target in targets if target not in embedding_lookup})
    if missing:
        raise KeyError(f"Targets missing embeddings: {missing[:10]}")
    bank = np.stack([embedding_lookup[target] for target in targets]).astype(np.float32)
    return ordered_ids, targets, bank


def build_representation_archive(
    config: dict[str, Any],
    embedding_path: Path,
    task_partition: str,
    row_partition: str,
    force: bool = False,
) -> Path:
    logger = get_logger()
    project_paths = ensure_project_directories(config)
    embedding_lookup, embedding_metadata = load_embeddings(embedding_path)
    embedding_name = slugify(embedding_metadata["name"])
    output_path = (
        project_paths["representations"]
        / embedding_name
        / f"tasks_{task_partition}__rows_{row_partition}.npz"
    )
    matrix_path = project_paths["processed"] / "nhanes_split.pkl"
    stats_path = project_paths["processed"] / "variable_stats.csv"
    tasks_path = project_paths["tasks"] / "tasks.csv"
    fingerprints = {
        "config_digest": config_digest(config),
        "embedding_sha256": file_sha256(embedding_path),
        "stats_sha256": file_sha256(stats_path),
        "tasks_sha256": file_sha256(tasks_path),
        "matrix_bytes": int(matrix_path.stat().st_size),
        "matrix_mtime_ns": int(matrix_path.stat().st_mtime_ns),
    }
    if output_path.exists() and not force:
        try:
            with np.load(output_path, allow_pickle=False) as archive:
                existing = json.loads(str(archive["metadata"].item()))
            if existing.get("source_fingerprints") == fingerprints:
                return output_path
            logger.info("Rebuilding stale representation archive %s", output_path)
        except (OSError, KeyError, ValueError, json.JSONDecodeError):
            logger.info("Rebuilding unreadable representation archive %s", output_path)

    frame = pd.read_pickle(matrix_path)
    stats = pd.read_csv(stats_path).set_index("variable")
    tasks = load_tasks(tasks_path)
    ordered_task_ids, targets, bank = target_embedding_bank(tasks, embedding_lookup)
    task_index_lookup = {task_id: index for index, task_id in enumerate(ordered_task_ids)}
    selected_tasks = tasks[tasks["task_split"] == task_partition]
    selected_rows = frame[frame["_row_split"] == row_partition]
    id_column = config["nhanes"]["id_column"].upper()
    max_rows = int(config["training"]["max_rows_per_task"])
    aggregation = config["training"]["aggregation"]
    seed = int(config["project"]["seed"])

    z_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    task_parts: list[np.ndarray] = []
    row_parts: list[np.ndarray] = []
    task_counts: dict[str, int] = {}

    for row in selected_tasks.itertuples(index=False):
        task_id = str(row.task_id)
        target = str(row.target)
        features = list(row.feature_list)
        missing_embeddings = [name for name in [target, *features] if name not in embedding_lookup]
        if missing_embeddings:
            raise KeyError(f"{task_id} variables missing embeddings: {missing_embeddings}")
        raw_target = pd.to_numeric(selected_rows[target], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(raw_target)
        indices = np.flatnonzero(valid)
        if len(indices) == 0:
            task_counts[task_id] = 0
            continue
        if len(indices) > max_rows:
            local_seed = seed + sum(ord(char) for char in f"{task_id}|{row_partition}")
            rng = np.random.default_rng(local_seed)
            indices = np.sort(rng.choice(indices, size=max_rows, replace=False))

        x_columns = []
        for feature in features:
            raw = pd.to_numeric(selected_rows.iloc[indices][feature], errors="coerce").to_numpy()
            x_columns.append(normalize_values(raw, stats.loc[feature]))
        x_matrix = np.column_stack(x_columns).astype(np.float32)
        input_embeddings = np.stack([embedding_lookup[name] for name in features]).astype(np.float32)
        denominator = len(features) if aggregation == "mean" else np.sqrt(len(features))
        z = (x_matrix @ input_embeddings / float(denominator)).astype(np.float32)
        y = normalize_target(raw_target[indices], stats.loc[target])
        row_ids = selected_rows.iloc[indices][id_column].to_numpy()
        task_index = task_index_lookup[task_id]

        z_parts.append(z)
        y_parts.append(y)
        task_parts.append(np.full(len(y), task_index, dtype=np.int32))
        row_parts.append(np.asarray(row_ids, dtype=np.int64))
        task_counts[task_id] = int(len(y))

    if not z_parts:
        raise RuntimeError(
            f"No examples for task partition={task_partition}, row partition={row_partition}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "embedding": embedding_metadata,
        "task_partition": task_partition,
        "row_partition": row_partition,
        "task_ids": ordered_task_ids,
        "targets": targets,
        "task_counts": task_counts,
        "aggregation": aggregation,
        "dimension": int(bank.shape[1]),
        "source_fingerprints": fingerprints,
    }
    np.savez_compressed(
        output_path,
        z=np.concatenate(z_parts).astype(np.float32),
        y=np.concatenate(y_parts).astype(np.float32),
        task_index=np.concatenate(task_parts).astype(np.int32),
        row_id=np.concatenate(row_parts).astype(np.int64),
        target_bank=bank.astype(np.float32),
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    logger.info(
        "Packed %s/%s for %s: %d examples",
        task_partition,
        row_partition,
        embedding_name,
        sum(task_counts.values()),
    )
    return output_path


def load_representation_archive(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        return {
            "z": archive["z"].astype(np.float32),
            "y": archive["y"].astype(np.float32),
            "task_index": archive["task_index"].astype(np.int64),
            "row_id": archive["row_id"].astype(np.int64),
            "target_bank": archive["target_bank"].astype(np.float32),
            "metadata": json.loads(str(archive["metadata"].item())),
        }


def build_required_representations(
    config: dict[str, Any], embedding_path: Path, force: bool = False
) -> dict[str, Path]:
    pairs = {
        "train": ("train", "train"),
        "validation": ("validation", "validation"),
        "test": ("test", "test"),
    }
    return {
        name: build_representation_archive(
            config, embedding_path, task_partition, row_partition, force=force
        )
        for name, (task_partition, row_partition) in pairs.items()
    }
