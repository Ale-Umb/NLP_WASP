from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import config_digest, ensure_project_directories
from .representations import load_representation_archive
from .training import load_trained_model
from .utils import choose_torch_device, file_sha256, get_logger, read_json, slugify, write_json


def _predict_archive(config: dict[str, Any], checkpoint_path: Path, archive_path: Path) -> pd.DataFrame:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    bundle = load_representation_archive(archive_path)
    device = choose_torch_device(config["project"]["device"])
    model, checkpoint = load_trained_model(checkpoint_path, device)
    if checkpoint["task_ids"] != bundle["metadata"]["task_ids"]:
        raise ValueError("Checkpoint and representation task order differ")
    dataset = TensorDataset(
        torch.from_numpy(bundle["z"]), torch.from_numpy(bundle["task_index"])
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["training"]["num_workers"]),
        pin_memory=device.type == "cuda",
    )
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for z, task_index in loader:
            z = z.to(device, non_blocking=True)
            task_index = task_index.to(device, non_blocking=True)
            predictions.append(model(z, task_index).float().cpu().numpy())
    semantic_prediction = np.concatenate(predictions)
    identity_prediction = np.einsum(
        "ij,ij->i", bundle["z"], bundle["target_bank"][bundle["task_index"]]
    )
    task_ids = np.asarray(bundle["metadata"]["task_ids"], dtype=object)
    targets = np.asarray(bundle["metadata"]["targets"], dtype=object)
    return pd.DataFrame(
        {
            "row_id": bundle["row_id"],
            "task_index": bundle["task_index"],
            "task_id": task_ids[bundle["task_index"]],
            "target": targets[bundle["task_index"]],
            "y": bundle["y"],
            "semantic_prediction": semantic_prediction,
            "identity_prediction": identity_prediction,
        }
    )


def _metric_row(
    task_id: str,
    target: str,
    method: str,
    y: np.ndarray,
    prediction: np.ndarray,
    method_family: str,
    embedding: str = "",
) -> dict[str, Any]:
    residual = np.asarray(prediction) - np.asarray(y)
    rmse = float(np.sqrt(np.mean(np.square(residual))))
    mae = float(np.mean(np.abs(residual)))
    denominator = float(np.sum(np.square(y - np.mean(y))))
    r2 = float(1.0 - np.sum(np.square(residual)) / denominator) if denominator > 0 else float("nan")
    if len(y) >= 3 and np.std(y) > 0 and np.std(prediction) > 0:
        pearson = float(np.corrcoef(y, prediction)[0, 1])
    else:
        pearson = float("nan")
    return {
        "task_id": task_id,
        "target": target,
        "method": method,
        "method_family": method_family,
        "embedding": embedding,
        "n_test": int(len(y)),
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "pearson": pearson,
    }


def _prediction_metric_rows(
    test: pd.DataFrame,
    embedding_name: str,
    model_kind: str,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    min_test_rows = int(config["evaluation"]["min_test_rows"])
    include_identity = bool(config["evaluation"].get("include_identity_diagnostic", False))
    family = "random_control" if model_kind == "random" else "semantic_zero_shot"
    rows: list[dict[str, Any]] = []
    for task_id, test_group in test.groupby("task_id", sort=True):
        test_group = test_group.sort_values("row_id")
        if len(test_group) < min_test_rows:
            continue
        target = str(test_group["target"].iloc[0])
        y = test_group["y"].to_numpy()
        rows.append(
            _metric_row(
                task_id,
                target,
                f"semantic_zero_shot__{embedding_name}",
                y,
                test_group["semantic_prediction"].to_numpy(),
                family,
                embedding_name,
            )
        )
        if include_identity:
            rows.append(
                _metric_row(
                    task_id,
                    target,
                    f"identity_semantic__{embedding_name}",
                    y,
                    test_group["identity_prediction"].to_numpy(),
                    "identity_diagnostic",
                    embedding_name,
                )
            )
    return rows


def _zero_metric_rows(test: pd.DataFrame, config: dict[str, Any]) -> list[dict[str, Any]]:
    min_test_rows = int(config["evaluation"]["min_test_rows"])
    rows: list[dict[str, Any]] = []
    for task_id, group in test.groupby("task_id", sort=True):
        if len(group) < min_test_rows:
            continue
        rows.append(
            _metric_row(
                str(task_id),
                str(group["target"].iloc[0]),
                "zero",
                group["y"].to_numpy(),
                np.zeros(len(group), dtype=float),
                "zero_baseline",
            )
        )
    return rows


def _summarize_metrics(metrics: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    repetitions = int(config["evaluation"]["bootstrap_repetitions"])
    rng = np.random.default_rng(int(config["project"]["seed"]) + 404)
    rows: list[dict[str, Any]] = []
    for method, group in metrics.groupby("method", sort=True):
        row: dict[str, Any] = {
            "method": method,
            "method_family": group["method_family"].iloc[0],
            "embedding": group["embedding"].iloc[0],
            "n_tasks": int(group["task_id"].nunique()),
            "n_targets": int(group["target"].nunique()),
            "n_total_test_rows": int(group["n_test"].sum()),
        }
        for metric in ["rmse", "mae", "r2", "pearson"]:
            values = group.groupby("target")[metric].mean().to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            row[f"{metric}_mean"] = float(np.mean(values)) if len(values) else float("nan")
            if len(values) and repetitions > 0:
                draws = rng.choice(values, size=(repetitions, len(values)), replace=True).mean(axis=1)
                row[f"{metric}_ci_low"] = float(np.quantile(draws, 0.025))
                row[f"{metric}_ci_high"] = float(np.quantile(draws, 0.975))
            else:
                row[f"{metric}_ci_low"] = float("nan")
                row[f"{metric}_ci_high"] = float("nan")
        rows.append(row)
    return pd.DataFrame(rows).sort_values("rmse_mean").reset_index(drop=True)


def _add_pair(
    output: list[pd.DataFrame],
    metrics: pd.DataFrame,
    pair_type: str,
    reference_method: str,
    candidate_method: str,
    reference_embedding: str,
    candidate_embedding: str,
) -> None:
    subset = metrics[metrics["method"].isin([reference_method, candidate_method])]
    pivot = subset.pivot_table(index=["task_id", "target"], columns="method", values="rmse")
    if reference_method not in pivot or candidate_method not in pivot:
        return
    paired = pivot[[reference_method, candidate_method]].dropna().reset_index()
    if paired.empty:
        return
    paired = paired.rename(
        columns={reference_method: "reference_rmse", candidate_method: "candidate_rmse"}
    )
    paired["pair_type"] = pair_type
    paired["reference_method"] = reference_method
    paired["candidate_method"] = candidate_method
    paired["reference_embedding"] = reference_embedding
    paired["candidate_embedding"] = candidate_embedding
    paired["rmse_reference_minus_candidate"] = (
        paired["reference_rmse"] - paired["candidate_rmse"]
    )
    output.append(paired)


def _paired_zero_shot_deltas(metrics: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    specs = config["embeddings"]["models"]
    real_specs = [
        spec
        for spec in specs
        if spec["kind"] in {"sentence_transformer", "adapted_sentence_transformer"}
    ]
    controls = {
        int(spec["dimension"]): spec
        for spec in specs
        if spec["kind"] == "random"
    }
    output: list[pd.DataFrame] = []

    by_family: dict[str, dict[str, dict[str, Any]]] = {}
    for spec in real_specs:
        by_family.setdefault(str(spec["model_family"]), {})[
            str(spec.get("adaptation", "none"))
        ] = spec
    for variants in by_family.values():
        if "none" in variants and "wikimed_contrastive" in variants:
            reference = slugify(variants["none"]["name"])
            candidate = slugify(variants["wikimed_contrastive"]["name"])
            _add_pair(
                output,
                metrics,
                "encoder_adaptation",
                f"semantic_zero_shot__{reference}",
                f"semantic_zero_shot__{candidate}",
                reference,
                candidate,
            )

    for spec in real_specs:
        candidate = slugify(spec["name"])
        control_spec = controls.get(int(spec["dimension"]))
        if control_spec is not None:
            reference = slugify(control_spec["name"])
            _add_pair(
                output,
                metrics,
                "dimension_matched_random",
                f"semantic_zero_shot__{reference}",
                f"semantic_zero_shot__{candidate}",
                reference,
                candidate,
            )
        _add_pair(
            output,
            metrics,
            "zero_baseline",
            "zero",
            f"semantic_zero_shot__{candidate}",
            "zero",
            candidate,
        )
    return pd.concat(output, ignore_index=True) if output else pd.DataFrame()


def _summarize_paired_deltas(paired: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    if paired.empty:
        return pd.DataFrame()
    repetitions = int(config["evaluation"]["bootstrap_repetitions"])
    rng = np.random.default_rng(int(config["project"]["seed"]) + 919)
    rows: list[dict[str, Any]] = []
    group_columns = [
        "pair_type",
        "reference_method",
        "candidate_method",
        "reference_embedding",
        "candidate_embedding",
    ]
    for keys, group in paired.groupby(group_columns, sort=True):
        target_values = (
            group.groupby("target")["rmse_reference_minus_candidate"].mean().dropna()
        )
        values = target_values.to_numpy(dtype=float)
        if len(values) and repetitions > 0:
            draws = rng.choice(values, size=(repetitions, len(values)), replace=True).mean(axis=1)
            ci_low = float(np.quantile(draws, 0.025))
            ci_high = float(np.quantile(draws, 0.975))
        else:
            ci_low = ci_high = float("nan")
        rows.append(
            {
                **dict(zip(group_columns, keys)),
                "n_tasks": int(group["task_id"].nunique()),
                "n_targets": int(group["target"].nunique()),
                "rmse_improvement_mean": float(values.mean()) if len(values) else float("nan"),
                "rmse_improvement_ci_low": ci_low,
                "rmse_improvement_ci_high": ci_high,
                "rmse_improvement_median": float(np.median(values)) if len(values) else float("nan"),
                "candidate_target_win_rate": float(np.mean(values > 0)) if len(values) else float("nan"),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["pair_type", "rmse_improvement_mean"], ascending=[True, False]
    )


def evaluate_all(config: dict[str, Any], force: bool = False) -> tuple[Path, Path]:
    logger = get_logger()
    if config["evaluation"].get("mode", "zero_shot") != "zero_shot":
        raise ValueError("This project configuration supports evaluation.mode: zero_shot")
    project_paths = ensure_project_directories(config)
    per_task_path = project_paths["metrics"] / "per_task_metrics.csv"
    summary_path = project_paths["metrics"] / "summary_metrics.csv"
    manifest_path = project_paths["metrics"] / "evaluation_manifest.json"
    checkpoint_paths = [
        project_paths["checkpoints"] / f"{slugify(spec['name'])}.pt"
        for spec in config["embeddings"]["models"]
    ]
    evaluation_fingerprints = {
        "config_digest": config_digest(config),
        "checkpoints": {
            path.name: file_sha256(path) for path in checkpoint_paths if path.exists()
        },
        "tasks_sha256": file_sha256(project_paths["tasks"] / "tasks.csv"),
    }
    if per_task_path.exists() and summary_path.exists() and not force:
        if manifest_path.exists() and read_json(manifest_path) == evaluation_fingerprints:
            logger.info("Reusing evaluation outputs")
            return per_task_path, summary_path
        logger.info("Rebuilding stale evaluation outputs")

    all_metric_rows: list[dict[str, Any]] = []
    reference_test: pd.DataFrame | None = None
    for model_spec in config["embeddings"]["models"]:
        name = slugify(model_spec["name"])
        checkpoint_path = project_paths["checkpoints"] / f"{name}.pt"
        test_path = (
            project_paths["representations"] / name / "tasks_test__rows_test.npz"
        )
        for path in [checkpoint_path, test_path]:
            if not path.exists():
                raise FileNotFoundError(f"Missing {path}; run train before evaluate")
        test = _predict_archive(config, checkpoint_path, test_path)
        test.to_csv(project_paths["metrics"] / f"test_predictions_{name}.csv", index=False)
        if reference_test is None:
            reference_test = test
        all_metric_rows.extend(
            _prediction_metric_rows(test, name, str(model_spec["kind"]), config)
        )

    assert reference_test is not None
    all_metric_rows.extend(_zero_metric_rows(reference_test, config))
    metrics = pd.DataFrame(all_metric_rows)
    if metrics.empty:
        raise RuntimeError("No test task had enough rows for evaluation")
    summary = _summarize_metrics(metrics, config)
    paired = _paired_zero_shot_deltas(metrics, config)
    paired_summary = _summarize_paired_deltas(paired, config)

    per_task_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(per_task_path, index=False)
    summary.to_csv(summary_path, index=False)
    paired.to_csv(project_paths["metrics"] / "paired_zero_shot_task_deltas.csv", index=False)
    paired_summary.to_csv(
        project_paths["metrics"] / "paired_zero_shot_summary.csv", index=False
    )
    write_json(manifest_path, evaluation_fingerprints)
    logger.info("Evaluated %d zero-shot method-task pairs", len(metrics))
    return per_task_path, summary_path
