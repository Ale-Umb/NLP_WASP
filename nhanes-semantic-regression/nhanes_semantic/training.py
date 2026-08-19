from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import config_digest, ensure_project_directories
from .embeddings import load_embeddings
from .representations import build_required_representations, load_representation_archive
from .utils import (
    choose_torch_device,
    file_sha256,
    get_logger,
    read_json,
    set_seed,
    slugify,
    write_json,
)


def _loader(bundle: dict[str, Any], config: dict[str, Any], shuffle: bool, device):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    counts = np.bincount(bundle["task_index"])
    row_counts = counts[bundle["task_index"]].clip(min=1)
    weights = (1.0 / row_counts).astype(np.float32)
    weights /= float(weights.mean())
    dataset = TensorDataset(
        torch.from_numpy(bundle["z"]),
        torch.from_numpy(bundle["task_index"]),
        torch.from_numpy(bundle["y"]),
        torch.from_numpy(weights),
    )
    return DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=shuffle,
        num_workers=int(config["training"]["num_workers"]),
        pin_memory=device.type == "cuda",
        persistent_workers=int(config["training"]["num_workers"]) > 0,
        drop_last=False,
    )


def _amp_context(torch, device, enabled: bool, dtype):
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled)


def _evaluate_loss(
    model,
    loader,
    torch,
    device,
    amp_enabled: bool,
    amp_dtype,
    targets: list[str],
) -> dict[str, float]:
    model.eval()
    total_squared_error = 0.0
    total_rows = 0
    task_sse: dict[int, float] = {}
    task_n: dict[int, int] = {}
    with torch.no_grad():
        for z, task_index, y, _ in loader:
            z = z.to(device, non_blocking=True)
            task_index = task_index.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with _amp_context(torch, device, amp_enabled, amp_dtype):
                prediction = model(z, task_index)
                squared_error = (prediction - y).square().sum()
            total_squared_error += float(squared_error.detach().cpu())
            total_rows += int(y.numel())
            errors = (prediction.float() - y.float()).square().detach().cpu().numpy()
            indices = task_index.detach().cpu().numpy()
            for index in np.unique(indices):
                mask = indices == index
                task_sse[int(index)] = task_sse.get(int(index), 0.0) + float(errors[mask].sum())
                task_n[int(index)] = task_n.get(int(index), 0) + int(mask.sum())
    target_task_rmse: dict[str, list[float]] = {}
    for index, sse in task_sse.items():
        if index >= len(targets):
            raise IndexError(f"Task index {index} exceeds target bank")
        target_task_rmse.setdefault(str(targets[index]), []).append(
            math.sqrt(sse / max(task_n[index], 1))
        )
    target_macro_rmse = float(
        np.mean([np.mean(values) for values in target_task_rmse.values()])
    )
    return {
        "pooled_mse": total_squared_error / max(total_rows, 1),
        "target_macro_rmse": target_macro_rmse,
    }


def selected_rank_path(config: dict[str, Any]) -> Path:
    return ensure_project_directories(config)["audit"] / "selected_operator_rank.json"


def resolve_operator_rank(config: dict[str, Any]) -> int:
    selection = config.get("rank_selection", {})
    if not bool(selection.get("enabled", False)):
        return int(config["training"]["rank"])
    path = selected_rank_path(config)
    if not path.exists():
        raise FileNotFoundError(
            "Validation-selected operator rank is missing; run `python run_project.py --stage rank`"
        )
    return int(read_json(path)["selected_rank"])


def train_embedding(
    config: dict[str, Any],
    embedding_path: Path,
    force: bool = False,
    *,
    rank_override: int | None = None,
    seed_override: int | None = None,
    checkpoint_subdirectory: str | None = None,
    checkpoint_stem: str | None = None,
    force_representations: bool | None = None,
) -> Path:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Training requires PyTorch. Install the CUDA wheel shown in README.md.") from exc
    from .model import LowRankBilinearRegressor

    logger = get_logger()
    project_paths = ensure_project_directories(config)
    _, embedding_metadata = load_embeddings(embedding_path)
    name = slugify(embedding_metadata["name"])
    checkpoint_directory = project_paths["checkpoints"]
    if checkpoint_subdirectory:
        checkpoint_directory = checkpoint_directory / checkpoint_subdirectory
    stem = checkpoint_stem or name
    checkpoint_path = checkpoint_directory / f"{stem}.pt"
    history_path = checkpoint_directory / f"{stem}_history.csv"
    requested_rank = resolve_operator_rank(config) if rank_override is None else int(rank_override)
    requested_seed = int(
        config["project"]["seed"] if seed_override is None else seed_override
    )
    source_fingerprints = {
        "config_digest": config_digest(config),
        "embedding_sha256": file_sha256(embedding_path),
        "tasks_sha256": file_sha256(project_paths["tasks"] / "tasks.csv"),
        "rank": requested_rank,
        "training_seed": requested_seed,
    }
    if checkpoint_path.exists() and history_path.exists() and not force:
        try:
            try:
                existing = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            except TypeError:
                existing = torch.load(checkpoint_path, map_location="cpu")
            if existing.get("source_fingerprints") == source_fingerprints:
                logger.info("Reusing checkpoint %s", checkpoint_path)
                return checkpoint_path
            logger.info("Rebuilding stale checkpoint %s", checkpoint_path)
        except (OSError, KeyError, RuntimeError, ValueError):
            logger.info("Rebuilding unreadable checkpoint %s", checkpoint_path)

    representation_force = force if force_representations is None else force_representations
    representations = build_required_representations(
        config, embedding_path, force=bool(representation_force)
    )
    train_bundle = load_representation_archive(representations["train"])
    validation_bundle = load_representation_archive(representations["validation"])
    if train_bundle["metadata"]["task_ids"] != validation_bundle["metadata"]["task_ids"]:
        raise ValueError("Representation task banks are inconsistent")
    if not np.allclose(train_bundle["target_bank"], validation_bundle["target_bank"]):
        raise ValueError("Representation target embedding banks are inconsistent")

    seed = requested_seed
    set_seed(seed, bool(config["project"]["deterministic_torch"]))
    device = choose_torch_device(config["project"]["device"])
    if device.type == "cuda" and bool(config["training"]["allow_tf32"]):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    target_bank = torch.from_numpy(train_bundle["target_bank"]).to(device)
    model = LowRankBilinearRegressor(target_bank, requested_rank).to(device)
    optimization_model = model
    if bool(config["training"]["compile_model"]) and hasattr(torch, "compile"):
        optimization_model = torch.compile(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    amp_enabled = device.type == "cuda" and bool(config["training"]["amp"])
    use_bfloat16 = (
        amp_enabled
        and bool(config["training"]["prefer_bfloat16"])
        and torch.cuda.is_bf16_supported()
    )
    amp_dtype = torch.bfloat16 if use_bfloat16 else torch.float16
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and not use_bfloat16)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled and not use_bfloat16)

    train_loader = _loader(train_bundle, config, shuffle=True, device=device)
    validation_loader = _loader(validation_bundle, config, shuffle=False, device=device)
    best_validation = float("inf")
    best_state: dict[str, Any] | None = None
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []

    logger.info(
        "Training %s on %s: %d train / %d validation examples, dim=%d, rank=%d",
        name,
        device,
        len(train_bundle["y"]),
        len(validation_bundle["y"]),
        model.dimension,
        model.rank,
    )
    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        optimization_model.train()
        total_squared_error = 0.0
        total_rows = 0
        for z, task_index, y, row_weight in train_loader:
            z = z.to(device, non_blocking=True)
            task_index = task_index.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            row_weight = row_weight.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with _amp_context(torch, device, amp_enabled, amp_dtype):
                prediction = optimization_model(z, task_index)
                squared_error = (prediction - y).square()
                loss = (squared_error * row_weight).sum() / row_weight.sum().clamp_min(1e-12)
            scaler.scale(loss).backward()
            if float(config["training"]["gradient_clip_norm"]) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(config["training"]["gradient_clip_norm"])
                )
            scaler.step(optimizer)
            scaler.update()
            total_squared_error += float(loss.detach().cpu()) * int(y.numel())
            total_rows += int(y.numel())

        train_mse = total_squared_error / max(total_rows, 1)
        validation = _evaluate_loss(
            optimization_model,
            validation_loader,
            torch,
            device,
            amp_enabled,
            amp_dtype,
            validation_bundle["metadata"]["targets"],
        )
        history.append(
            {
                "epoch": epoch,
                "train_task_balanced_mse": train_mse,
                "validation_pooled_mse": validation["pooled_mse"],
                "validation_target_macro_rmse": validation["target_macro_rmse"],
            }
        )
        logger.info(
            "%s epoch %03d | task-balanced train MSE %.5f | held-target validation RMSE %.5f",
            name,
            epoch,
            train_mse,
            validation["target_macro_rmse"],
        )
        if validation["target_macro_rmse"] < best_validation - float(config["training"]["min_delta"]):
            best_validation = validation["target_macro_rmse"]
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= int(config["training"]["patience"]):
                logger.info("Early stopping %s after epoch %d", name, epoch)
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a finite validation checkpoint")
    model.load_state_dict(best_state)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "state_dict": best_state,
        "embedding_metadata": embedding_metadata,
        "rank": model.rank,
        "dimension": model.dimension,
        "task_ids": train_bundle["metadata"]["task_ids"],
        "targets": train_bundle["metadata"]["targets"],
        "best_validation_target_macro_rmse": best_validation,
        "training_seed": seed,
        "config_digest": config_digest(config),
        "representations": {key: str(value) for key, value in representations.items()},
        "amp_dtype": str(amp_dtype) if amp_enabled else "disabled",
        "device_used": str(device),
        "source_fingerprints": source_fingerprints,
    }
    torch.save(checkpoint, checkpoint_path)
    pd.DataFrame(history).to_csv(history_path, index=False)
    return checkpoint_path


def train_all_embeddings(config: dict[str, Any], force: bool = False) -> list[Path]:
    project_paths = ensure_project_directories(config)
    paths = []
    for model_spec in config["embeddings"]["models"]:
        embedding_path = project_paths["embeddings"] / f"{slugify(model_spec['name'])}.npz"
        if not embedding_path.exists():
            raise FileNotFoundError(f"Missing {embedding_path}; run embed first")
        paths.append(train_embedding(config, embedding_path, force=force))
    return paths


def select_operator_rank(config: dict[str, Any], force: bool = False) -> Path:
    """Select one global rank on held-target validation data, never on test tasks."""

    logger = get_logger()
    project_paths = ensure_project_directories(config)
    output_path = selected_rank_path(config)
    selection = config.get("rank_selection", {})
    if output_path.exists() and not force:
        existing = read_json(output_path)
        tasks_path = project_paths["tasks"] / "tasks.csv"
        current_tasks = file_sha256(tasks_path) if tasks_path.exists() else ""
        if (
            existing.get("config_digest") == config_digest(config)
            and existing.get("tasks_sha256") == current_tasks
        ):
            logger.info("Reusing validation-selected operator rank")
            return output_path
        logger.info("Rebuilding stale operator-rank selection")
    if not bool(selection.get("enabled", False)):
        write_json(
            output_path,
            {
                "selected_rank": int(config["training"]["rank"]),
                "selection_enabled": False,
                "config_digest": config_digest(config),
                "tasks_sha256": file_sha256(project_paths["tasks"] / "tasks.csv"),
            },
        )
        return output_path

    reference = slugify(str(selection["reference_embedding"]))
    embedding_path = project_paths["embeddings"] / f"{reference}.npz"
    if not embedding_path.exists():
        raise FileNotFoundError(f"Rank reference embedding is missing: {embedding_path}")
    candidates = sorted({int(value) for value in selection["candidates"]})
    repeats = int(selection["repeats"])
    base_seed = int(config["project"]["seed"])
    stride = int(selection.get("seed_stride", 1009))
    rows: list[dict[str, Any]] = []
    for rank in candidates:
        for repeat in range(repeats):
            stem = f"{reference}__rank_{rank:03d}__repeat_{repeat:02d}"
            path = train_embedding(
                config,
                embedding_path,
                force=force,
                rank_override=rank,
                seed_override=base_seed + repeat * stride,
                checkpoint_subdirectory="rank_selection",
                checkpoint_stem=stem,
                force_representations=False,
            )
            try:
                import torch

                checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            except TypeError:
                checkpoint = torch.load(path, map_location="cpu")
            rows.append(
                {
                    "reference_embedding": reference,
                    "rank": rank,
                    "repeat": repeat,
                    "seed": base_seed + repeat * stride,
                    "validation_target_macro_rmse": float(
                        checkpoint["best_validation_target_macro_rmse"]
                    ),
                    "checkpoint": str(path),
                }
            )
    runs = pd.DataFrame(rows)
    summary = (
        runs.groupby("rank")["validation_target_macro_rmse"]
        .agg([("validation_rmse_mean", "mean"), ("validation_rmse_sd", "std")])
        .reset_index()
        .sort_values(["validation_rmse_mean", "rank"])
    )
    selected = int(summary.iloc[0]["rank"])
    rank_dir = project_paths["checkpoints"] / "rank_selection"
    rank_dir.mkdir(parents=True, exist_ok=True)
    runs.to_csv(rank_dir / "rank_selection_runs.csv", index=False)
    summary.sort_values("rank").to_csv(rank_dir / "rank_selection_summary.csv", index=False)
    write_json(
        output_path,
        {
            "selected_rank": selected,
            "selection_enabled": True,
            "selection_metric": "held-target validation target-macro RMSE",
            "reference_embedding": reference,
            "candidates": candidates,
            "repeats": repeats,
            "test_tasks_used": False,
            "config_digest": config_digest(config),
            "tasks_sha256": file_sha256(project_paths["tasks"] / "tasks.csv"),
            "reference_embedding_sha256": file_sha256(embedding_path),
        },
    )
    logger.info("Selected global operator rank %d using held-target validation RMSE", selected)
    return output_path


def load_trained_model(checkpoint_path: Path, device):
    import torch
    from .model import LowRankBilinearRegressor

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint["state_dict"]
    target_bank = state["target_bank"].to(device)
    model = LowRankBilinearRegressor(target_bank, int(checkpoint["rank"])).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, checkpoint
