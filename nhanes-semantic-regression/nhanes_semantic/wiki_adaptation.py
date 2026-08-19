from __future__ import annotations

import gc
import hashlib
import heapq
import json
import math
import shutil
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .config import config_digest, ensure_project_directories
from .utils import (
    choose_torch_device,
    environment_summary,
    file_sha256,
    get_logger,
    set_seed,
    stable_unit_interval,
    text_sha256,
    write_json,
)
from .wikipedia_zim import OfflineWikipediaSnapshot, ensure_snapshot, plain_text_paragraphs


def adapted_model_path(config: dict[str, Any]) -> Path:
    project_paths = ensure_project_directories(config)
    configured = Path(str(config["wikipedia_adaptation"]["output_model_dir"]))
    path = configured if configured.is_absolute() else project_paths["data"] / configured
    resolved = path.resolve()
    data_root = project_paths["data"].resolve()
    if data_root not in resolved.parents:
        raise ValueError("wikipedia_adaptation.output_model_dir must be below project.data_dir")
    return resolved


def article_pairs(
    title: str,
    paragraphs: Iterable[str],
    *,
    article_path: str,
) -> list[dict[str, str]]:
    """Construct external-corpus positives without using any NHANES metadata."""

    clean_title = " ".join(str(title).split())
    clean_paragraphs = [" ".join(str(value).split()) for value in paragraphs if str(value).strip()]
    if len(clean_title) < 3 or not clean_paragraphs:
        return []
    if clean_title.lower().endswith("(disambiguation)") or "may refer to" in clean_paragraphs[0][:240].lower():
        return []
    rows = [
        {
            "article_path": article_path,
            "article_title": clean_title,
            "pair_type": "title_to_lead",
            "anchor": clean_title,
            "positive": clean_paragraphs[0],
        }
    ]
    if len(clean_paragraphs) >= 2:
        rows.append(
            {
                "article_path": article_path,
                "article_title": clean_title,
                "pair_type": "adjacent_paragraphs",
                "anchor": clean_paragraphs[0],
                "positive": clean_paragraphs[1],
            }
        )
    return rows


def _priority(value: str, seed: int) -> int:
    payload = f"{seed}|wikimed_pair|{value}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _build_pair_corpus(
    config: dict[str, Any],
    snapshot: OfflineWikipediaSnapshot,
    output_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    logger = get_logger()
    settings = config["wikipedia_adaptation"]
    maximum = int(settings["max_pairs"])
    seed = int(config["project"]["seed"])
    getter = getattr(snapshot.archive, "_get_entry_by_id", None)
    if getter is None:
        raise RuntimeError(
            "This libzim build cannot iterate the fixed snapshot. Install the pinned "
            "libzim requirement, which exposes Archive._get_entry_by_id."
        )

    heap: list[tuple[int, int, dict[str, str]]] = []
    valid_articles = 0
    scanned = int(snapshot.archive.all_entry_count)
    for entry_id in range(scanned):
        if entry_id and entry_id % 25000 == 0:
            logger.info(
                "WikiMed corpus scan: %d/%d entries, %d candidate articles",
                entry_id,
                scanned,
                valid_articles,
            )
        try:
            entry = getter(entry_id)
            if bool(entry.is_redirect):
                continue
            item = entry.get_item()
            if not str(item.mimetype).lower().startswith("text/html"):
                continue
            raw = bytes(item.content).decode("utf-8", errors="replace")
            paragraphs = plain_text_paragraphs(
                raw,
                min_characters=int(settings["min_paragraph_characters"]),
                max_characters=int(settings["max_paragraph_characters"]),
                maximum=3,
            )
            rows = article_pairs(
                str(item.title or entry.title),
                paragraphs,
                article_path=str(item.path),
            )
        except (KeyError, RuntimeError, UnicodeError, ValueError):
            continue
        if not rows:
            continue
        valid_articles += 1
        for row in rows:
            key = f"{row['article_path']}|{row['pair_type']}"
            priority = _priority(key, seed)
            tie = _priority(key, seed + 1)
            candidate = (-priority, -tie, row)
            if len(heap) < maximum:
                heapq.heappush(heap, candidate)
            elif priority < -heap[0][0]:
                heapq.heapreplace(heap, candidate)

    rows = [value[2] for value in heap]
    pairs = pd.DataFrame(rows)
    if len(pairs) < 100:
        raise RuntimeError(f"Only {len(pairs)} usable WikiMed pairs were extracted")
    pairs["priority"] = pairs.apply(
        lambda row: _priority(f"{row['article_path']}|{row['pair_type']}", seed), axis=1
    )
    pairs = pairs.sort_values(["priority", "article_path", "pair_type"]).reset_index(drop=True)
    fraction = float(settings["validation_fraction"])
    pairs["split"] = pairs["article_path"].map(
        lambda value: (
            "validation"
            if stable_unit_interval(value, seed, "wikimed_adaptation_split") < fraction
            else "train"
        )
    )
    if pairs["split"].nunique() != 2:
        raise RuntimeError("WikiMed pair hashing did not create train and validation splits")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(output_path, index=False, compression="gzip")
    audit = {
        "entries_scanned": scanned,
        "valid_articles_seen": valid_articles,
        "retained_pairs": int(len(pairs)),
        "train_pairs": int(pairs["split"].eq("train").sum()),
        "validation_pairs": int(pairs["split"].eq("validation").sum()),
        "unique_articles": int(pairs["article_path"].nunique()),
        "pair_text_sha256": text_sha256(
            f"{row.anchor}\0{row.positive}" for row in pairs.itertuples(index=False)
        ),
    }
    return pairs, audit


def build_wikimed_pair_corpus(
    config: dict[str, Any], force: bool = False
) -> tuple[Path, dict[str, Any]]:
    project_paths = ensure_project_directories(config)
    pair_path = project_paths["processed"] / "wikimed_adaptation_pairs.csv.gz"
    audit_path = project_paths["audit"] / "wikimed_corpus_manifest.json"
    snapshot_path, downloaded = ensure_snapshot(config, project_paths)
    snapshot = OfflineWikipediaSnapshot(
        snapshot_path,
        verify_checksum=bool(config["wikipedia"].get("verify_zim_checksum", True)),
    )
    if pair_path.exists() and audit_path.exists() and not force:
        with audit_path.open("r", encoding="utf-8") as handle:
            existing = json.load(handle)
        if existing.get("config_digest") == config_digest(config):
            return pair_path, existing
    pairs, corpus_audit = _build_pair_corpus(config, snapshot, pair_path)
    manifest = {
        **corpus_audit,
        "snapshot": snapshot.metadata(),
        "snapshot_path": str(snapshot_path),
        "snapshot_sha256": file_sha256(snapshot_path),
        "snapshot_downloaded_this_run": bool(downloaded),
        "corpus_file": str(pair_path),
        "construction_uses_nhanes_metadata": False,
        "construction_uses_participant_values": False,
        "construction_uses_benchmark_results": False,
        "config_digest": config_digest(config),
    }
    write_json(audit_path, manifest)
    return pair_path, manifest


def _retrieval_metrics(model, pairs: pd.DataFrame, batch_size: int) -> dict[str, float]:
    anchors = pairs["anchor"].astype(str).tolist()
    positives = pairs["positive"].astype(str).tolist()
    anchor_embeddings = np.asarray(
        model.encode(
            anchors,
            batch_size=int(batch_size),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ),
        dtype=np.float32,
    )
    positive_embeddings = np.asarray(
        model.encode(
            positives,
            batch_size=int(batch_size),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ),
        dtype=np.float32,
    )
    similarity = anchor_embeddings @ positive_embeddings.T
    diagonal = np.diag(similarity)
    ranks = 1 + (similarity > diagonal[:, None]).sum(axis=1)
    return {
        "n_pairs": int(len(pairs)),
        "recall_at_1": float(np.mean(ranks <= 1)),
        "recall_at_5": float(np.mean(ranks <= 5)),
        "mean_reciprocal_rank": float(np.mean(1.0 / ranks)),
        "mean_positive_cosine": float(np.mean(diagonal)),
    }


def train_wikipedia_adapted_encoder(config: dict[str, Any], force: bool = False) -> Path:
    settings = config["wikipedia_adaptation"]
    if not bool(settings.get("enabled", False)):
        raise RuntimeError("wikipedia_adaptation.enabled is false")
    logger = get_logger()
    project_paths = ensure_project_directories(config)
    model_path = adapted_model_path(config)
    manifest_path = project_paths["audit"] / "wikimed_adaptation_manifest.json"
    if model_path.exists() and manifest_path.exists() and not force:
        with manifest_path.open("r", encoding="utf-8") as handle:
            existing = json.load(handle)
        if existing.get("config_digest") == config_digest(config):
            logger.info("Reusing WikiMed-adapted encoder at %s", model_path)
            return model_path
        logger.info("Rebuilding stale WikiMed-adapted encoder")

    try:
        import torch
        from datasets import Dataset
        from sentence_transformers import (
            SentenceTransformer,
            SentenceTransformerTrainer,
            SentenceTransformerTrainingArguments,
            losses,
        )
        from sentence_transformers.sentence_transformer.training_args import BatchSamplers
    except ImportError as exc:
        raise RuntimeError(
            "Wikipedia encoder adaptation requires PyTorch and sentence-transformers"
        ) from exc

    pair_path, corpus_manifest = build_wikimed_pair_corpus(config, force=force)
    pairs = pd.read_csv(pair_path).fillna("")
    train_pairs = pairs[pairs["split"] == "train"].copy()
    validation_pairs = pairs[pairs["split"] == "validation"].copy()
    validation_pairs = validation_pairs.head(int(settings["retrieval_validation_pairs"]))
    if train_pairs.empty or validation_pairs.empty:
        raise RuntimeError("WikiMed adaptation needs non-empty train and validation pairs")

    seed = int(config["project"]["seed"])
    set_seed(seed, bool(config["project"]["deterministic_torch"]))
    device = choose_torch_device(config["project"]["device"])
    if device.type == "cuda" and bool(config["training"].get("allow_tf32", True)):
        torch.backends.cuda.matmul.allow_tf32 = True
    model = SentenceTransformer(str(settings["base_model_id"]), device=str(device))
    model.max_seq_length = int(settings["max_seq_length"])
    before = _retrieval_metrics(model, validation_pairs, int(settings["mini_batch_size"]))

    train_dataset = Dataset.from_dict(
        {
            "anchor": train_pairs["anchor"].astype(str).tolist(),
            "positive": train_pairs["positive"].astype(str).tolist(),
        }
    )
    objective = str(settings.get("objective", "cached_multiple_negatives_ranking"))
    if objective == "cached_multiple_negatives_ranking":
        try:
            loss = losses.CachedMultipleNegativesRankingLoss(
                model, mini_batch_size=int(settings["mini_batch_size"])
            )
        except AttributeError as exc:
            raise RuntimeError(
                "The installed sentence-transformers version lacks "
                "CachedMultipleNegativesRankingLoss"
            ) from exc
    elif objective == "multiple_negatives_ranking":
        loss = losses.MultipleNegativesRankingLoss(model)
    else:
        raise ValueError(f"Unknown Wikipedia adaptation objective: {objective}")

    total_steps = int(
        math.ceil(len(train_pairs) / int(settings["batch_size"]))
        * int(settings["epochs"])
    )
    warmup_steps = int(round(total_steps * float(settings["warmup_fraction"])))
    checkpoint_path = project_paths["data"] / "models" / "wikimed_adaptation_checkpoints"
    if model_path.exists():
        shutil.rmtree(model_path)
    logger.info(
        "Adapting %s on %d external WikiMed pairs (%d steps) using %s",
        settings["base_model_id"],
        len(train_pairs),
        total_steps,
        device,
    )
    arguments = SentenceTransformerTrainingArguments(
        output_dir=str(checkpoint_path),
        num_train_epochs=int(settings["epochs"]),
        per_device_train_batch_size=int(settings["batch_size"]),
        learning_rate=float(settings["learning_rate"]),
        warmup_ratio=float(settings["warmup_fraction"]),
        weight_decay=float(settings["weight_decay"]),
        fp16=device.type == "cuda" and bool(settings.get("use_amp", True)),
        bf16=False,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        save_strategy="steps",
        save_steps=int(settings["checkpoint_save_steps"]),
        save_total_limit=2,
        logging_steps=max(1, min(100, total_steps // 10)),
        report_to="none",
        seed=seed,
        data_seed=seed,
        run_name=str(settings["name"]),
    )
    trainer = SentenceTransformerTrainer(
        model=model,
        args=arguments,
        train_dataset=train_dataset,
        loss=loss,
    )
    trainer.train()
    model.save_pretrained(str(model_path))
    pd.DataFrame(trainer.state.log_history).to_csv(
        project_paths["audit"] / "wikimed_adaptation_training_log.csv", index=False
    )
    after = _retrieval_metrics(model, validation_pairs, int(settings["mini_batch_size"]))
    metrics = pd.DataFrame(
        [
            {"encoder_state": "base", **before},
            {"encoder_state": "wikimed_adapted", **after},
        ]
    )
    metrics.to_csv(project_paths["audit"] / "wikimed_adaptation_retrieval.csv", index=False)
    manifest = {
        "name": settings["name"],
        "base_model_id": settings["base_model_id"],
        "base_model_source_url": settings["base_model_source_url"],
        "output_model_dir": str(model_path),
        "objective": objective,
        "epochs": int(settings["epochs"]),
        "batch_size": int(settings["batch_size"]),
        "mini_batch_size": int(settings["mini_batch_size"]),
        "learning_rate": float(settings["learning_rate"]),
        "warmup_steps": warmup_steps,
        "seed": seed,
        "config_digest": config_digest(config),
        "corpus": corpus_manifest,
        "validation_retrieval": {"base": before, "adapted": after},
        "environment": environment_summary(),
        "nhanes_text_used_for_adaptation": False,
        "participant_values_used_for_adaptation": False,
        "benchmark_outcomes_used_for_adaptation": False,
    }
    write_json(manifest_path, manifest)
    del model, loss, trainer, train_dataset
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return model_path
