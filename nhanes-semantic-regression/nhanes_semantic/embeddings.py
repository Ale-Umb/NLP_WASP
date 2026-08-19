from __future__ import annotations

import gc
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import ensure_project_directories
from .utils import choose_torch_device, get_logger, slugify, text_sha256


TEXT_COLUMNS = {
    "official": "official_embedding_text",
    "control": "official_embedding_text",
}


def _catalog_for_models(config: dict[str, Any], project_paths: dict[str, Path]) -> pd.DataFrame:
    catalog_path = project_paths["processed"] / "eligible_catalog.csv"
    if not catalog_path.exists():
        raise FileNotFoundError("Run preprocessing before embedding generation")
    catalog = pd.read_csv(catalog_path).fillna("").sort_values("variable").reset_index(drop=True)
    if "official_embedding_text" not in catalog:
        catalog["official_embedding_text"] = catalog["embedding_text"]
    return catalog


def _texts_for_spec(catalog: pd.DataFrame, model_spec: dict[str, Any]) -> list[str]:
    variant = str(model_spec.get("text_variant", "official"))
    column = TEXT_COLUMNS.get(variant)
    if column is None:
        raise ValueError(f"Unknown embedding text_variant: {variant}")
    if column not in catalog:
        raise ValueError(f"Catalog has no text column for variant {variant!r}: {column}")
    texts = catalog[column].astype(str).tolist()
    if any(not text.strip() for text in texts):
        raise ValueError(f"At least one variable has empty {variant} documentation text")
    return texts


def _metadata(
    model_spec: dict[str, Any], model_id: str, matrix: np.ndarray, texts: list[str]
) -> dict[str, Any]:
    return {
        "name": slugify(model_spec["name"]),
        "display_name": model_spec.get("display_name", model_spec["name"]),
        "kind": model_spec["kind"],
        "model_id": model_id,
        "model_family": model_spec.get("model_family", model_spec["name"]),
        "domain": model_spec.get("domain", "unspecified"),
        "text_variant": model_spec.get("text_variant", "official"),
        "source_url": model_spec.get("source_url", ""),
        "license": model_spec.get("license", "unspecified"),
        "dimension": int(matrix.shape[1]),
        "variables": int(matrix.shape[0]),
        "documentation_sha256": text_sha256(texts),
        "normalized": True,
        "max_seq_length": model_spec.get("max_seq_length"),
        "base_model_id": model_spec.get("base_model_id", ""),
        "adaptation": model_spec.get("adaptation", "none"),
        "training_corpus_url": model_spec.get("training_corpus_url", ""),
    }


def _save_archive(
    output_path: Path,
    variables: list[str],
    matrix: np.ndarray,
    metadata: dict[str, Any],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        variables=np.asarray(variables, dtype=str),
        embeddings=matrix.astype(np.float32),
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
    )


def _source_row(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "embedding": metadata["name"],
        "display_name": metadata.get("display_name", metadata["name"]),
        "model_family": metadata.get("model_family", ""),
        "domain": metadata.get("domain", ""),
        "text_variant": metadata.get("text_variant", ""),
        "model_id": metadata.get("model_id", ""),
        "dimension": metadata.get("dimension", ""),
        "max_seq_length": metadata.get("max_seq_length", ""),
        "base_model_id": metadata.get("base_model_id", ""),
        "adaptation": metadata.get("adaptation", "none"),
        "training_corpus_url": metadata.get("training_corpus_url", ""),
        "source_url": metadata.get("source_url", ""),
        "license": metadata.get("license", ""),
        "documentation_sha256": metadata.get("documentation_sha256", ""),
    }


def generate_embeddings(config: dict[str, Any], force: bool = False) -> list[Path]:
    logger = get_logger()
    project_paths = ensure_project_directories(config)
    catalog = _catalog_for_models(config, project_paths)
    variables = catalog["variable"].astype(str).tolist()
    model_specs = config["embeddings"]["models"]
    output_paths: list[Path] = []
    source_rows: list[dict[str, Any]] = []
    pending_sentence: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)

    for model_spec in model_specs:
        name = slugify(model_spec["name"])
        output_path = project_paths["embeddings"] / f"{name}.npz"
        output_paths.append(output_path)
        if output_path.exists() and not force:
            lookup, metadata = load_embeddings(output_path)
            expected_texts = _texts_for_spec(catalog, model_spec)
            current = (
                list(lookup) == variables
                and metadata.get("documentation_sha256") == text_sha256(expected_texts)
                and str(metadata.get("kind")) == str(model_spec["kind"])
                and str(metadata.get("adaptation", "none"))
                == str(model_spec.get("adaptation", "none"))
            )
            if current:
                source_rows.append(_source_row(metadata))
                logger.info("Reusing %s", output_path)
                continue
            logger.info("Rebuilding stale embedding archive %s", output_path)
        if model_spec["kind"] == "random":
            texts = _texts_for_spec(catalog, model_spec)
            dimension = int(model_spec["dimension"])
            seed = int(config["project"]["seed"]) + sum(ord(char) for char in name)
            rng = np.random.default_rng(seed)
            matrix = rng.normal(size=(len(variables), dimension)).astype(np.float32)
            matrix /= np.linalg.norm(matrix, axis=1, keepdims=True).clip(min=1e-12)
            metadata = _metadata(
                model_spec, f"random_normal_seed_{seed}", matrix, texts
            )
            _save_archive(output_path, variables, matrix, metadata)
            source_rows.append(_source_row(metadata))
            logger.info("Saved %s embeddings with shape %s", name, matrix.shape)
        elif model_spec["kind"] in {"sentence_transformer", "adapted_sentence_transformer"}:
            model_id = str(model_spec["model_id"])
            if model_spec["kind"] == "adapted_sentence_transformer":
                from .wiki_adaptation import adapted_model_path

                resolved = adapted_model_path(config)
                if not resolved.exists():
                    raise FileNotFoundError(
                        f"Adapted encoder is missing at {resolved}; run --stage adapt first"
                    )
                model_id = str(resolved)
            key = (
                model_id,
                int(model_spec.get("max_seq_length", 0)),
                int(model_spec.get("batch_size", config["embeddings"]["batch_size"])),
            )
            pending_sentence[key].append(model_spec)
        else:
            raise ValueError(f"Unknown embedding kind: {model_spec['kind']}")

    for (model_id, max_seq_length, batch_size), grouped_specs in pending_sentence.items():
        text_sets = [_texts_for_spec(catalog, spec) for spec in grouped_specs]
        combined = [text for texts in text_sets for text in texts]
        combined_matrix = _encode_sentence_transformer(
            combined,
            model_id,
            config,
            max_seq_length=max_seq_length or None,
            batch_size=batch_size,
        )
        offset = 0
        for model_spec, texts in zip(grouped_specs, text_sets):
            matrix = combined_matrix[offset : offset + len(texts)]
            offset += len(texts)
            expected_dimension = model_spec.get("dimension")
            if expected_dimension is not None and matrix.shape[1] != int(expected_dimension):
                raise ValueError(
                    f"{model_spec['name']} produced dimension {matrix.shape[1]}, "
                    f"expected {expected_dimension}"
                )
            name = slugify(model_spec["name"])
            output_path = project_paths["embeddings"] / f"{name}.npz"
            recorded_model_id = str(model_spec.get("model_id", model_id))
            metadata = _metadata(model_spec, recorded_model_id, matrix, texts)
            _save_archive(output_path, variables, matrix, metadata)
            source_rows.append(_source_row(metadata))
            logger.info("Saved %s embeddings with shape %s", name, matrix.shape)

    sources = pd.DataFrame(source_rows).drop_duplicates("embedding").sort_values("embedding")
    sources.to_csv(project_paths["embeddings"] / "embedding_sources.csv", index=False)
    return output_paths


def _encode_sentence_transformer(
    texts: list[str],
    model_id: str,
    config: dict[str, Any],
    max_seq_length: int | None = None,
    batch_size: int | None = None,
) -> np.ndarray:
    try:
        import torch
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "Embedding generation requires PyTorch and sentence-transformers. See README.md."
        ) from exc

    device = choose_torch_device(config["project"]["device"])
    if device.type == "cuda" and bool(config["training"]["allow_tf32"]):
        torch.backends.cuda.matmul.allow_tf32 = True
    model = SentenceTransformer(model_id, device=str(device))
    if max_seq_length is not None:
        model.max_seq_length = int(max_seq_length)
    if device.type == "cuda" and bool(config["embeddings"]["fp16_on_cuda"]):
        model.half()
    matrix = model.encode(
        texts,
        batch_size=int(batch_size or config["embeddings"]["batch_size"]),
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
        device=str(device),
    )
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return np.asarray(matrix, dtype=np.float32)


def load_embeddings(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with np.load(path, allow_pickle=False) as archive:
        variables = archive["variables"].astype(str)
        matrix = archive["embeddings"].astype(np.float32)
        metadata = json.loads(str(archive["metadata"].item()))
    if len(variables) != len(matrix):
        raise ValueError(f"Corrupt embedding archive: {path}")
    lookup = {variable: matrix[index] for index, variable in enumerate(variables)}
    return lookup, metadata
