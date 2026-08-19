from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from .config import config_digest, ensure_project_directories
from .utils import assign_partition, file_sha256, get_logger, read_json, write_json


def preprocess_nhanes(config: dict[str, Any], force: bool = False) -> tuple[Path, Path, Path]:
    logger = get_logger()
    project_paths = ensure_project_directories(config)
    input_matrix = project_paths["processed"] / "nhanes_wide.pkl"
    input_catalog = project_paths["processed"] / "variable_catalog.csv"
    split_matrix = project_paths["processed"] / "nhanes_split.pkl"
    stats_path = project_paths["processed"] / "variable_stats.csv"
    eligible_catalog_path = project_paths["processed"] / "eligible_catalog.csv"
    manifest_path = project_paths["processed"] / "preprocess_manifest.json"
    expected_manifest = {
        "config_digest": config_digest(config),
        "input_matrix_bytes": int(input_matrix.stat().st_size) if input_matrix.exists() else -1,
        "input_matrix_mtime_ns": int(input_matrix.stat().st_mtime_ns) if input_matrix.exists() else -1,
        "input_catalog_sha256": file_sha256(input_catalog) if input_catalog.exists() else "",
    }
    if all(path.exists() for path in (split_matrix, stats_path, eligible_catalog_path)) and not force:
        if manifest_path.exists() and read_json(manifest_path) == expected_manifest:
            logger.info("Reusing participant splits and variable audit")
            return split_matrix, stats_path, eligible_catalog_path
        logger.info("Rebuilding stale participant splits and variable audit")
    if not input_matrix.exists() or not input_catalog.exists():
        raise FileNotFoundError("Run prepare before preprocessing")

    frame = pd.read_pickle(input_matrix)
    catalog = pd.read_csv(input_catalog).fillna("")
    id_column = config["nhanes"]["id_column"].upper()
    seed = int(config["project"]["seed"])
    fractions = config["preprocessing"]["row_split_fractions"]
    frame["_row_split"] = assign_partition(frame[id_column], fractions, seed, "participants")
    factory = frame.loc[frame["_row_split"] == "factory"]

    pre = config["preprocessing"]
    code_patterns = [re.compile(pattern, flags=re.IGNORECASE) for pattern in pre["excluded_code_regex"]]
    label_terms = [str(term).lower() for term in pre["excluded_label_terms"]]
    catalog_by_variable = catalog.set_index("variable", drop=False).to_dict("index")
    rows: list[dict[str, Any]] = []

    for variable in frame.columns:
        if variable == "_row_split":
            continue
        meta = catalog_by_variable.get(variable, {})
        label = " ".join(
            [str(meta.get("sas_label", "")), str(meta.get("english_text", ""))]
        ).lower()
        numeric = is_numeric_dtype(frame[variable])
        values = pd.to_numeric(factory[variable], errors="coerce") if numeric else pd.Series(dtype=float)
        finite = values[np.isfinite(values)] if numeric else values
        observed = int(finite.shape[0])
        observed_fraction = observed / max(len(factory), 1)
        unique = int(finite.nunique()) if observed else 0
        excluded_reason = ""
        if not numeric:
            excluded_reason = "non_numeric"
        elif not meta or not str(meta.get("embedding_text", "")).strip():
            excluded_reason = "missing_documentation"
        elif any(pattern.search(variable) for pattern in code_patterns):
            excluded_reason = "excluded_code"
        elif any(term in label for term in label_terms):
            excluded_reason = "excluded_documentation_term"
        elif observed < int(pre["min_factory_observations"]):
            excluded_reason = "too_few_factory_observations"
        elif observed_fraction < float(pre["min_observed_fraction"]):
            excluded_reason = "too_sparse"
        elif unique < int(pre["min_unique_values"]):
            excluded_reason = "too_few_unique_values"

        if observed:
            q_low = float(finite.quantile(float(pre["lower_winsor_quantile"])))
            q_high = float(finite.quantile(float(pre["upper_winsor_quantile"])))
            clipped = finite.clip(q_low, q_high)
            median = float(clipped.median())
            mean = float(clipped.mean())
            scale = float(clipped.std(ddof=0))
        else:
            q_low = q_high = median = mean = scale = float("nan")
        if np.isfinite(scale) and scale < float(pre["min_scale"]):
            excluded_reason = excluded_reason or "near_zero_scale"

        rows.append(
            {
                "variable": variable,
                "numeric": bool(numeric),
                "factory_observed": observed,
                "factory_observed_fraction": observed_fraction,
                "factory_unique": unique,
                "q_low": q_low,
                "q_high": q_high,
                "median": median,
                "mean": mean,
                "scale": scale,
                "eligible": excluded_reason == "",
                "excluded_reason": excluded_reason,
                "sas_label": meta.get("sas_label", ""),
                "english_text": meta.get("english_text", ""),
                "file_id": meta.get("file_id", ""),
                "component": meta.get("component", ""),
                "collection_component": meta.get("collection_component", ""),
                "semantic_domain": meta.get("semantic_domain", "unclassified"),
                "domain_rule": meta.get("domain_rule", ""),
                "missing_codes": meta.get("missing_codes", "[]"),
                "embedding_text": meta.get("embedding_text", ""),
            }
        )

    stats = pd.DataFrame(rows).sort_values("variable").reset_index(drop=True)
    eligible_variables = set(stats.loc[stats["eligible"], "variable"])
    eligible_catalog = catalog[catalog["variable"].isin(eligible_variables)].copy()
    eligible_catalog = eligible_catalog.sort_values("variable").reset_index(drop=True)

    frame.to_pickle(split_matrix)
    stats.to_csv(stats_path, index=False)
    eligible_catalog.to_csv(eligible_catalog_path, index=False)
    split_counts = frame["_row_split"].value_counts().sort_index().to_dict()
    write_json(
        project_paths["audit"] / "preprocessing_audit.json",
        {
            "participant_split_counts": {key: int(value) for key, value in split_counts.items()},
            "eligible_variables": int(stats["eligible"].sum()),
            "excluded_variables": int((~stats["eligible"]).sum()),
            "eligible_tables": int(eligible_catalog["file_id"].nunique()),
            "eligible_semantic_domains": int(eligible_catalog["semantic_domain"].nunique()),
            "exclusion_reasons": stats.loc[~stats["eligible"], "excluded_reason"]
            .value_counts()
            .to_dict(),
        },
    )
    write_json(manifest_path, expected_manifest)
    logger.info("Eligible continuous variables: %d", len(eligible_catalog))
    return split_matrix, stats_path, eligible_catalog_path


def normalize_values(values: np.ndarray, stat: pd.Series | dict[str, Any]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    q_low = float(stat["q_low"])
    q_high = float(stat["q_high"])
    median = float(stat["median"])
    mean = float(stat["mean"])
    scale = float(stat["scale"])
    array = np.where(np.isfinite(array), array, median)
    array = np.clip(array, q_low, q_high)
    return ((array - mean) / scale).astype(np.float32)


def normalize_target(values: np.ndarray, stat: pd.Series | dict[str, Any]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    q_low = float(stat["q_low"])
    q_high = float(stat["q_high"])
    mean = float(stat["mean"])
    scale = float(stat["scale"])
    array = np.clip(array, q_low, q_high)
    return ((array - mean) / scale).astype(np.float32)
