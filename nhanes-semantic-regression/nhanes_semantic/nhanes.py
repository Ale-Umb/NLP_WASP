from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import config_digest, ensure_project_directories
from .documentation import parse_codebook_html
from .nhanes_catalog import resolve_file_manifest
from .utils import file_sha256, get_logger, read_json, write_json


def _urls(config: dict[str, Any], file_id: str) -> tuple[str, str]:
    nhanes = config["nhanes"]
    base = nhanes["base_url"].format(year=nhanes["public_year_path"]).rstrip("/")
    return f"{base}/{file_id}.XPT", f"{base}/{file_id}.htm"


def download_url(
    url: str, destination: Path, timeout: int, retries: int, force: bool = False
) -> Path:
    logger = get_logger()
    if destination.exists() and destination.stat().st_size > 0 and not force:
        logger.info("Reusing %s", destination)
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "NHANES-semantic-course-project/0.1"},
    )
    for attempt in range(1, retries + 1):
        try:
            logger.info("Downloading %s", url)
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = response.read()
            if len(payload) < 128:
                raise IOError(f"Downloaded payload is unexpectedly small ({len(payload)} bytes)")
            temporary.write_bytes(payload)
            temporary.replace(destination)
            return destination
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            if attempt == retries:
                raise RuntimeError(f"Failed to download {url} after {retries} attempts") from exc
            delay = min(2 ** (attempt - 1), 20)
            logger.warning("Download attempt %d failed; retrying in %ds", attempt, delay)
            time.sleep(delay)
    raise AssertionError("unreachable")


def download_nhanes(config: dict[str, Any], force: bool = False) -> None:
    project_paths = ensure_project_directories(config)
    raw = project_paths["raw"]
    nhanes = config["nhanes"]
    source_manifest = resolve_file_manifest(
        config,
        refresh=bool(nhanes.get("discovery", {}).get("refresh_manifest", False)),
    )
    manifest: list[dict[str, Any]] = []
    for item in source_manifest.to_dict("records"):
        file_id = str(item["file_id"]).upper()
        default_data_url, default_doc_url = _urls(config, file_id)
        data_url = str(item.get("data_url", "") or default_data_url)
        doc_url = str(item.get("documentation_url", "") or default_doc_url)
        xpt_path = download_url(
            data_url,
            raw / f"{file_id}.XPT",
            timeout=int(nhanes["timeout_seconds"]),
            retries=int(nhanes["retries"]),
            force=force,
        )
        doc_path = download_url(
            doc_url,
            raw / f"{file_id}.htm",
            timeout=int(nhanes["timeout_seconds"]),
            retries=int(nhanes["retries"]),
            force=force,
        )
        manifest.append(
            {
                "file_id": file_id,
                "component": item["component"],
                "collection_component": item.get("collection_component", ""),
                "semantic_domain": item.get("semantic_domain", ""),
                "domain_rule": item.get("domain_rule", ""),
                "data_url": data_url,
                "documentation_url": doc_url,
                "xpt_path": str(xpt_path),
                "documentation_path": str(doc_path),
                "xpt_bytes": xpt_path.stat().st_size,
                "documentation_bytes": doc_path.stat().st_size,
            }
        )
    write_json(project_paths["audit"] / "download_manifest.json", manifest)


def participant_record_profile(frame: pd.DataFrame, id_column: str) -> dict[str, Any]:
    """Describe whether an NHANES table is participant- or repeated-record level."""

    if id_column not in frame.columns:
        return {
            "participant_id_present": False,
            "raw_rows": int(len(frame)),
            "unique_participants": 0,
            "missing_id_rows": 0,
            "participants_with_multiple_records": 0,
            "repeated_rows_beyond_first": 0,
            "median_records_per_participant": 0.0,
            "max_records_per_participant": 0,
            "row_granularity": "non_participant_reference",
        }
    missing_ids = int(frame[id_column].isna().sum())
    counts = frame.loc[frame[id_column].notna(), id_column].value_counts(sort=False)
    participants = int(len(counts))
    participants_with_multiple = int(counts.gt(1).sum()) if participants else 0
    repeated_rows = int((counts - 1).clip(lower=0).sum()) if participants else 0
    maximum = int(counts.max()) if participants else 0
    median = float(counts.median()) if participants else 0.0
    repeated = participants_with_multiple > 0
    return {
        "participant_id_present": True,
        "raw_rows": int(len(frame)),
        "unique_participants": participants,
        "missing_id_rows": missing_ids,
        "participants_with_multiple_records": participants_with_multiple,
        "repeated_rows_beyond_first": repeated_rows,
        "median_records_per_participant": median,
        "max_records_per_participant": maximum,
        "row_granularity": "repeated_record" if repeated else "participant",
    }


def read_xpt_with_fallback(
    path: Path,
    encodings: list[str] | tuple[str, ...] = ("utf-8", "cp1252", "latin-1"),
) -> tuple[pd.DataFrame, str]:
    """Read XPORT text as bytes, then apply one audited table-wide decoding."""

    frame = pd.read_sas(path, format="xport", encoding=None)
    object_columns = [
        column for column in frame.columns if pd.api.types.is_object_dtype(frame[column])
    ]
    byte_types = (bytes, bytearray, np.bytes_)
    contains_bytes = any(
        any(isinstance(value, byte_types) for value in frame[column].dropna())
        for column in object_columns
    )
    if not contains_bytes:
        return frame, "not_applicable"

    last_error: UnicodeDecodeError | None = None
    for encoding in encodings:
        decoded: dict[str, pd.Series] = {}
        try:
            for column in object_columns:
                decoded[column] = frame[column].map(
                    lambda value: (
                        bytes(value).decode(str(encoding))
                        if isinstance(value, byte_types)
                        else value
                    )
                )
        except UnicodeDecodeError as exc:
            last_error = exc
            continue
        for column, values in decoded.items():
            frame[column] = values
        return frame, str(encoding)

    attempted = ", ".join(str(value) for value in encodings)
    raise RuntimeError(f"Could not decode text in {path.name} using: {attempted}") from last_error


def prepare_nhanes(config: dict[str, Any], force: bool = False) -> tuple[Path, Path]:
    logger = get_logger()
    project_paths = ensure_project_directories(config)
    matrix_path = project_paths["processed"] / "nhanes_wide.pkl"
    catalog_path = project_paths["processed"] / "variable_catalog.csv"
    manifest = resolve_file_manifest(config, refresh=False)
    prepare_manifest_path = project_paths["processed"] / "prepare_manifest.json"
    source_manifest_path = project_paths["audit"] / "nhanes_file_manifest.csv"
    raw = project_paths["raw"]
    raw_state: dict[str, Any] = {}
    for file_id in manifest["file_id"].astype(str):
        paths = [raw / f"{file_id}.XPT", raw / f"{file_id}.htm"]
        raw_state[file_id] = [
            {"bytes": int(path.stat().st_size), "mtime_ns": int(path.stat().st_mtime_ns)}
            if path.exists()
            else {"missing": True}
            for path in paths
        ]
    expected_prepare = {
        "config_digest": config_digest(config),
        "source_manifest_sha256": file_sha256(source_manifest_path),
        "raw_state": raw_state,
    }
    if matrix_path.exists() and catalog_path.exists() and not force:
        if prepare_manifest_path.exists() and read_json(prepare_manifest_path) == expected_prepare:
            logger.info("Reusing prepared NHANES matrix and catalog")
            return matrix_path, catalog_path
        logger.info("Rebuilding stale prepared NHANES matrix and catalog")

    id_column = config["nhanes"]["id_column"].upper()
    repeated_record_policy = str(
        config["nhanes"].get("repeated_record_policy", "exclude")
    ).lower()
    non_participant_table_policy = str(
        config["nhanes"].get("non_participant_table_policy", "exclude")
    ).lower()
    xpt_text_encodings = tuple(
        str(value) for value in config["nhanes"].get(
            "xpt_text_encodings", ["utf-8", "cp1252", "latin-1"]
        )
    )
    merged: pd.DataFrame | None = None
    catalogs: list[pd.DataFrame] = []
    file_audit: list[dict[str, Any]] = []

    for item in manifest.to_dict("records"):
        file_id = str(item["file_id"]).upper()
        xpt_path = raw / f"{file_id}.XPT"
        doc_path = raw / f"{file_id}.htm"
        if not xpt_path.exists() or not doc_path.exists():
            raise FileNotFoundError(
                f"Missing {file_id} inputs. Run the download stage before prepare."
            )
        frame, xpt_text_encoding = read_xpt_with_fallback(xpt_path, xpt_text_encodings)
        if xpt_text_encoding not in {"utf-8", "not_applicable"}:
            logger.warning(
                "%s text required %s decoding after UTF-8 was rejected",
                file_id,
                xpt_text_encoding,
            )
        frame.columns = [str(column).upper() for column in frame.columns]
        profile = participant_record_profile(frame, id_column)
        default_data_url, default_doc_url = _urls(config, file_id)
        data_url = str(item.get("data_url", "") or default_data_url)
        doc_url = str(item.get("documentation_url", "") or default_doc_url)
        if profile["row_granularity"] == "non_participant_reference":
            if non_participant_table_policy == "error":
                raise ValueError(
                    f"{file_id} does not contain {id_column} and "
                    "nhanes.non_participant_table_policy is 'error'"
                )
            logger.warning(
                "%s has no %s and is not participant-level (%d rows); "
                "excluding it from the person-level matrix",
                file_id,
                id_column,
                int(profile["raw_rows"]),
            )
            file_audit.append(
                {
                    "file_id": file_id,
                    **profile,
                    "rows": int(profile["raw_rows"]),
                    "columns_after_overlap_drop": 0,
                    "dropped_overlap": [],
                    "merge_action": "excluded_non_participant_table",
                    "data_url": data_url,
                    "collection_component": item.get("collection_component", ""),
                    "semantic_domain": item.get("semantic_domain", ""),
                    "xpt_text_encoding": xpt_text_encoding,
                    "documented_missing_values_replaced": 0,
                }
            )
            continue
        if int(profile["missing_id_rows"]) > 0:
            raise ValueError(f"{file_id} has missing {id_column} rows")

        catalog = parse_codebook_html(
            doc_path.read_bytes(),
            file_id,
            str(item["component"]),
            doc_url,
            collection_component=str(item.get("collection_component", "")),
            semantic_domain=str(item.get("semantic_domain", "")),
            domain_rule=str(item.get("domain_rule", "")),
        )
        catalog["present_in_xpt"] = catalog["variable"].isin(frame.columns)
        repeated_records = profile["row_granularity"] == "repeated_record"
        catalog["table_row_granularity"] = profile["row_granularity"]
        catalog["included_in_person_level_matrix"] = not repeated_records
        catalog["table_exclusion_reason"] = (
            "repeated_participant_records" if repeated_records else ""
        )
        catalogs.append(catalog)

        if repeated_records:
            if repeated_record_policy == "error":
                raise ValueError(
                    f"{file_id} has repeated {id_column} rows and "
                    "nhanes.repeated_record_policy is 'error'"
                )
            logger.warning(
                "%s is repeated-record (%d rows, %d participants, up to %d records/person); "
                "excluding it from the person-level matrix",
                file_id,
                int(profile["raw_rows"]),
                int(profile["unique_participants"]),
                int(profile["max_records_per_participant"]),
            )
            file_audit.append(
                {
                    "file_id": file_id,
                    **profile,
                    "rows": int(profile["raw_rows"]),
                    "columns_after_overlap_drop": 0,
                    "dropped_overlap": [],
                    "merge_action": "excluded_repeated_record_table",
                    "data_url": data_url,
                    "collection_component": item.get("collection_component", ""),
                    "semantic_domain": item.get("semantic_domain", ""),
                    "xpt_text_encoding": xpt_text_encoding,
                    "documented_missing_values_replaced": 0,
                }
            )
            continue

        missing_replacements = 0
        for metadata in catalog.loc[catalog["present_in_xpt"]].to_dict("records"):
            variable = str(metadata["variable"])
            try:
                documented_missing = json.loads(str(metadata.get("missing_codes", "[]")))
            except json.JSONDecodeError:
                documented_missing = []
            if not documented_missing or variable not in frame:
                continue
            numeric = pd.to_numeric(frame[variable], errors="coerce")
            mask = numeric.isin([float(value) for value in documented_missing])
            missing_replacements += int(mask.sum())
            frame.loc[mask, variable] = np.nan

        if bool(config["nhanes"].get("keep_documented_columns_only", True)):
            documented = set(catalog.loc[catalog["present_in_xpt"], "variable"])
            frame = frame[[id_column, *sorted((set(frame.columns) & documented) - {id_column})]]

        if merged is None:
            merged = frame
            dropped_overlap: list[str] = []
        else:
            dropped_overlap = sorted((set(merged.columns) & set(frame.columns)) - {id_column})
            if dropped_overlap:
                logger.warning(
                    "%s overlaps earlier files on %d columns; keeping first occurrence",
                    file_id,
                    len(dropped_overlap),
                )
                frame = frame.drop(columns=dropped_overlap)
            merged = merged.merge(frame, on=id_column, how="outer", validate="one_to_one")
        file_audit.append(
            {
                "file_id": file_id,
                **profile,
                "rows": int(len(frame)),
                "columns_after_overlap_drop": int(frame.shape[1]),
                "dropped_overlap": dropped_overlap,
                "merge_action": "merged_person_level_table",
                "data_url": data_url,
                "collection_component": item.get("collection_component", ""),
                "semantic_domain": item.get("semantic_domain", ""),
                "xpt_text_encoding": xpt_text_encoding,
                "documented_missing_values_replaced": missing_replacements,
            }
        )

    assert merged is not None
    all_catalogs = pd.concat(catalogs, ignore_index=True)
    all_catalogs["_matrix_priority"] = all_catalogs[
        "included_in_person_level_matrix"
    ].astype(bool).astype(int)
    catalog_frame = (
        all_catalogs.sort_values(
            ["variable", "_matrix_priority"],
            ascending=[True, False],
            kind="stable",
        )
        .drop_duplicates("variable", keep="first")
        .drop(columns="_matrix_priority")
        .sort_values("variable")
        .reset_index(drop=True)
    )
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    merged.sort_values(id_column).reset_index(drop=True).to_pickle(matrix_path)
    catalog_frame.to_csv(catalog_path, index=False)
    granularity_audit = pd.DataFrame(file_audit).sort_values("file_id").reset_index(drop=True)
    granularity_audit.to_csv(
        project_paths["audit"] / "nhanes_table_granularity.csv", index=False
    )
    write_json(
        project_paths["audit"] / "merge_audit.json",
        {
            "rows": int(len(merged)),
            "columns": int(merged.shape[1]),
            "catalog_variables": int(len(catalog_frame)),
            "tables": int(len(manifest)),
            "person_level_tables_merged": int(
                granularity_audit["merge_action"].eq("merged_person_level_table").sum()
            ),
            "repeated_record_tables_excluded": int(
                granularity_audit["merge_action"]
                .eq("excluded_repeated_record_table")
                .sum()
            ),
            "non_participant_tables_excluded": int(
                granularity_audit["merge_action"]
                .eq("excluded_non_participant_table")
                .sum()
            ),
            "repeated_record_policy": repeated_record_policy,
            "non_participant_table_policy": non_participant_table_policy,
            "collection_components": sorted(manifest["collection_component"].unique().tolist()),
            "semantic_domains": sorted(catalog_frame["semantic_domain"].unique().tolist()),
            "files": file_audit,
        },
    )
    write_json(prepare_manifest_path, expected_prepare)
    logger.info("Prepared matrix with %d rows and %d columns", *merged.shape)
    return matrix_path, catalog_path
