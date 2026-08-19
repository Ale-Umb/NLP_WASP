from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import config_digest, ensure_project_directories
from .utils import assign_partition, file_sha256, get_logger, read_json, write_json


UNIT_PAREN_RE = re.compile(
    r"\([^)]*(?:mg|g|kg|cm|mm|ml|l|mol|iu|percent|%|unit|conversion)[^)]*\)",
    flags=re.IGNORECASE,
)
NON_WORD_RE = re.compile(r"[^a-z0-9]+")


def concept_key(label: str) -> str:
    text = UNIT_PAREN_RE.sub(" ", str(label).lower())
    text = re.sub(r"\b(?:si units?|standard units?|converted)\b", " ", text)
    return NON_WORD_RE.sub(" ", text).strip()


def _manual_group_lookup(config: dict[str, Any]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for group, variables in config["task_factory"]["manual_leakage_groups"].items():
        for variable in variables:
            lookup[str(variable).upper()] = str(group)
    return lookup


def _pair_correlation(x: pd.Series, y: pd.Series) -> tuple[float, int]:
    pair = pd.concat(
        [pd.to_numeric(x, errors="coerce"), pd.to_numeric(y, errors="coerce")], axis=1
    ).replace([np.inf, -np.inf], np.nan).dropna()
    if len(pair) < 3 or pair.iloc[:, 0].nunique() < 2 or pair.iloc[:, 1].nunique() < 2:
        return float("nan"), int(len(pair))
    return float(pair.iloc[:, 0].corr(pair.iloc[:, 1])), int(len(pair))


def _qualify_task(
    factory: pd.DataFrame,
    target: str,
    features: list[str],
    config: dict[str, Any],
) -> dict[str, Any] | None:
    tf = config["task_factory"]
    columns = [config["nhanes"]["id_column"].upper(), "_factory_subsplit", target, *features]
    task_data = factory[columns].replace([np.inf, -np.inf], np.nan).dropna()
    if len(task_data) < int(tf["min_complete_rows"]):
        return None
    discovery = task_data[task_data["_factory_subsplit"] == "discovery"]
    qualification = task_data[task_data["_factory_subsplit"] == "qualification"]
    if len(discovery) < max(60, len(features) * 8) or len(qualification) < 40:
        return None

    model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("ridge", RidgeCV(alphas=np.asarray(tf["ridge_alphas"], dtype=float))),
        ]
    )
    model.fit(discovery[features], discovery[target])
    discovery_prediction = model.predict(discovery[features])
    prediction = model.predict(qualification[features])
    baseline = np.full(len(qualification), float(discovery[target].mean()))
    model_rmse = math.sqrt(mean_squared_error(qualification[target], prediction))
    baseline_rmse = math.sqrt(mean_squared_error(qualification[target], baseline))
    improvement = 1.0 - model_rmse / max(baseline_rmse, 1e-12)
    qualification_r2 = r2_score(qualification[target], prediction)
    if qualification_r2 < float(tf["min_qualification_r2"]):
        return None
    if improvement < float(tf["min_rmse_improvement_fraction"]):
        return None
    return {
        "n_factory_complete": int(len(task_data)),
        "n_discovery": int(len(discovery)),
        "n_qualification": int(len(qualification)),
        "discovery_r2": float(r2_score(discovery[target], discovery_prediction)),
        "qualification_r2": float(qualification_r2),
        "qualification_rmse": float(model_rmse),
        "qualification_baseline_rmse": float(baseline_rmse),
        "qualification_rmse_improvement": float(improvement),
        "qualification_ridge_alpha": float(model.named_steps["ridge"].alpha_),
    }


def build_tasks(config: dict[str, Any], force: bool = False) -> Path:
    logger = get_logger()
    project_paths = ensure_project_directories(config)
    tasks_path = project_paths["tasks"] / "tasks.csv"
    matrix_path = project_paths["processed"] / "nhanes_split.pkl"
    stats_path = project_paths["processed"] / "variable_stats.csv"
    manifest_path = project_paths["tasks"] / "task_factory_manifest.json"
    expected_manifest = {
        "config_digest": config_digest(config),
        "matrix_bytes": int(matrix_path.stat().st_size) if matrix_path.exists() else -1,
        "matrix_mtime_ns": int(matrix_path.stat().st_mtime_ns) if matrix_path.exists() else -1,
        "stats_sha256": file_sha256(stats_path) if stats_path.exists() else "",
    }
    if tasks_path.exists() and not force:
        if manifest_path.exists() and read_json(manifest_path) == expected_manifest:
            logger.info("Reusing %s", tasks_path)
            return tasks_path
        logger.info("Rebuilding stale task benchmark")

    if not matrix_path.exists() or not stats_path.exists():
        raise FileNotFoundError("Run prepare/preprocess before task creation")
    frame = pd.read_pickle(matrix_path)
    stats = pd.read_csv(stats_path).fillna("")
    eligible_stats = stats[stats["eligible"].astype(bool)].copy()
    eligible = [variable for variable in eligible_stats["variable"] if variable in frame.columns]
    if len(eligible) < 6:
        raise RuntimeError(f"Only {len(eligible)} eligible variables; cannot create task benchmark")

    seed = int(config["project"]["seed"])
    rng = np.random.default_rng(seed)
    id_column = config["nhanes"]["id_column"].upper()
    factory = frame.loc[frame["_row_split"] == "factory", [id_column, *eligible]].copy()
    factory["_factory_subsplit"] = assign_partition(
        factory[id_column],
        {"discovery": 0.60, "qualification": 0.40},
        seed,
        "task_qualification",
    )
    discovery = factory[factory["_factory_subsplit"] == "discovery"]

    label_lookup = eligible_stats.set_index("variable")["sas_label"].astype(str).to_dict()
    domain_lookup = (
        eligible_stats.set_index("variable")["semantic_domain"].astype(str).to_dict()
    )
    table_lookup = eligible_stats.set_index("variable")["file_id"].astype(str).to_dict()
    concept_lookup = {variable: concept_key(label_lookup.get(variable, "")) for variable in eligible}
    manual_group = _manual_group_lookup(config)
    tf = config["task_factory"]
    near_affine = float(tf["near_affine_correlation"])
    min_pairwise = int(tf["min_pairwise_observations"])
    pool_size = int(tf["correlation_pool_size"])

    accepted: list[dict[str, Any]] = []
    target_audit: list[dict[str, Any]] = []
    leakage_rows: list[dict[str, Any]] = []

    target_order = list(eligible)
    rng.shuffle(target_order)
    for target in target_order:
        if len(accepted) >= int(tf["max_total_tasks"]):
            break
        safe_correlations: dict[str, float] = {}
        reasons = Counter()
        for feature in eligible:
            reason = ""
            correlation = float("nan")
            pair_n = 0
            if feature == target:
                reason = "target_itself"
            elif manual_group.get(feature) and manual_group.get(feature) == manual_group.get(target):
                reason = "manual_same_concept_group"
            elif (
                concept_lookup.get(feature)
                and concept_lookup.get(feature) == concept_lookup.get(target)
            ):
                reason = "same_normalized_documentation_concept"
            elif bool(tf.get("require_target_feature_domain_difference", True)) and (
                domain_lookup.get(feature, "unclassified")
                == domain_lookup.get(target, "unclassified")
            ):
                reason = "same_target_semantic_domain"
            else:
                correlation, pair_n = _pair_correlation(discovery[feature], discovery[target])
                if pair_n < min_pairwise:
                    reason = "too_few_pairwise_rows"
                elif not np.isfinite(correlation):
                    reason = "undefined_correlation"
                elif abs(correlation) >= near_affine:
                    reason = "near_affine_duplicate"
            if reason:
                reasons[reason] += 1
                if reason in {
                    "manual_same_concept_group",
                    "same_normalized_documentation_concept",
                    "near_affine_duplicate",
                }:
                    leakage_rows.append(
                        {
                            "target": target,
                            "blocked_feature": feature,
                            "reason": reason,
                            "factory_discovery_correlation": correlation,
                            "pairwise_n": pair_n,
                        }
                    )
            else:
                safe_correlations[feature] = correlation

        ranked = sorted(safe_correlations, key=lambda key: abs(safe_correlations[key]), reverse=True)
        signal_pool = ranked[:pool_size]
        safe_pool = list(safe_correlations)
        accepted_for_target = 0
        attempts = 0
        seen_feature_sets: set[tuple[str, ...]] = set()

        while (
            attempts < int(tf["max_attempts_per_target"])
            and accepted_for_target < int(tf["tasks_per_target"])
            and len(accepted) < int(tf["max_total_tasks"])
        ):
            attempts += 1
            maximum = min(int(tf["feature_count_max"]), len(safe_pool))
            minimum = min(int(tf["feature_count_min"]), maximum)
            if maximum < 2 or minimum < 2 or not signal_pool:
                break
            feature_count = int(rng.integers(minimum, maximum + 1))
            signal_count = min(
                len(signal_pool),
                feature_count,
                max(1, int(round(feature_count * float(tf["signal_fraction"])))),
            )
            weights = np.asarray([abs(safe_correlations[name]) + 1e-4 for name in signal_pool])
            weights = weights / weights.sum()
            signal = list(rng.choice(signal_pool, size=signal_count, replace=False, p=weights))
            remainder_pool = [name for name in safe_pool if name not in signal]
            remainder_count = feature_count - len(signal)
            if remainder_count > len(remainder_pool):
                continue
            distractors = (
                list(rng.choice(remainder_pool, size=remainder_count, replace=False))
                if remainder_count
                else []
            )
            features = sorted(signal + distractors)
            feature_domains = sorted(
                {domain_lookup.get(feature, "unclassified") for feature in features}
            )
            feature_tables = sorted({table_lookup.get(feature, "") for feature in features})
            if len(feature_domains) < int(tf.get("min_feature_domains", 2)):
                reasons["insufficient_feature_domains"] += 1
                continue
            if len(feature_tables) < int(tf.get("min_feature_tables", 2)):
                reasons["insufficient_feature_tables"] += 1
                continue
            signature = tuple(features)
            if signature in seen_feature_sets:
                continue
            seen_feature_sets.add(signature)
            qualification = _qualify_task(factory, target, features, config)
            if qualification is None:
                reasons["failed_linear_qualification"] += 1
                continue
            accepted_for_target += 1
            task_id = f"task_{len(accepted):05d}"
            accepted.append(
                {
                    "task_id": task_id,
                    "target": target,
                    "target_domain": domain_lookup.get(target, "unclassified"),
                    "target_table": table_lookup.get(target, ""),
                    "features": json.dumps(features),
                    "feature_domains": json.dumps(feature_domains),
                    "feature_tables": json.dumps(feature_tables),
                    "n_features": len(features),
                    "n_feature_domains": len(feature_domains),
                    "n_feature_tables": len(feature_tables),
                    "cross_domain_verified": bool(
                        domain_lookup.get(target, "unclassified") not in feature_domains
                    ),
                    "mean_abs_discovery_correlation": float(
                        np.mean([abs(safe_correlations[name]) for name in features])
                    ),
                    **qualification,
                }
            )

        target_audit.append(
            {
                "target": target,
                "target_domain": domain_lookup.get(target, "unclassified"),
                "target_table": table_lookup.get(target, ""),
                "attempts": attempts,
                "accepted": accepted_for_target,
                "safe_predictors": len(safe_pool),
                **{f"rejected_{key}": int(value) for key, value in sorted(reasons.items())},
            }
        )

    tasks = pd.DataFrame(accepted)
    if tasks.empty:
        raise RuntimeError(
            "No tasks passed linear qualification. Lower min_qualification_r2 or add denser files."
        )
    tasks = _assign_target_splits(tasks, config)
    if bool(tf.get("require_target_feature_domain_difference", True)) and not tasks[
        "cross_domain_verified"
    ].astype(bool).all():
        raise AssertionError("At least one accepted task violates the cross-domain constraint")
    tasks_path.parent.mkdir(parents=True, exist_ok=True)
    tasks.to_csv(tasks_path, index=False)
    pd.DataFrame(target_audit).to_csv(project_paths["tasks"] / "task_factory_audit.csv", index=False)
    pd.DataFrame(leakage_rows).to_csv(project_paths["tasks"] / "leakage_pair_audit.csv", index=False)
    domain_rows: list[dict[str, str]] = []
    for row in tasks.itertuples(index=False):
        for feature_domain in json.loads(str(row.feature_domains)):
            domain_rows.append(
                {
                    "task_id": str(row.task_id),
                    "task_split": str(row.task_split),
                    "target_domain": str(row.target_domain),
                    "feature_domain": str(feature_domain),
                }
            )
    domain_frame = pd.DataFrame(domain_rows)
    domain_matrix = (
        domain_frame.groupby(["task_split", "target_domain", "feature_domain"])
        .size()
        .rename("n_tasks")
        .reset_index()
    )
    domain_matrix.to_csv(project_paths["tasks"] / "cross_domain_task_matrix.csv", index=False)
    partitions = {
        split: sorted(group["target"].unique().tolist())
        for split, group in tasks.groupby("task_split")
    }
    write_json(project_paths["tasks"] / "target_partitions.json", partitions)
    write_json(
        project_paths["audit"] / "task_summary.json",
        {
            "tasks": int(len(tasks)),
            "targets": int(tasks["target"].nunique()),
            "tasks_by_split": tasks["task_split"].value_counts().to_dict(),
            "targets_by_split": tasks.groupby("task_split")["target"].nunique().to_dict(),
            "mean_qualification_r2": float(tasks["qualification_r2"].mean()),
            "target_domains": int(tasks["target_domain"].nunique()),
            "mean_feature_domains_per_task": float(tasks["n_feature_domains"].mean()),
            "all_tasks_cross_domain_verified": bool(tasks["cross_domain_verified"].all()),
        },
    )
    write_json(manifest_path, expected_manifest)
    logger.info("Created %d qualified tasks over %d targets", len(tasks), tasks["target"].nunique())
    return tasks_path


def _assign_target_splits(tasks: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    fractions = config["task_factory"]["target_split_fractions"]
    targets = sorted(tasks["target"].unique().tolist())
    if len(targets) < 3:
        raise RuntimeError("At least three qualified targets are needed for target-disjoint splits")
    rng = np.random.default_rng(int(config["project"]["seed"]) + 911)
    rng.shuffle(targets)
    n_targets = len(targets)
    n_validation = max(1, int(round(n_targets * float(fractions["validation"]))))
    n_test = max(1, int(round(n_targets * float(fractions["test"]))))
    while n_validation + n_test >= n_targets:
        if n_test >= n_validation and n_test > 1:
            n_test -= 1
        elif n_validation > 1:
            n_validation -= 1
        else:
            raise RuntimeError("Could not allocate at least one target to every task split")
    validation_targets = set(targets[:n_validation])
    test_targets = set(targets[n_validation : n_validation + n_test])
    split = np.where(
        tasks["target"].isin(validation_targets),
        "validation",
        np.where(tasks["target"].isin(test_targets), "test", "train"),
    )
    result = tasks.copy()
    result["task_split"] = split
    return result.sort_values(["task_split", "target", "task_id"]).reset_index(drop=True)


def load_tasks(path: Path) -> pd.DataFrame:
    tasks = pd.read_csv(path)
    tasks["feature_list"] = tasks["features"].map(json.loads)
    return tasks
