from __future__ import annotations

import copy
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from nhanes_semantic.config import REPO_ROOT, ensure_project_directories, load_config
from nhanes_semantic.embeddings import generate_embeddings
from nhanes_semantic.evaluation import evaluate_all
from nhanes_semantic.preprocess import preprocess_nhanes
from nhanes_semantic.reporting import generate_report
from nhanes_semantic.representations import build_required_representations, load_representation_archive
from nhanes_semantic.task_factory import build_tasks
from nhanes_semantic.training import select_operator_rank, train_all_embeddings
from nhanes_semantic.utils import setup_logging


def synthetic_inputs(config: dict) -> None:
    paths = ensure_project_directories(config)
    rng = np.random.default_rng(int(config["project"]["seed"]))
    n_rows = 1600
    n_variables = 20
    latent = rng.normal(size=(n_rows, 5))
    loadings = rng.normal(size=(5, n_variables - 1))
    matrix = latent @ loadings + rng.normal(scale=0.35, size=(n_rows, n_variables - 1))
    columns = [f"V{index:03d}" for index in range(n_variables)]
    frame = pd.DataFrame(matrix, columns=columns[:-1])
    frame[columns[-1]] = 2.0 * frame[columns[0]] + 1e-7 * rng.normal(size=n_rows)
    missing = rng.uniform(size=(n_rows, n_variables)) < 0.025
    frame[columns] = frame[columns].mask(missing)
    frame.insert(0, "SEQN", np.arange(100000, 100000 + n_rows))

    catalog_rows = []
    domains = [
        "anthropometry",
        "cardiovascular",
        "glucose_metabolism",
        "hematology",
        "respiratory",
    ]
    for index, variable in enumerate(columns):
        label = f"Synthetic biomarker {index}"
        domain = domains[index % len(domains)]
        file_id = f"SYNTH_{domain.upper()}"
        component = f"Synthetic {domain.replace('_', ' ')} panel"
        catalog_rows.append(
            {
                "variable": variable,
                "sas_label": label,
                "english_text": f"Continuous synthetic health measurement number {index}.",
                "target_population": "Adults and children in the synthetic smoke test.",
                "file_id": file_id,
                "component": component,
                "collection_component": "Synthetic",
                "semantic_domain": domain,
                "domain_rule": "smoke_fixture",
                "missing_codes": "[]",
                "source_url": "local-smoke-fixture",
                "embedding_text": (
                    f"NHANES variable: {variable}. Component: {component}. "
                    f"SAS label: {label}. Official description: Continuous synthetic health "
                    f"measurement number {index}. Eligible population: synthetic participants."
                ),
                "present_in_xpt": True,
            }
        )
    paths["processed"].mkdir(parents=True, exist_ok=True)
    frame.to_pickle(paths["processed"] / "nhanes_wide.pkl")
    pd.DataFrame(catalog_rows).to_csv(
        paths["processed"] / "variable_catalog.csv", index=False
    )


def smoke_config() -> dict:
    config = copy.deepcopy(load_config(REPO_ROOT / "config" / "default.yaml"))
    smoke_root = REPO_ROOT / "smoke_artifacts"
    if smoke_root.exists():
        shutil.rmtree(smoke_root)
    config["project"]["data_dir"] = str(smoke_root / "data")
    config["project"]["output_dir"] = str(smoke_root / "outputs")
    config["project"]["device"] = "cpu"
    config["wikipedia"]["enabled"] = False
    config["wikipedia_adaptation"]["enabled"] = False
    config["rank_selection"]["enabled"] = False
    config["preprocessing"].update(
        {
            "min_observed_fraction": 0.25,
            "min_unique_values": 10,
            "min_factory_observations": 100,
        }
    )
    config["task_factory"].update(
        {
            "tasks_per_target": 2,
            "max_attempts_per_target": 80,
            "max_total_tasks": 36,
            "feature_count_min": 3,
            "feature_count_max": 6,
            "correlation_pool_size": 12,
            "min_pairwise_observations": 75,
            "min_complete_rows": 70,
            "min_qualification_r2": 0.02,
            "min_rmse_improvement_fraction": 0.01,
        }
    )
    config["embeddings"]["models"] = [
        {
            "name": "random_control_32",
            "display_name": "Random control — 32d",
            "kind": "random",
            "dimension": 32,
            "model_family": "random_control_32",
            "domain": "control",
            "text_variant": "control",
            "source_url": "generated-locally",
            "license": "not-applicable",
        }
    ]
    config["training"].update(
        {
            "rank": 8,
            "epochs": 3,
            "batch_size": 256,
            "max_rows_per_task": 120,
            "patience": 3,
            "amp": False,
        }
    )
    config["evaluation"].update({"min_test_rows": 20, "bootstrap_repetitions": 30})
    return config


def main() -> int:
    config = smoke_config()
    paths = ensure_project_directories(config)
    setup_logging(paths["logs"] / "smoke.log")
    synthetic_inputs(config)
    preprocess_nhanes(config, force=True)
    tasks_path = build_tasks(config, force=True)
    embedding_paths = generate_embeddings(config, force=True)
    representations = build_required_representations(
        config, embedding_paths[0], force=True
    )
    train_bundle = load_representation_archive(representations["train"])
    validation_bundle = load_representation_archive(representations["validation"])
    tasks = pd.read_csv(tasks_path)
    print(
        "Non-neural smoke check passed: "
        f"{len(tasks)} tasks, {len(train_bundle['y'])} train examples, "
        f"{len(validation_bundle['y'])} validation examples."
    )

    try:
        import torch  # noqa: F401
    except ImportError:
        print("PyTorch is not installed; neural smoke section was skipped.")
        print("Install the CUDA wheel from README.md and rerun this script on the target machine.")
        return 0

    select_operator_rank(config, force=True)
    train_all_embeddings(config, force=True)
    evaluate_all(config, force=True)
    report_path = generate_report(config, force=True)
    print(f"Neural smoke check passed. Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
