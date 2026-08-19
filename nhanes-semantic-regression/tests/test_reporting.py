from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from nhanes_semantic.config import REPO_ROOT, ensure_project_directories, load_config
from nhanes_semantic.reporting import generate_report
from nhanes_semantic.utils import write_json


class ReportingTest(unittest.TestCase):
    def test_publication_report_and_five_figures(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = copy.deepcopy(load_config(REPO_ROOT / "config" / "default.yaml"))
            root = Path(temporary)
            config["project"]["data_dir"] = str(root / "data")
            config["project"]["output_dir"] = str(root / "outputs")
            paths = ensure_project_directories(config)

            definitions = [
                ("semantic_zero_shot__demo__base", "semantic_zero_shot", "demo__base", 0.82),
                ("semantic_zero_shot__demo__adapted", "semantic_zero_shot", "demo__adapted", 0.77),
                ("semantic_zero_shot__random_control_16", "random_control", "random_control_16", 0.97),
                ("zero", "zero_baseline", "", 1.00),
            ]
            summary_rows = []
            for method, family, embedding, rmse in definitions:
                summary_rows.append(
                    {
                        "method": method,
                        "method_family": family,
                        "embedding": embedding,
                        "n_tasks": 2,
                        "n_targets": 2,
                        "n_total_test_rows": 160,
                        "rmse_mean": rmse,
                        "rmse_ci_low": rmse - 0.03,
                        "rmse_ci_high": rmse + 0.03,
                        "mae_mean": rmse * 0.75,
                        "mae_ci_low": rmse * 0.70,
                        "mae_ci_high": rmse * 0.80,
                        "r2_mean": 1.0 - rmse,
                        "r2_ci_low": 0.0,
                        "r2_ci_high": 0.3,
                        "pearson_mean": 0.4,
                        "pearson_ci_low": 0.2,
                        "pearson_ci_high": 0.6,
                    }
                )
            pd.DataFrame(summary_rows).to_csv(paths["metrics"] / "summary_metrics.csv", index=False)

            metric_rows = []
            for target_index, target in enumerate(["TARGET_A", "TARGET_B"]):
                for method, family, embedding, rmse in definitions:
                    metric_rows.append(
                        {
                            "task_id": f"task_{target_index}",
                            "target": target,
                            "method": method,
                            "method_family": family,
                            "embedding": embedding,
                            "n_test": 80,
                            "rmse": rmse + 0.01 * target_index,
                            "mae": rmse * 0.75,
                            "r2": 1.0 - rmse,
                            "pearson": 0.4,
                        }
                    )
            pd.DataFrame(metric_rows).to_csv(paths["metrics"] / "per_task_metrics.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "task_id": "task_0",
                        "target": "TARGET_A",
                        "task_split": "test",
                        "target_domain": "hematology",
                    },
                    {
                        "task_id": "task_1",
                        "target": "TARGET_B",
                        "task_split": "test",
                        "target_domain": "cardiovascular",
                    },
                ]
            ).to_csv(paths["tasks"] / "tasks.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "embedding": "demo__base",
                        "display_name": "Demo BGE — base",
                        "model_family": "demo",
                        "domain": "general",
                        "text_variant": "official",
                        "adaptation": "none",
                        "model_id": "example/demo",
                        "base_model_id": "",
                        "dimension": 16,
                        "source_url": "https://example.test/model",
                        "training_corpus_url": "",
                        "license": "test",
                    },
                    {
                        "embedding": "demo__adapted",
                        "display_name": "Demo BGE — WikiMed adapted",
                        "model_family": "demo",
                        "domain": "wikipedia_adapted",
                        "text_variant": "official",
                        "adaptation": "wikimed_contrastive",
                        "model_id": "models/demo",
                        "base_model_id": "example/demo",
                        "dimension": 16,
                        "source_url": "https://example.test/model",
                        "training_corpus_url": "https://example.test/wiki.zim",
                        "license": "test",
                    },
                    {
                        "embedding": "random_control_16",
                        "display_name": "Random control — 16d",
                        "model_family": "random_control_16",
                        "domain": "control",
                        "text_variant": "control",
                        "adaptation": "none",
                        "model_id": "random",
                        "base_model_id": "",
                        "dimension": 16,
                        "source_url": "generated-locally",
                        "training_corpus_url": "",
                        "license": "not-applicable",
                    },
                ]
            ).to_csv(paths["embeddings"] / "embedding_sources.csv", index=False)
            paired_rows = []
            for index, target in enumerate(["TARGET_A", "TARGET_B"]):
                paired_rows.append(
                    {
                        "task_id": f"task_{index}",
                        "target": target,
                        "pair_type": "encoder_adaptation",
                        "reference_method": "semantic_zero_shot__demo__base",
                        "candidate_method": "semantic_zero_shot__demo__adapted",
                        "reference_embedding": "demo__base",
                        "candidate_embedding": "demo__adapted",
                        "reference_rmse": 0.82,
                        "candidate_rmse": 0.77,
                        "rmse_reference_minus_candidate": 0.05,
                    }
                )
            pd.DataFrame(paired_rows).to_csv(
                paths["metrics"] / "paired_zero_shot_task_deltas.csv", index=False
            )
            pd.DataFrame(
                [
                    {
                        "pair_type": "encoder_adaptation",
                        "n_tasks": 2,
                        "n_targets": 2,
                        "rmse_improvement_mean": 0.05,
                        "rmse_improvement_ci_low": 0.02,
                        "rmse_improvement_ci_high": 0.08,
                        "candidate_target_win_rate": 1.0,
                    }
                ]
            ).to_csv(paths["metrics"] / "paired_zero_shot_summary.csv", index=False)
            rank_dir = paths["checkpoints"] / "rank_selection"
            rank_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                [
                    {"rank": 8, "validation_rmse_mean": 0.91, "validation_rmse_sd": 0.01},
                    {"rank": 16, "validation_rmse_mean": 0.88, "validation_rmse_sd": 0.02},
                    {"rank": 32, "validation_rmse_mean": 0.89, "validation_rmse_sd": 0.01},
                ]
            ).to_csv(rank_dir / "rank_selection_summary.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "task_split": "test",
                        "target_domain": "hematology",
                        "feature_domain": "cardiovascular",
                        "n_tasks": 2,
                    },
                    {
                        "task_split": "test",
                        "target_domain": "cardiovascular",
                        "feature_domain": "anthropometry",
                        "n_tasks": 2,
                    },
                ]
            ).to_csv(paths["tasks"] / "cross_domain_task_matrix.csv", index=False)
            write_json(paths["audit"] / "selected_operator_rank.json", {"selected_rank": 16})

            report = generate_report(config, force=True)
            text = report.read_text(encoding="utf-8")
            self.assertIn("purely zero-shot", text)
            self.assertIn("never appended", text)
            for filename in [
                "zero_shot_rmse_comparison.png",
                "wikimed_adaptation_paired_targets.png",
                "target_domain_improvement_heatmap.png",
                "rank_selection_validation.png",
                "cross_domain_task_coverage.png",
            ]:
                self.assertTrue((paths["figures"] / filename).exists(), filename)


if __name__ == "__main__":
    unittest.main()
