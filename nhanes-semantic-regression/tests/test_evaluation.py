from __future__ import annotations

import copy
import unittest

import pandas as pd

from nhanes_semantic.config import REPO_ROOT, load_config
from nhanes_semantic.evaluation import _paired_zero_shot_deltas, _summarize_paired_deltas


class PairedEvaluationTest(unittest.TestCase):
    def test_adaptation_and_dimension_matched_pairs_are_target_clustered(self) -> None:
        config = copy.deepcopy(load_config(REPO_ROOT / "config" / "default.yaml"))
        config["embeddings"]["models"] = [
            {
                "name": "demo__base",
                "kind": "sentence_transformer",
                "model_family": "demo",
                "text_variant": "official",
                "adaptation": "none",
                "dimension": 16,
            },
            {
                "name": "demo__adapted",
                "kind": "adapted_sentence_transformer",
                "model_family": "demo",
                "text_variant": "official",
                "adaptation": "wikimed_contrastive",
                "dimension": 16,
            },
            {"name": "random_16", "kind": "random", "dimension": 16},
        ]
        rows = []
        methods = {
            "semantic_zero_shot__demo__base": ("demo__base", 0.8),
            "semantic_zero_shot__demo__adapted": ("demo__adapted", 0.7),
            "semantic_zero_shot__random_16": ("random_16", 0.95),
            "zero": ("", 1.0),
        }
        for target in ["A", "B"]:
            for task_index in range(2):
                for method, (embedding, rmse) in methods.items():
                    rows.append(
                        {
                            "task_id": f"{target}_{task_index}",
                            "target": target,
                            "method": method,
                            "embedding": embedding,
                            "rmse": rmse,
                        }
                    )
        paired = _paired_zero_shot_deltas(pd.DataFrame(rows), config)
        summary = _summarize_paired_deltas(paired, config)
        adaptation = summary[summary["pair_type"] == "encoder_adaptation"].iloc[0]
        self.assertEqual(adaptation["n_targets"], 2)
        self.assertAlmostEqual(adaptation["rmse_improvement_mean"], 0.1)
        self.assertEqual(adaptation["candidate_target_win_rate"], 1.0)
        self.assertIn("dimension_matched_random", set(summary["pair_type"]))


if __name__ == "__main__":
    unittest.main()
