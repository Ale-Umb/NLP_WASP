from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from nhanes_semantic.config import REPO_ROOT, ensure_project_directories, load_config
from nhanes_semantic.embeddings import _catalog_for_models, _texts_for_spec


class EmbeddingCatalogTest(unittest.TestCase):
    def test_every_encoder_uses_official_text_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = copy.deepcopy(load_config(REPO_ROOT / "config" / "default.yaml"))
            root = Path(temporary)
            config["project"]["data_dir"] = str(root / "data")
            config["project"]["output_dir"] = str(root / "outputs")
            paths = ensure_project_directories(config)
            pd.DataFrame(
                [
                    {
                        "variable": "LBXTC",
                        "embedding_text": "Official cholesterol description.",
                        "official_embedding_text": "Official cholesterol description.",
                    }
                ]
            ).to_csv(paths["processed"] / "eligible_catalog.csv", index=False)
            catalog = _catalog_for_models(config, paths)
            for spec in config["embeddings"]["models"]:
                self.assertNotEqual(spec.get("text_variant"), "wikipedia")
                self.assertEqual(
                    _texts_for_spec(catalog, spec), ["Official cholesterol description."]
                )


if __name__ == "__main__":
    unittest.main()
