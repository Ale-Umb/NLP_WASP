from __future__ import annotations

import unittest

from nhanes_semantic.wiki_adaptation import article_pairs
from nhanes_semantic.wikipedia_zim import plain_text_paragraphs


class WikiAdaptationTest(unittest.TestCase):
    def test_external_article_pairs_do_not_require_nhanes_text(self) -> None:
        html = (
            "<html><body><p>Glucose is a monosaccharide used by organisms as an important "
            "source of metabolic energy. It circulates in blood and participates in many "
            "biochemical pathways throughout the body.</p>"
            "<p>Clinical measurements of glucose are commonly used when assessing metabolic "
            "health and disorders affecting regulation of blood sugar over time.</p></body></html>"
        )
        paragraphs = plain_text_paragraphs(
            html, min_characters=80, max_characters=500, maximum=3
        )
        pairs = article_pairs("Glucose", paragraphs, article_path="A/Glucose")
        self.assertEqual([row["pair_type"] for row in pairs], ["title_to_lead", "adjacent_paragraphs"])
        self.assertNotIn("NHANES", " ".join(row["anchor"] + row["positive"] for row in pairs))

    def test_disambiguation_is_rejected(self) -> None:
        self.assertEqual(
            article_pairs(
                "Iron (disambiguation)",
                ["Iron may refer to several unrelated topics in chemistry and culture."],
                article_path="A/Iron",
            ),
            [],
        )


if __name__ == "__main__":
    unittest.main()
