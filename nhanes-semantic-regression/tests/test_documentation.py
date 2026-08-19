from __future__ import annotations

import unittest

from nhanes_semantic.documentation import parse_codebook_html


FIXTURE = b"""
<html><body>
<h3>Codebook and Frequencies</h3>
<h3>LBXTEST - Example serum test</h3>
<dl>
  <dt>Variable Name:</dt><dd>LBXTEST</dd>
  <dt>SAS Label:</dt><dd>Example serum test (mg/dL)</dd>
  <dt>English Text:</dt><dd>Measured example concentration in serum.</dd>
  <dt>Target:</dt><dd>Both males and females 12 YEARS - 150 YEARS</dd>
</dl>
<table>
<tr><th>Code or Value</th><th>Value Description</th></tr>
<tr><td>7777</td><td>Refused</td></tr>
<tr><td>9999</td><td>Don't know</td></tr>
</table>
<h3>LBXOTHER - Other test</h3>
<dl>
  <dt>Variable Name:</dt><dd>LBXOTHER</dd>
  <dt>SAS Label:</dt><dd>Other test</dd>
  <dt>English Text:</dt><dd>Another continuous measurement.</dd>
  <dt>Target:</dt><dd>Adults</dd>
</dl>
<h3>LBXFINAL - Final test</h3>
<dl>
  <dt>Variable Name:</dt><dd>LBXFINAL</dd>
  <dt>SAS Label:</dt><dd>Final test</dd>
  <dt>English Text:</dt><dd>Final continuous measurement.</dd>
  <dt>Target:</dt><dd>Adults</dd>
</dl>
</body></html>
"""


class DocumentationParserTest(unittest.TestCase):
    def test_parses_codebook_fields(self) -> None:
        frame = parse_codebook_html(FIXTURE, "TEST_J", "Test component", "https://example.test")
        self.assertEqual(frame["variable"].tolist(), ["LBXTEST", "LBXOTHER", "LBXFINAL"])
        row = frame.set_index("variable").loc["LBXTEST"]
        self.assertEqual(row["sas_label"], "Example serum test (mg/dL)")
        self.assertIn("Measured example concentration", row["english_text"])
        self.assertIn("Test component", row["embedding_text"])
        self.assertNotIn("Value Description", row["target_population"])
        self.assertEqual(row["missing_codes"], "[7777.0, 9999.0]")
        self.assertEqual(frame.set_index("variable").loc["LBXOTHER", "target_population"], "Adults")


if __name__ == "__main__":
    unittest.main()
