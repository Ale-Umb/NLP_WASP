from __future__ import annotations

import unittest

from nhanes_semantic.domains import classify_table_domain
from nhanes_semantic.nhanes_catalog import parse_data_page


DATA_PAGE = b"""
<html><body><table>
<tr><th>Data File Name</th><th>Doc File</th><th>Data File</th></tr>
<tr><td>Complete Blood Count with 5-Part Differential</td>
<td><a href='/Nchs/Data/Nhanes/Public/2017/DataFiles/CBC_J.htm'>CBC_J Doc</a></td>
<td><a href='/Nchs/Data/Nhanes/Public/2017/DataFiles/CBC_J.XPT'>CBC_J Data</a></td></tr>
<tr><td>Withdrawn file</td><td>No documentation</td><td>No data</td></tr>
</table></body></html>
"""


class DomainAndCatalogTest(unittest.TestCase):
    def test_public_data_page_is_parsed_and_domain_labeled(self) -> None:
        frame = parse_data_page(
            DATA_PAGE,
            collection_component="Laboratory",
            source_url="https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx",
        )
        self.assertEqual(frame["file_id"].tolist(), ["CBC_J"])
        self.assertEqual(frame.iloc[0]["semantic_domain"], "hematology")
        self.assertTrue(frame.iloc[0]["data_url"].endswith("CBC_J.XPT"))

    def test_domain_rules_are_deterministic(self) -> None:
        first = classify_table_domain("BMX_J", "Body Measures", "Examination")
        second = classify_table_domain("BMX_J", "Body Measures", "Examination")
        self.assertEqual(first, second)
        self.assertEqual(first[0], "anthropometry")


if __name__ == "__main__":
    unittest.main()
