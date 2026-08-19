from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from nhanes_semantic.config import ensure_project_directories, load_config
from nhanes_semantic.nhanes import (
    participant_record_profile,
    prepare_nhanes,
    read_xpt_with_fallback,
)


def _codebook(variable: str) -> bytes:
    return f"""
    <html><body>
    <h3>SEQN - Respondent sequence number</h3>
    <dl><dt>Variable Name:</dt><dd>SEQN</dd>
    <dt>SAS Label:</dt><dd>Respondent sequence number</dd>
    <dt>English Text:</dt><dd>Respondent sequence number.</dd>
    <dt>Target:</dt><dd>All participants</dd></dl>
    <h3>{variable} - Example continuous measurement</h3>
    <dl><dt>Variable Name:</dt><dd>{variable}</dd>
    <dt>SAS Label:</dt><dd>Example continuous measurement</dd>
    <dt>English Text:</dt><dd>An example continuous measurement.</dd>
    <dt>Target:</dt><dd>All participants</dd></dl>
    </body></html>
    """.encode("utf-8")


class NhanesRowGranularityTest(unittest.TestCase):
    def test_xpt_text_falls_back_to_windows_1252(self) -> None:
        raw = pd.DataFrame(
            {"SEQN": [1.0], "NOTE": [b"Participant\x92s measurement"]}
        )
        path = Path("legacy_text.XPT")
        with mock.patch("nhanes_semantic.nhanes.pd.read_sas", return_value=raw) as reader:
            frame, encoding = read_xpt_with_fallback(path)
        reader.assert_called_once_with(path, format="xport", encoding=None)
        self.assertEqual(encoding, "cp1252")
        self.assertEqual(frame.loc[0, "NOTE"], "Participant’s measurement")

    def test_repeated_participant_records_are_detected(self) -> None:
        frame = pd.DataFrame(
            {
                "SEQN": [1, 1, 1, 2, 2],
                "VALUE": [10.0, 11.0, 12.0, 20.0, 21.0],
            }
        )
        profile = participant_record_profile(frame, "SEQN")
        self.assertEqual(profile["row_granularity"], "repeated_record")
        self.assertEqual(profile["unique_participants"], 2)
        self.assertEqual(profile["repeated_rows_beyond_first"], 3)
        self.assertEqual(profile["max_records_per_participant"], 3)

    def test_non_participant_reference_table_is_detected(self) -> None:
        frame = pd.DataFrame(
            {
                "DRXFDCD": [11000000, 11100000],
                "DRXFCSD": ["Milk", "Cheese"],
            }
        )
        profile = participant_record_profile(frame, "SEQN")
        self.assertFalse(profile["participant_id_present"])
        self.assertEqual(profile["row_granularity"], "non_participant_reference")
        self.assertEqual(profile["raw_rows"], 2)

    def test_prepare_excludes_repeated_record_table_and_audits_it(self) -> None:
        base = load_config(Path(__file__).parents[1] / "config" / "default.yaml")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = copy.deepcopy(base)
            config["project"]["data_dir"] = str(root / "data")
            config["project"]["output_dir"] = str(root / "outputs")
            config["nhanes"]["discovery"]["enabled"] = False
            config["nhanes"]["repeated_record_policy"] = "exclude"
            config["nhanes"]["non_participant_table_policy"] = "exclude"
            config["nhanes"]["files"] = [
                {"id": "ONE_J", "component": "One row examination"},
                {"id": "REF_J", "component": "Technical support lookup"},
                {"id": "RPT_J", "component": "Repeated raw examination"},
            ]
            paths = ensure_project_directories(config)
            for file_id, variable in (("ONE_J", "ONEVAL"), ("RPT_J", "RPTVAL")):
                (paths["raw"] / f"{file_id}.XPT").write_bytes(b"fixture")
                (paths["raw"] / f"{file_id}.htm").write_bytes(_codebook(variable))
            (paths["raw"] / "REF_J.XPT").write_bytes(b"fixture")
            # A support file need not expose the standard participant-variable codebook.
            (paths["raw"] / "REF_J.htm").write_bytes(
                b"<html><body><h1>Food-code lookup table</h1></body></html>"
            )

            frames = {
                "ONE_J": pd.DataFrame({"SEQN": [1, 2], "ONEVAL": [1.0, 2.0]}),
                "RPT_J": pd.DataFrame(
                    {"SEQN": [1, 1, 2, 2], "RPTVAL": [3.0, 4.0, 5.0, 6.0]}
                ),
                "REF_J": pd.DataFrame(
                    {"DRXFDCD": [11000000, 11100000], "DRXFCSD": ["Milk", "Cheese"]}
                ),
            }

            def fake_read_sas(path, **_kwargs):
                return frames[Path(path).stem.upper()].copy()

            with mock.patch("nhanes_semantic.nhanes.pd.read_sas", side_effect=fake_read_sas):
                matrix_path, catalog_path = prepare_nhanes(config, force=True)

            matrix = pd.read_pickle(matrix_path)
            self.assertEqual(matrix.columns.tolist(), ["SEQN", "ONEVAL"])
            catalog = pd.read_csv(catalog_path).set_index("variable")
            self.assertFalse(bool(catalog.loc["RPTVAL", "included_in_person_level_matrix"]))
            audit = pd.read_csv(paths["audit"] / "nhanes_table_granularity.csv")
            action = audit.set_index("file_id").loc["RPT_J", "merge_action"]
            self.assertEqual(action, "excluded_repeated_record_table")
            reference_action = audit.set_index("file_id").loc["REF_J", "merge_action"]
            self.assertEqual(reference_action, "excluded_non_participant_table")


if __name__ == "__main__":
    unittest.main()
