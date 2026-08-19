from __future__ import annotations

import json
import re
from typing import Any

import pandas as pd
from lxml import html


SPACE_RE = re.compile(r"\s+")


def clean_text(value: str | None) -> str:
    if not value:
        return ""
    return SPACE_RE.sub(" ", value.replace("\xa0", " ")).strip()


def parse_codebook_html(
    raw_html: bytes | str,
    file_id: str,
    component: str,
    source_url: str,
    collection_component: str = "",
    semantic_domain: str = "",
    domain_rule: str = "",
) -> pd.DataFrame:
    """Parse the stable Variable Name/SAS Label/English Text blocks in NCHS HTML."""
    document = html.fromstring(raw_html)
    full_text = clean_text(document.text_content())
    variable_headings = []
    for heading in document.xpath("//h3 | //h4"):
        title = clean_text(heading.text_content())
        if re.match(r"^[A-Za-z][A-Za-z0-9_]{0,31}\s+-\s+", title):
            variable_headings.append(title)
    missing_codes = _missing_codes_by_variable(document)
    chunks = re.split(r"(?=\bVariable Name:\s*)", full_text)
    rows: list[dict[str, Any]] = []
    for chunk in chunks:
        if not chunk.startswith("Variable Name:"):
            continue
        variable = _extract(chunk, r"Variable Name:\s*([A-Za-z][A-Za-z0-9_]{0,31})")
        if not variable:
            continue
        sas_label = _extract_between(chunk, "SAS Label:", "English Text:")
        english_text = _extract_between(chunk, "English Text:", "Target:")
        target = _extract_after_target(chunk, variable_headings)
        rows.append(
            {
                "variable": variable.upper(),
                "sas_label": sas_label,
                "english_text": english_text,
                "target_population": target,
                "file_id": file_id,
                "component": component,
                "collection_component": collection_component,
                "semantic_domain": semantic_domain,
                "domain_rule": domain_rule,
                "source_url": source_url,
                "missing_codes": json.dumps(missing_codes.get(variable.upper(), [])),
            }
        )
    if not rows:
        raise ValueError(f"No codebook variables parsed from {file_id}: {source_url}")
    frame = pd.DataFrame(rows).drop_duplicates("variable", keep="first")
    frame["embedding_text"] = frame.apply(build_embedding_text, axis=1)
    return frame.reset_index(drop=True)


def _missing_codes_by_variable(document) -> dict[str, list[float]]:
    """Extract documented refused/don't-know/missing sentinels from codebook tables."""

    result: dict[str, list[float]] = {}
    missing_terms = re.compile(
        r"\b(?:refused|don['’]?t know|missing|not ascertained|unknown)\b", re.IGNORECASE
    )
    numeric_code = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$")
    for heading in document.xpath("//h3 | //h4"):
        title = clean_text(heading.text_content())
        match = re.match(r"^([A-Za-z][A-Za-z0-9_]{0,31})\s+-\s+", title)
        if not match:
            continue
        variable = match.group(1).upper()
        codes: list[float] = []
        for sibling in heading.itersiblings():
            if str(getattr(sibling, "tag", "")).lower() in {"h3", "h4"}:
                break
            table_rows = [sibling] if str(getattr(sibling, "tag", "")).lower() == "tr" else sibling.xpath(".//tr")
            for table_row in table_rows:
                cells = [clean_text(" ".join(cell.itertext())) for cell in table_row.xpath("./th|./td")]
                if len(cells) < 2 or not missing_terms.search(cells[1]):
                    continue
                raw_code = cells[0].strip()
                if numeric_code.fullmatch(raw_code):
                    value = float(raw_code)
                    if value not in codes:
                        codes.append(value)
        result[variable] = sorted(codes)
    return result


def _extract(text: str, pattern: str) -> str:
    match = re.search(pattern, text)
    return clean_text(match.group(1)) if match else ""


def _extract_between(text: str, start: str, end: str) -> str:
    pattern = re.escape(start) + r"\s*(.*?)\s*" + re.escape(end)
    match = re.search(pattern, text)
    return clean_text(match.group(1)) if match else ""


def _extract_after_target(text: str, variable_headings: list[str]) -> str:
    # NCHS table header cells may become either "Code or Value" or
    # "Code or ValueValue Description" after HTML text extraction.
    match = re.search(r"Target:\s*(.*?)(?=\s*Code\s+or\s+Value|$)", text)
    value = clean_text(match.group(1)) if match else ""
    # Variables without a frequency table run directly into the next h3 title.
    # Use actual document headings instead of a generic dash regex, which would
    # incorrectly split target ranges such as "0 YEARS - 150 YEARS".
    positions = [value.find(title) for title in variable_headings if value.find(title) >= 0]
    if positions:
        value = clean_text(value[: min(positions)])
    return value


def build_embedding_text(row: pd.Series | dict[str, Any]) -> str:
    return clean_text(
        " ".join(
            [
                f"NHANES variable: {row['variable']}.",
                f"Component: {row.get('component', '')}.",
                f"SAS label: {row.get('sas_label', '')}.",
                f"Official description: {row.get('english_text', '')}.",
                f"Eligible population: {row.get('target_population', '')}.",
            ]
        )
    )
