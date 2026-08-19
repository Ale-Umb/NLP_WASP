from __future__ import annotations

import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd
from lxml import html

from .config import ensure_project_directories
from .domains import classify_table_domain
from .utils import get_logger


XPT_RE = re.compile(r"(?P<file>[A-Za-z0-9_]+)\.xpt(?:$|[?#])", flags=re.IGNORECASE)
DOC_RE = re.compile(r"(?P<file>[A-Za-z0-9_]+)\.htm(?:l)?(?:$|[?#])", flags=re.IGNORECASE)


def parse_data_page(
    raw_html: bytes | str,
    *,
    collection_component: str,
    source_url: str,
) -> pd.DataFrame:
    """Parse one public NHANES component page into a reproducible file manifest."""

    document = html.fromstring(raw_html)
    rows: list[dict[str, Any]] = []
    for table_row in document.xpath("//tr"):
        links = table_row.xpath(".//a[@href]")
        xpt_link = None
        file_id = ""
        for link in links:
            href = str(link.get("href", ""))
            match = XPT_RE.search(href)
            if match:
                xpt_link = link
                file_id = match.group("file").upper()
                break
        if xpt_link is None:
            continue

        cells = table_row.xpath("./th|./td")
        title = " ".join(cells[0].itertext()).strip() if cells else file_id
        title = re.sub(r"\s+", " ", title)
        doc_href = ""
        for link in links:
            href = str(link.get("href", ""))
            match = DOC_RE.search(href)
            if match and match.group("file").upper() == file_id:
                doc_href = href
                break
        data_url = urllib.parse.urljoin(source_url, str(xpt_link.get("href", "")))
        documentation_url = urllib.parse.urljoin(source_url, doc_href) if doc_href else ""
        domain, domain_rule = classify_table_domain(file_id, title, collection_component)
        rows.append(
            {
                "file_id": file_id,
                "component": title,
                "collection_component": collection_component,
                "semantic_domain": domain,
                "domain_rule": domain_rule,
                "data_url": data_url,
                "documentation_url": documentation_url,
                "source_page": source_url,
            }
        )
    if not rows:
        raise ValueError(f"No XPT rows were found on NHANES data page: {source_url}")
    return pd.DataFrame(rows).drop_duplicates("file_id", keep="first").reset_index(drop=True)


def _fetch(url: str, timeout: int, retries: int, user_agent: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": user_agent})
    last_error: Exception | None = None
    for _ in range(max(int(retries), 1)):
        try:
            with urllib.request.urlopen(request, timeout=int(timeout)) as response:
                return response.read()
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
    raise RuntimeError(f"Could not retrieve NHANES catalog page {url}") from last_error


def _configured_manifest(config: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in config["nhanes"].get("files", []):
        file_id = str(item["id"]).upper()
        component = str(item["component"])
        collection = str(item.get("collection_component", "Configured"))
        domain, rule = classify_table_domain(file_id, component, collection)
        rows.append(
            {
                "file_id": file_id,
                "component": component,
                "collection_component": collection,
                "semantic_domain": str(item.get("semantic_domain", domain)),
                "domain_rule": str(item.get("domain_rule", rule)),
                "data_url": "",
                "documentation_url": "",
                "source_page": "configured_manifest",
            }
        )
    return pd.DataFrame(rows)


def resolve_file_manifest(config: dict[str, Any], refresh: bool = False) -> pd.DataFrame:
    """Resolve and lock the complete public-file list used by the experiment."""

    logger = get_logger()
    project_paths = ensure_project_directories(config)
    path = project_paths["audit"] / "nhanes_file_manifest.csv"
    discovery = config["nhanes"].get("discovery", {})
    if path.exists() and not refresh:
        manifest = pd.read_csv(path).fillna("")
        required = {
            "file_id",
            "component",
            "collection_component",
            "semantic_domain",
            "domain_rule",
            "data_url",
            "documentation_url",
        }
        if required.issubset(manifest.columns):
            logger.info("Reusing locked NHANES file manifest with %d tables", len(manifest))
            return manifest
        logger.info("Replacing an obsolete NHANES file manifest schema")

    if not bool(discovery.get("enabled", False)):
        manifest = _configured_manifest(config)
    else:
        frames: list[pd.DataFrame] = []
        template = str(discovery["data_page_url"])
        for collection_component in discovery["components"]:
            url = template.format(
                component=urllib.parse.quote(str(collection_component)),
                cycle_begin_year=config["nhanes"]["public_year_path"],
            )
            payload = _fetch(
                url,
                timeout=int(config["nhanes"]["timeout_seconds"]),
                retries=int(config["nhanes"]["retries"]),
                user_agent=str(discovery.get("user_agent", "NHANES-semantic-course-project/4.0")),
            )
            frames.append(
                parse_data_page(
                    payload,
                    collection_component=str(collection_component),
                    source_url=url,
                )
            )
        manifest = pd.concat(frames, ignore_index=True)

    excluded = {str(value).upper() for value in discovery.get("exclude_file_ids", [])}
    manifest = manifest[~manifest["file_id"].astype(str).str.upper().isin(excluded)].copy()
    manifest = manifest.drop_duplicates("file_id", keep="first").sort_values("file_id")
    if manifest.empty:
        raise RuntimeError("The resolved NHANES file manifest is empty")
    if manifest["documentation_url"].astype(str).eq("").any() and bool(
        discovery.get("enabled", False)
    ):
        missing = manifest.loc[manifest["documentation_url"].eq(""), "file_id"].tolist()
        raise RuntimeError(f"Public NHANES tables without documentation links: {missing[:10]}")
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(path, index=False)
    logger.info(
        "Locked %d public NHANES tables across %d collection components",
        len(manifest),
        manifest["collection_component"].nunique(),
    )
    return manifest.reset_index(drop=True)
