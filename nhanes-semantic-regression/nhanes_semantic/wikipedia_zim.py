from __future__ import annotations

import re
import time
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .config import ensure_project_directories
from .documentation import clean_text
from .utils import get_logger


CITATION_RE = re.compile(r"\[(?:\d+|citation needed|clarification needed)[^]]*\]", re.I)
HIDDEN_HTML_TAGS = {
    "aside",
    "figcaption",
    "figure",
    "footer",
    "header",
    "math",
    "nav",
    "noscript",
    "script",
    "style",
    "sup",
    "table",
}


class _ParagraphParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.hidden_depth = 0
        self.paragraph_depth = 0
        self.current: list[str] = []
        self.paragraphs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        tag = tag.lower()
        if tag in HIDDEN_HTML_TAGS:
            self.hidden_depth += 1
            return
        if self.hidden_depth:
            return
        if tag == "p":
            self.paragraph_depth += 1
            if self.paragraph_depth == 1:
                self.current = []
        elif tag == "br" and self.paragraph_depth:
            self.current.append(" ")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in HIDDEN_HTML_TAGS and self.hidden_depth:
            self.hidden_depth -= 1
            return
        if self.hidden_depth:
            return
        if tag == "p" and self.paragraph_depth:
            self.paragraph_depth -= 1
            if self.paragraph_depth == 0:
                paragraph = clean_text(" ".join(self.current))
                if paragraph:
                    self.paragraphs.append(paragraph)
                self.current = []

    def handle_data(self, data: str) -> None:
        if self.hidden_depth or not self.paragraph_depth:
            return
        value = data.strip()
        if value:
            self.current.append(value)


def plain_text_paragraphs(
    html: str,
    *,
    min_characters: int = 120,
    max_characters: int = 900,
    maximum: int = 3,
) -> list[str]:
    parser = _ParagraphParser()
    parser.feed(html)
    parser.close()
    result: list[str] = []
    for paragraph in parser.paragraphs:
        value = clean_text(CITATION_RE.sub(" ", paragraph))
        if len(value) < int(min_characters):
            continue
        if len(value) > int(max_characters):
            prefix = value[: int(max_characters) + 1]
            sentence_end = max(prefix.rfind(". "), prefix.rfind("! "), prefix.rfind("? "))
            value = (
                prefix[: sentence_end + 1]
                if sentence_end >= int(min_characters)
                else prefix.rsplit(" ", 1)[0]
            )
        if value and value not in result:
            result.append(value)
        if len(result) >= int(maximum):
            break
    return result


def plain_text_extract(html: str, max_characters: int, max_paragraphs: int = 3) -> str:
    paragraphs = plain_text_paragraphs(
        html,
        min_characters=1,
        max_characters=max_characters,
        maximum=max_paragraphs,
    )
    text = clean_text(" ".join(paragraphs))
    return text[: int(max_characters)].strip()


def _snapshot_path(settings: dict[str, Any], project_paths: dict[str, Path]) -> Path:
    configured = Path(str(settings["zim_path"])).expanduser()
    return configured if configured.is_absolute() else project_paths["raw"] / configured


def _response_status(response: Any) -> int:
    status = getattr(response, "status", None)
    if status is None and hasattr(response, "getcode"):
        status = response.getcode()
    return int(status or 200)


def download_snapshot(
    url: str,
    destination: Path,
    *,
    expected_bytes: int = 0,
    timeout_seconds: float = 180.0,
    retries: int = 4,
    user_agent: str = "NHANESSemanticRegression/4.0 (educational course project)",
) -> bool:
    """Download one immutable ZIM, resuming an existing .part file when possible."""

    logger = get_logger()
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected = max(int(expected_bytes), 0)
    if destination.exists():
        actual = destination.stat().st_size
        if expected and actual != expected:
            raise RuntimeError(
                f"Existing Wikipedia snapshot has {actual} bytes, expected {expected}: "
                f"{destination}. Remove or rename that single file, then rerun adaptation."
            )
        return False

    partial = destination.with_suffix(destination.suffix + ".part")
    transferred = False
    for attempt in range(int(retries) + 1):
        start = partial.stat().st_size if partial.exists() else 0
        headers = {"User-Agent": user_agent, "Accept-Encoding": "identity"}
        if start:
            headers["Range"] = f"bytes={start}-"
        try:
            with urlopen(Request(url, headers=headers), timeout=float(timeout_seconds)) as response:
                status = _response_status(response)
                append = bool(start and status == 206)
                if start and not append:
                    logger.info("Snapshot server did not resume; restarting the partial file")
                    start = 0
                downloaded = start
                next_log = downloaded + 25 * 1024 * 1024
                with partial.open("ab" if append else "wb") as handle:
                    while True:
                        chunk = response.read(4 * 1024 * 1024)
                        if not chunk:
                            break
                        handle.write(chunk)
                        downloaded += len(chunk)
                        transferred = True
                        if downloaded >= next_log:
                            logger.info("Wikipedia snapshot download: %.1f MiB", downloaded / 2**20)
                            next_log = downloaded + 25 * 1024 * 1024
            actual = partial.stat().st_size
            if expected and actual != expected:
                raise OSError(f"snapshot has {actual} bytes, expected {expected}")
            partial.replace(destination)
            logger.info("Wikipedia snapshot ready: %s", destination)
            return transferred
        except HTTPError as exc:
            if exc.code == 416 and partial.exists() and expected and partial.stat().st_size == expected:
                partial.replace(destination)
                return transferred
            last_error: Exception = exc
        except (URLError, TimeoutError, OSError) as exc:
            last_error = exc
        if attempt >= int(retries):
            raise RuntimeError(
                f"Could not download the fixed Wikipedia snapshot. The partial file remains "
                f"at {partial}; rerun to resume."
            ) from last_error
        wait = min(2.0**attempt, 15.0)
        logger.warning("Snapshot download interrupted; retrying in %.1f seconds", wait)
        time.sleep(wait)
    raise AssertionError("unreachable")


def ensure_snapshot(
    config: dict[str, Any], project_paths: dict[str, Path] | None = None
) -> tuple[Path, bool]:
    project_paths = project_paths or ensure_project_directories(config)
    settings = config["wikipedia"]
    path = _snapshot_path(settings, project_paths)
    if path.exists():
        expected = int(settings.get("zim_expected_bytes", 0))
        if expected and path.stat().st_size != expected:
            raise RuntimeError(
                f"Wikipedia snapshot size mismatch at {path}: found {path.stat().st_size}, "
                f"expected {expected}."
            )
        return path, False
    if not bool(settings.get("auto_download", True)):
        raise FileNotFoundError(
            f"Offline Wikipedia snapshot is missing: {path}. Download {settings['zim_url']} "
            "there or enable wikipedia.auto_download."
        )
    downloaded = download_snapshot(
        str(settings["zim_url"]),
        path,
        expected_bytes=int(settings.get("zim_expected_bytes", 0)),
        timeout_seconds=float(settings.get("download_timeout_seconds", 180)),
        retries=int(settings.get("download_retries", 4)),
        user_agent=str(settings.get("user_agent", "NHANESSemanticRegression/4.0")),
    )
    return path, downloaded


class OfflineWikipediaSnapshot:
    def __init__(self, path: Path, *, verify_checksum: bool = True) -> None:
        try:
            from libzim.reader import Archive
        except ImportError as exc:
            raise RuntimeError(
                "Offline WikiMed adaptation requires libzim. Install requirements.txt."
            ) from exc
        self.path = Path(path)
        try:
            self.archive = Archive(self.path)
        except Exception as exc:
            raise RuntimeError(f"Could not open Wikipedia ZIM snapshot {self.path}: {exc}") from exc
        if verify_checksum and bool(self.archive.has_checksum) and not self.archive.check():
            raise RuntimeError(f"Wikipedia ZIM checksum verification failed: {self.path}")

    def metadata(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "filename": self.path.name,
            "bytes": int(self.path.stat().st_size),
            "uuid": str(self.archive.uuid),
            "article_count": int(self.archive.article_count),
            "all_entry_count": int(self.archive.all_entry_count),
            "has_checksum": bool(self.archive.has_checksum),
            "checksum": str(self.archive.checksum) if self.archive.has_checksum else "",
        }
        available = set(self.archive.metadata_keys)
        for key in ["Name", "Title", "Description", "Creator", "Publisher", "Date", "Language"]:
            if key not in available:
                continue
            try:
                result[key.lower()] = self.archive.get_metadata(key).decode(
                    "utf-8", errors="replace"
                )
            except Exception:
                continue
        return result
