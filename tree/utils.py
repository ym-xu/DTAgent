import json
import logging
import os
import re
from html.parser import HTMLParser
from typing import Any, List, Optional, Tuple


logger = logging.getLogger(__name__)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def try_ocr_image(path: str) -> Optional[str]:
    """
    Interface: Best-effort OCR using pytesseract if available. Returns None on failure.
    The adapter does not call this by default; pass it explicitly if needed.
    """
    try:
        from PIL import Image
        try:
            import pytesseract
        except Exception:
            logger.info("pytesseract not available; skip OCR for %s", path)
            return None

        img = Image.open(path)
        text = pytesseract.image_to_string(img)
        return text.strip() or None
    except Exception as e:
        logger.info("OCR failed for %s: %s", path, e)
        return None


def describe_image_with_llm(image_path: str, prompt: Optional[str] = None, **kwargs) -> Optional[str]:
    """
    Interface: LLM-based image description hook.
    Implement with your preferred VLM/LLM client and return a short description.
    Returning None keeps the field absent.
    """
    logger.info("LLM description not implemented for %s", image_path)
    return None


class _SimpleTableParser(HTMLParser):
    """
    Minimal HTML table parser to extract rows for Markdown rendering.
    Handles <table>, <thead>, <tbody>, <tr>, <th>, <td>, <br>.
    Colspan/rowspan are ignored.
    """

    def __init__(self) -> None:
        super().__init__()
        self.in_table = False
        self.in_thead = False
        self.in_row = False
        self.in_cell = False
        self.current_cell_is_header = False
        self.current_cell_text: List[str] = []
        self.current_row: List[str] = []
        self.current_row_is_header = False
        self.rows: List[List[str]] = []
        self.row_is_header: List[bool] = []
        self._done = False  # stop after first table

    def handle_starttag(self, tag: str, attrs):  # type: ignore[override]
        if self._done:
            return
        if tag == "table":
            if not self.in_table:
                self.in_table = True
            return
        if not self.in_table:
            return
        if tag == "thead":
            self.in_thead = True
        elif tag == "tr":
            self._start_row()
        elif tag in ("td", "th"):
            self._start_cell(is_header=(tag == "th") or self.in_thead)
        elif tag == "br":
            if self.in_cell:
                self.current_cell_text.append("\n")

    def handle_endtag(self, tag: str):  # type: ignore[override]
        if self._done:
            return
        if tag == "table" and self.in_table:
            # finish the last row/cell if still open
            if self.in_cell:
                self._end_cell()
            if self.in_row:
                self._end_row()
            self.in_table = False
            self._done = True
        if not self.in_table:
            return
        if tag == "thead":
            self.in_thead = False
        elif tag in ("td", "th") and self.in_cell:
            self._end_cell()
        elif tag == "tr" and self.in_row:
            self._end_row()

    def handle_data(self, data: str):  # type: ignore[override]
        if self.in_cell and data:
            self.current_cell_text.append(data)

    def _start_row(self) -> None:
        self.in_row = True
        self.current_row = []
        self.current_row_is_header = False

    def _end_row(self) -> None:
        # push row (cells are already normalized on cell end)
        self.rows.append(self.current_row)
        self.row_is_header.append(self.current_row_is_header)
        self.in_row = False
        self.current_row = []
        self.current_row_is_header = False

    def _start_cell(self, is_header: bool) -> None:
        self.in_cell = True
        self.current_cell_is_header = is_header
        self.current_cell_text = []

    def _end_cell(self) -> None:
        text = self._normalize_text("".join(self.current_cell_text))
        self.current_row.append(text)
        if self.current_cell_is_header:
            self.current_row_is_header = True
        self.in_cell = False
        self.current_cell_is_header = False
        self.current_cell_text = []

    @staticmethod
    def _normalize_text(t: str) -> str:
        # Collapse whitespace but preserve newlines introduced via <br>
        t = t.replace("\r", "").strip()
        # Replace internal runs of spaces/tabs with single space, but keep line breaks
        parts = [re.sub(r"[ \t\f\v]+", " ", line.strip()) for line in t.split("\n")]
        return "<br>".join([p for p in parts if p])


def _escape_md_cell(text: str) -> str:
    # Escape pipes to avoid breaking the table; keep <br> as line break
    return text.replace("|", r"\|")


def _rows_to_markdown(rows: List[List[str]], header_flags: List[bool]) -> Optional[str]:
    if not rows:
        return None
    # Determine header row
    header_idx = next((i for i, is_h in enumerate(header_flags) if is_h), 0)
    num_cols = max((len(r) for r in rows), default=0)
    if num_cols == 0:
        return None
    # Normalize row lengths
    norm = [r + [""] * (num_cols - len(r)) for r in rows]

    header = [_escape_md_cell(c) or f"col{j+1}" for j, c in enumerate(norm[header_idx])]
    sep = ["---"] * num_cols
    data_rows: List[List[str]] = [r for i, r in enumerate(norm) if i != header_idx]
    data_rows = [[_escape_md_cell(c) for c in r] for r in data_rows]

    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(sep) + " |",
    ]
    for r in data_rows:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)


def html_table_to_markdown(html_str: str) -> Optional[str]:
    """
    Convert a single HTML <table> to Markdown table text when possible.
    Returns None if no rows detected. On failure, returns a plain-text fallback.
    """
    try:
        parser = _SimpleTableParser()
        parser.feed(html_str)
        md = _rows_to_markdown(parser.rows, parser.row_is_header)
        if md:
            return md
    except Exception as e:
        logger.info("HTML→Markdown table parse failed: %s", e)

    # Fallback: strip tags and collapse whitespace as plain text
    try:
        text = re.sub(r"<[^>]+>", " ", html_str)
        text = re.sub(r"\s+", " ", text).strip()
        return text or None
    except Exception:
        return None
