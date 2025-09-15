import json
import logging
import os
import re
from html.parser import HTMLParser
from typing import Any, List, Optional, Tuple
import glob
try:
    import fitz  # type: ignore
except Exception:  # runtime import later
    fitz = None  # type: ignore
import unicodedata


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


# === Text sanitization utilities ===
_RE_ZERO_WIDTH = re.compile(r"[\u200B-\u200D\uFEFF]")
_RE_CTRL = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]")


def sanitize_text(text: Optional[str]) -> Optional[str]:
    """
    Best-effort text cleanup for noisy OCR/MinerU artifacts.
    - Unicode NFKC normalization
    - Remove zero-width chars and BOM
    - Remove control chars (keep \n and \t)
    - Normalize whitespace (collapse runs; keep newlines; cap empty lines)
    - Drop long runs of single-letter tokens (>=4 in a row) within a line
    Returns cleaned text or original None.
    """
    if text is None:
        return None

    s = unicodedata.normalize("NFKC", text)
    s = s.replace("\r", "\n")
    s = _RE_ZERO_WIDTH.sub("", s)
    # Remove control characters except \n and \t
    s = _RE_CTRL.sub("", s)

    # Process line by line to preserve newlines
    lines = s.split("\n")
    cleaned_lines: List[str] = []
    for line in lines:
        # Collapse spaces/tabs within the line
        line = re.sub(r"[ \t\f\v]+", " ", line).strip()

        # Remove runs of single-letter tokens (Latin) of length >= 4
        tokens = line.split()
        new_tokens: List[str] = []
        i = 0
        while i < len(tokens):
            # Rule A: run of single-letter alphabetic tokens
            j = i
            while j < len(tokens) and len(tokens[j]) == 1 and tokens[j].isalpha():
                j += 1
            if j - i >= 4:
                i = j
                continue

            # Rule B: run of identical token (digits or punctuation-like), length >= 5
            tok = tokens[i]
            k = i
            while k < len(tokens) and tokens[k] == tok:
                k += 1
            is_punct_like = not re.search(r"[A-Za-z0-9]", tok or "")
            if k - i >= 5 and (tok.isdigit() or is_punct_like or (len(tok) == 1 and not tok.isalpha())):
                i = k
                continue

            # Keep current token
            new_tokens.append(tok)
            i += 1

        line = " ".join(new_tokens).strip()
        cleaned_lines.append(line)

    # Collapse multiple empty lines → at most 2
    out: List[str] = []
    empty_count = 0
    for ln in cleaned_lines:
        if ln:
            out.append(ln)
            empty_count = 0
        else:
            if empty_count < 2:
                out.append("")
            empty_count += 1

    cleaned = "\n".join(out).strip()
    return cleaned


def sanitize_item_fields(item: dict) -> None:
    """In-place sanitize textual fields based on node type."""
    typ = item.get("type")
    if typ == "text" or typ == "equation":
        if isinstance(item.get("text"), str):
            item["text"] = sanitize_text(item.get("text")) or ""
    elif typ == "image":
        if isinstance(item.get("text"), str):
            item["text"] = sanitize_text(item.get("text")) or ""
        if isinstance(item.get("description"), str):
            item["description"] = sanitize_text(item.get("description")) or ""
        # Ensure caption/footnote lists exist and sanitize entries
        cap = item.get("image_caption")
        if not isinstance(cap, list):
            item["image_caption"] = []
        else:
            item["image_caption"] = _sanitize_string_list(cap)
        fn = item.get("image_footnote")
        if not isinstance(fn, list):
            item["image_footnote"] = []
        else:
            item["image_footnote"] = _sanitize_string_list(fn)
    else:
        # For other types we keep as-is; do not touch table_body HTML etc.
        # But for table, normalize caption/footnote lists if present
        if typ == "table":
            cap = item.get("table_caption")
            if not isinstance(cap, list):
                item["table_caption"] = []
            else:
                item["table_caption"] = _sanitize_string_list(cap)
            fn = item.get("table_footnote")
            if not isinstance(fn, list):
                item["table_footnote"] = []
            else:
                item["table_footnote"] = _sanitize_string_list(fn)


def sanitize_content_list(items: List[dict]) -> List[dict]:
    out: List[dict] = []
    for it in items:
        obj = dict(it)
        try:
            sanitize_item_fields(obj)
        except Exception:
            # Failsafe: keep original if any error occurs
            pass
        out.append(obj)
    return out


# === Page image helpers ===
def page_images_dir(source_dir: str) -> str:
    return os.path.join(source_dir or ".", "images")


def find_existing_page_image(source_dir: str, page_idx: int) -> Optional[str]:
    """
    Look for an existing page image in <source_dir>/images using common patterns.
    Primary pattern per user spec: page_{idx}.(png|jpg|jpeg). Also try 1-based.
    """
    img_dir = page_images_dir(source_dir)
    if not os.path.isdir(img_dir):
        return None
    patterns = []
    # 0-based
    for ext in ("png", "jpg", "jpeg"):
        patterns.append(os.path.join(img_dir, f"page_{page_idx}.{ext}"))
    # 1-based fallback
    one_based = page_idx + 1
    for ext in ("png", "jpg", "jpeg"):
        patterns.append(os.path.join(img_dir, f"page_{one_based}.{ext}"))
    for p in patterns:
        if os.path.exists(p):
            return p
    # also try subfolder patterns like images/pages/page_{n}.*
    for n in (page_idx, page_idx + 1):
        for ext in ("png", "jpg", "jpeg"):
            for pat in (
                os.path.join(img_dir, "pages", f"page_{n}.{ext}"),
                os.path.join(img_dir, "pages", f"{n:03d}.{ext}"),
            ):
                if os.path.exists(pat):
                    return pat
    return None


def find_pdf_in_dir(source_dir: str) -> Optional[str]:
    pdfs = glob.glob(os.path.join(source_dir or ".", "*.pdf"))
    return pdfs[0] if pdfs else None


def render_pdf_page_to_image(pdf_path: str, page_idx: int, out_dir: Optional[str] = None, *, dpi: int = 150) -> Optional[str]:
    """
    Render one PDF page to <out_dir>/images/page_{idx}.png (0-based idx by user spec). Returns saved path or None.
    """
    try:
        import fitz  # type: ignore
    except Exception:
        logger.info("PyMuPDF not available; cannot render page %s from %s", page_idx, pdf_path)
        return None
    try:
        doc = fitz.open(pdf_path)
        if page_idx < 0 or page_idx >= len(doc):
            return None
        page = doc.load_page(page_idx)
        mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        out_base = out_dir or page_images_dir(os.path.dirname(pdf_path))
        os.makedirs(out_base, exist_ok=True)
        out_path = os.path.join(out_base, f"page_{page_idx}.png")
        pix.save(out_path)
        return out_path
    except Exception as e:
        logger.info("Failed to render PDF page %s: %s", page_idx, e)
        return None


def ensure_page_image(source_dir: str, page_idx: int, *, pdf_path: Optional[str] = None, dpi: int = 150) -> Optional[str]:
    """
    Ensure there is a page image for page_idx under <source_dir>/images named page_{idx}.ext.
    - If exists, return its path
    - Else render from sibling PDF (first *.pdf or provided pdf_path) and save as page_{idx}.png
    Returns the path or None on failure.
    """
    # Existing
    existing = find_existing_page_image(source_dir, page_idx)
    if existing:
        return existing
    # Render
    pdf = pdf_path or find_pdf_in_dir(source_dir)
    if not pdf:
        return None
    return render_pdf_page_to_image(pdf, page_idx, out_dir=page_images_dir(source_dir), dpi=dpi)


def _sanitize_string_list(values: List[Any]) -> List[str]:
    """Sanitize a list of strings: clean text, drop empties, deduplicate preserving order."""
    seen = set()
    out: List[str] = []
    for v in values:
        if not isinstance(v, str):
            continue
        s = sanitize_text(v) or ""
        # drop extremely short leftovers like single letters
        if len(s) < 2:
            continue
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


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
