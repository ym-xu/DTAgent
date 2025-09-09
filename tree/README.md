MM DocTree Adapter (MinerU content_list)

Overview
- Purpose: Normalize and enrich MinerU content_list for later MM DocTree building.
- Scope now: Deterministic adapter only; no TOC/heading/caption detection yet.
- Key additions: node_idx, node_level (rename), outline/bbox for images/tables, optional OCR/LLM hooks.
  Also extracts Markdown preview for tables.

What It Does
- Adds `node_idx` sequentially (0..n-1).
- Renames `text_level` to `node_level` when present; keeps other fields intact.
- Sets `node_level = -1` for any item missing it (including non-title `text` and non-text modalities).
- Enriches `image`/`table` items with `outline` (bbox) from `layout.json` by image filename matching.
- Provides optional hooks for OCR and LLM description (not enabled by default).
- Ensures `image` items always contain `text` and `description` fields (empty if OCR/LLM not used).
- Extracts `table_text` from `table_body` HTML as Markdown when possible; falls back to plain text.

Input Example (MinerU content_list)
```
[
  {
    "type": "text",
    "text": "UndergraduateProspectus",
    "text_level": 1,
    "page_idx": 0
  },
  {
    "type": "image",
    "img_path": "images/ca5dfe45....jpg",
    "image_caption": [],
    "image_footnote": [],
    "page_idx": 1
  },
  {
    "type": "table",
    "img_path": "images/f0f54084....jpg",
    "table_caption": ["APPLICATION PERIOD AND FEE"],
    "table_footnote": [],
    "table_body": "<table>...</table>",
    "page_idx": 24
  },
  {
    "type": "equation",
    "text": "\n$$\nI_{l} = \\left|\\sum_{i}A_{h,i}\\odot \\frac{\\partial\\mathcal{L}(x)}{\\partial A_{h,i}}\\right|. \\tag{1}\n$$\n",
    "text_format": "latex",
    "page_idx": 1
  }
]
```

Output Example (adapted)
```
[
  {
    "type": "text",
    "text": "UndergraduateProspectus",
    "node_level": 1,
    "page_idx": 0,
    "node_idx": 0
  },
  {
    "type": "image",
    "img_path": "images/ca5dfe45....jpg",
    "image_caption": [],
    "image_footnote": [],
    "page_idx": 1,
    "node_level": -1,
    "node_idx": 1,
    "outline": [x0, y0, x1, y1],
    "text": "",                         // may contain OCR text
    "description": ""                  // may contain LLM description
  },
  {
    "type": "table",
    "img_path": "images/f0f54084....jpg",
    "table_caption": ["APPLICATION PERIOD AND FEE"],
    "table_footnote": [],
    "table_body": "<table>...</table>",
    "page_idx": 24,
    "node_level": -1,
    "node_idx": 2,
    "outline": [x0, y0, x1, y1],
    "table_text": "| Col1 | Col2 |\n| --- | --- |\n| v1 | v2 |"  
  },
  {
    "type": "equation",
    "text": "\n$$\nI_{l} = \\left|\\sum_{i}A_{h,i}\\odot \\frac{\\partial\\mathcal{L}(x)}{\\partial A_{h,i}}\\right|. \\tag{1}\n$$\n",
    "text_format": "latex",
    "page_idx": 1,
    "node_level": -1,
    "node_idx": 3
  }
]
```

CLI Usage
- Directory batch:
  - Default (keep originals): `python -m tree.adapter --in-dir /path/to/MinerU_MMLB`
    - Writes alongside as `content_list.adapted.json`
    - Customize name: `python -m tree.adapter --in-dir /path/to/MinerU_MMLB --suffix ".mm.json"` → `content_list.mm.json`
  - Overwrite originals: `python -m tree.adapter --in-dir /path/to/MinerU_MMLB --in-place`
- Single file:
  - Default (keep original): `python -m tree.adapter --in-file /dir/content_list.json --layout-file /dir/layout.json`
    - Writes `/dir/content_list.adapted.json` unless you pass `--out` or customize `--suffix`
  - Overwrite original: `python -m tree.adapter --in-file /dir/content_list.json --layout-file /dir/layout.json --in-place`

Python API
```
from tree.adapter import adapt_content_list, adapt_single_file
from tree.utils import try_ocr_image, describe_image_with_llm

# Adapt an in-memory list
adapted = adapt_content_list(
    content_list,
    layout_obj=layout,                     # dict loaded from layout.json
    ocr_image_func=None,                   # or try_ocr_image
    describe_image_func=None               # or your LLM hook
)

# Adapt one file (write back)
adapt_single_file(
    dom_file="/dir/content_list.json",
    layout_file="/dir/layout.json",
    ocr_image_func=None,
    describe_image_func=None,
    write_in_place=True,
)
```

Layout → outline extraction
- Searches `layout.json` (or `layout['pdf_info']` when present) recursively for entries with `image_path` and sibling `bbox`.
- Builds an index and matches by image filename (basename). If exact match fails, uses partial path inclusion as fallback.
- Adds `outline` to the content_list item when a match is found.

OCR / LLM Hooks (Optional)
- OCR: `tree.utils.try_ocr_image(path)` uses Pillow + pytesseract if available; returns text or None.
  - Dependencies: `pip install pillow pytesseract` and system tesseract installed.
- LLM (VLM) Description: `tree.utils.describe_image_with_llm(path, prompt=None, **kwargs)` is a placeholder; implement with your client (e.g., qwen2.5-vl) and return a short description.
- Hooks are not called by default; pass them explicitly in API calls.

Notes
- `page_idx` is preserved as-is from MinerU.
- The adapter does not perform title disambiguation, TOC detection, caption pairing, or list nesting inference.
- Unknown fields from MinerU are preserved untouched.

Planned Next (Enhance Stage)
- Heading/TOC detection, list nesting with stack, caption pairing (figure/table), equations and code blocks refinement.
- LLM-assisted heuristics for ambiguous cases.
