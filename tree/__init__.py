"""
Tree package: adapter to enrich MinerU content_list for MM DocTree.

Focus of the adapter step:
- Add node_idx (sequential index)
- Rename MinerU's text_level -> node_level when present
- Default node_level = -1 for non-text modalities (image/table/equation/code)
- Attach outline/bbox for images/tables by reading layout.json
- Optional interfaces for OCR and LLM description are provided in utils.py
"""

from .adapter import (
    adapt_content_list,
    adapt_single_file,
    enrich_content_with_layout,
    process_directory,
)
from .builder import (
    build_flat_doctree,
    build_single as build_doctree_single,
    build_directory as build_doctree_directory,
)

__all__ = [
    "adapt_content_list",
    "adapt_single_file",
    "enrich_content_with_layout",
    "process_directory",
    "build_flat_doctree",
    "build_doctree_single",
    "build_doctree_directory",
]

__version__ = "0.3.0"
