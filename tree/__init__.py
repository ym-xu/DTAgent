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
from .page_payload import build_page_payload
from .page_plan import (
    validate_and_normalize_jsonlist,
    validate_and_normalize_plan,
)
from .refine_apply import (
    apply_items_order_and_levels,
    apply_merges,
    apply_virtual_titles,
    replace_page_children,
    apply_plan_to_document,
    apply_nodes_to_document,
)
from .llm_providers import (
    ProviderConfig,
    BaseProvider,
    QwenProvider,
    GPTProvider,
    MockProvider,
)
from .prompts import (
    render_plan_prompt,
    render_jsonlist_prompt,
)

__all__ = [
    "adapt_content_list",
    "adapt_single_file",
    "enrich_content_with_layout",
    "process_directory",
    "build_flat_doctree",
    "build_doctree_single",
    "build_doctree_directory",
    "build_page_payload",
    "validate_and_normalize_jsonlist",
    "validate_and_normalize_plan",
    "apply_items_order_and_levels",
    "apply_merges",
    "apply_virtual_titles",
    "replace_page_children",
    "apply_plan_to_document",
    "apply_nodes_to_document",
    "ProviderConfig",
    "BaseProvider",
    "QwenProvider",
    "GPTProvider",
    "MockProvider",
    "render_plan_prompt",
    "render_jsonlist_prompt",
]

__version__ = "0.4.0"
