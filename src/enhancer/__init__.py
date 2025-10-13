"""
Enhancer wrappers (M1) for DocTree pipeline.

This package re-exports stable enhancer utilities from the legacy `tree` package
so callers can import from `src.enhancer.*` consistently. We keep the original
`tree` modules intact as a known-good fallback and can progressively replace
implementations here without touching `tree`.
"""

# Re-export commonly used enhancers
from .caption_cleanup import *  # noqa: F401,F403
from .heading_levels import *  # noqa: F401,F403
from .headings import *  # noqa: F401,F403
from .media_enhance import *  # noqa: F401,F403
from .slides_enhance import *  # noqa: F401,F403
from .enhancer_2 import *  # noqa: F401,F403

__all__ = []  # filled by star-imports above

