import json
import logging
import os
from typing import Any, Optional


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

