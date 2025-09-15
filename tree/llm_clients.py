import base64
import json
import os
from typing import Any, Dict, List, Optional


def _to_data_url(path: str) -> Optional[str]:
    try:
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        ext = os.path.splitext(path)[1].lower()
        mime = "image/png" if ext == ".png" else "image/jpeg"
        return f"data:{mime};base64,{b64}"
    except Exception:
        return None


def gpt_llm_call(
    messages: List[Dict[str, Any]],
    images: Optional[List[Any]] = None,
    *,
    model: str = "gpt-4o",
    api_key: Optional[str] = None,
    endpoint: Optional[str] = None,
    temperature: float = 0.0,
    top_p: float = 0.1,
    json_mode: bool = True,
) -> str:
    """
    Minimal GPT (OpenAI) caller for toc.* helpers.
    - messages: list of {role, content}. content may be str; if images provided, last user message is converted to parts.
    - images: list of URL or local file paths. Local files are inlined as data URLs.
    Returns the model's message.content (string).
    """
    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError("openai package not installed. `pip install openai`. Error: %s" % e)

    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not provided.")

    client = OpenAI(api_key=key)

    # Prepare messages; attach images to last user message if any
    msgs: List[Dict[str, Any]] = []
    for m in messages:
        msgs.append({"role": m.get("role", "user"), "content": m.get("content", "")})

    if images:
        idx = max((i for i, mm in enumerate(msgs) if mm.get("role") == "user"), default=None)
        if idx is None:
            msgs.append({"role": "user", "content": ""})
            idx = len(msgs) - 1
        parts: List[Dict[str, Any]] = [{"type": "text", "text": str(msgs[idx].get("content", ""))}]
        for im in images:
            url = None
            if isinstance(im, str) and im.startswith("http"):
                url = im
            elif isinstance(im, str) and os.path.exists(im):
                url = _to_data_url(im)
            if url:
                parts.append({"type": "image_url", "image_url": {"url": url}})
        msgs[idx]["content"] = parts

    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": msgs,
        "temperature": temperature,
        "top_p": top_p,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    resp = client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content or ""


def qwen_llm_call(
    messages: List[Dict[str, Any]],
    images: Optional[List[Any]] = None,
    *,
    model: str = "qwen-vl-max",
    api_key: Optional[str] = None,
    temperature: float = 0.0,
    top_p: float = 0.1,
    json_mode: bool = True,
    json_schema: Optional[Dict[str, Any]] = None,
    seed: Optional[int] = None,
) -> str:
    """
    Minimal Qwen (DashScope) caller for toc.* helpers.
    - messages: list of {role, content} where content is text; we convert to Qwen multimodal content with images if provided.
    - images: list of URL or local paths; local paths converted to file:// URIs (DashScope uploads them).
    Returns content string (concatenated text segments if Qwen returns list segments).
    """
    try:
        import dashscope
        from dashscope import MultiModalConversation
    except Exception as e:
        raise RuntimeError("dashscope package not installed. `pip install dashscope`. Error: %s" % e)

    key = api_key or os.environ.get("DASHSCOPE_API_KEY")
    if not key:
        raise RuntimeError("DASHSCOPE_API_KEY not provided.")
    dashscope.api_key = key

    # Merge all user contents into one text block and attach images as multimodal content
    sys_msgs = [m for m in messages if m.get("role") == "system"]
    user_msgs = [m for m in messages if m.get("role") != "system"]
    user_text = "\n\n".join(str(m.get("content", "")) for m in user_msgs)

    mm_content: List[Dict[str, Any]] = []
    if images:
        for im in images:
            if isinstance(im, str):
                uri = im
                if os.path.exists(im):
                    uri = "file://" + os.path.abspath(im)
                mm_content.append({"image": uri})
    mm_content.append({"text": user_text})

    mm_messages: List[Dict[str, Any]] = []
    if sys_msgs:
        mm_messages.append({"role": "system", "content": sys_msgs[0].get("content", "")})
    mm_messages.append({"role": "user", "content": mm_content})

    call_kwargs: Dict[str, Any] = {
        "model": model,
        "messages": mm_messages,
        "temperature": temperature,
        "top_p": top_p,
    }
    if seed is not None:
        call_kwargs["seed"] = seed
    if json_mode:
        if json_schema:
            call_kwargs["response_format"] = {"type": "json_schema", "json_schema": json_schema}
        else:
            call_kwargs["response_format"] = {"type": "json_object"}

    rsp = MultiModalConversation.call(**call_kwargs)
    try:
        content = rsp.output.choices[0].message.content
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            texts: List[str] = []
            for seg in content:
                if isinstance(seg, dict) and isinstance(seg.get("text"), str):
                    texts.append(seg["text"])
            return "".join(texts)
        return str(content)
    except Exception:
        out = getattr(rsp, "output_text", None)
        if out is not None:
            return out
        return json.dumps(rsp, ensure_ascii=False)

