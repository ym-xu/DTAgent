# tree/cache_utils.py
import os, json, hashlib, time, inspect
from typing import Any, Callable, Dict, List, Optional, Tuple

def _stable_dumps(obj: Any) -> str:
    def default(o):
        # 回退：将不可序列化对象转为字符串
        return repr(o)
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=default)

def _sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def _sha256_obj(o: Any) -> str:
    return _sha256_str(_stable_dumps(o))

def _atomic_write(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(content)
    os.replace(tmp, path)

class StepCache:
    """
    每一步输出以 {step_name}-{sig}.json 形式保存。
    sig 由：params + deps 的 digest + sig_extra 组成。
    deps 通过前置步骤的 meta.output_sha 采集，保证上游变化→签名变化。
    """
    def __init__(self, source_dir: str, cache_dir_name: str = "_stats"):
        self.base = os.path.join(source_dir, cache_dir_name)
        os.makedirs(self.base, exist_ok=True)

    def subdir(self, name: str) -> str:
        p = os.path.join(self.base, name)
        os.makedirs(p, exist_ok=True)
        return p

    def artifact_path(self, step_name: str, sig: Optional[str] = None) -> str:
        filename = f"{step_name}.json" if not sig else f"{step_name}-{sig}.json"
        return os.path.join(self.base, filename)

    def _load_payload(self, path: str) -> Optional[Dict[str, Any]]:
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def load(self, step_name: str, sig: Optional[str] = None) -> Optional[Tuple[Any, Dict[str, Any]]]:
        p = self.artifact_path(step_name, sig)
        payload = self._load_payload(p)
        if payload is None:
            return None
        return payload.get("data"), payload.get("meta", {})

    def digest_of(self, step_name: str) -> Optional[str]:
        """
        查找该步骤*任意签名*的最近文件并取 output_sha。
        一般不直接用；更常用在 load_or_compute 里用 deps 的已知签名。
        """
        # 优先找无签名的（如 00_items.json），否则找匹配前缀的最新文件
        base = self.artifact_path(step_name)
        if os.path.exists(base):
            payload = self._load_payload(base)
            if payload:
                return payload.get("meta", {}).get("output_sha")
        # 找所有匹配 step_name-*.json
        prefix = os.path.join(self.base, f"{step_name}-")
        best = None
        if os.path.isdir(self.base):
            for fn in os.listdir(self.base):
                if fn.startswith(f"{step_name}-") and fn.endswith(".json"):
                    path = os.path.join(self.base, fn)
                    if best is None or os.path.getmtime(path) > os.path.getmtime(best):
                        best = path
        if best:
            payload = self._load_payload(best)
            if payload:
                return payload.get("meta", {}).get("output_sha")
        return None

    def save(
        self,
        step_name: str,
        data: Any,
        *,
        params: Dict[str, Any],
        deps: Dict[str, str],
        code_refs: Optional[List[Any]] = None,
        sig: Optional[str] = None,
    ) -> str:
        code_hash: Dict[str, Optional[str]] = {}
        if code_refs:
            for ref in code_refs:
                try:
                    src = inspect.getsource(ref)
                    code_hash[getattr(ref, "__name__", str(ref))] = _sha256_str(src)
                except Exception:
                    code_hash[str(ref)] = None
        meta = {
            "step": step_name,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "params": params,
            "deps": deps,
            "code_hash": code_hash,
            "output_sha": _sha256_obj(data),
        }
        p = self.artifact_path(step_name, sig)
        _atomic_write(p, _stable_dumps({"meta": meta, "data": data}))
        return p

    def load_or_compute(
        self,
        step_name: str,
        *,
        params: Dict[str, Any],
        deps_step_names: List[str],
        compute_fn: Callable[[], Any],
        code_refs: Optional[List[Any]] = None,
        force: bool = False,
        cache_only: bool = False,
        sig_extra: Optional[Any] = None,
    ) -> Tuple[Any, Dict[str, Any], str]:
        dep_digests = {n: self.digest_of(n) for n in deps_step_names}
        sig_input = {"params": params, "deps": dep_digests, "sig_extra": sig_extra}
        sig = _sha256_obj(sig_input)[:10]
        if not force:
            loaded = self.load(step_name, sig)
            if loaded is not None:
                data, meta = loaded
                print(f"[cache] hit {step_name} ({sig})")
                return data, meta, sig
        if cache_only:
            raise FileNotFoundError(f"Cache miss for {step_name} (sig={sig}) while cache_only=True.")
        print(f"[cache] miss {step_name} ({sig}) → computing…")
        data = compute_fn()
        path = self.save(step_name, data, params=params, deps=dep_digests, code_refs=code_refs, sig=sig)
        print(f"[cache] saved {step_name} → {os.path.basename(path)}")
        return data, {"params": params, "deps": dep_digests}, sig

def make_cached_llm(llm_fn: Callable, cache_dir: str) -> Callable:
    """
    将任意 LLM 客户端函数包装为带磁盘缓存的版本（以请求内容为 key）。
    """
    os.makedirs(cache_dir, exist_ok=True)
    def _call(*args, **kwargs):
        req = {"args": args, "kwargs": kwargs}
        key = _sha256_obj(req)
        # 分片目录防止单目录文件过多
        shard = os.path.join(cache_dir, key[:2], key[2:4])
        os.makedirs(shard, exist_ok=True)
        path = os.path.join(shard, f"{key}.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            return payload["response"]
        resp = llm_fn(*args, **kwargs)
        _atomic_write(path, _stable_dumps({"request": req, "response": resp}))
        return resp
    return _call