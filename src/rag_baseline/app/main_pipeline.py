from __future__ import annotations

import argparse
import json
import os
from typing import Any

import yaml  # type: ignore

from ..backends.llamaindex_backend import LlamaIndexBackend
from ..backends.ragatouille_backend import RAGatouilleBackend
from ..backends.base import Response


def run_build_and_query(doc_path: str, cfg_path: str, query: str | None = None, cfg_override: dict | None = None) -> None:
    cfg = yaml.safe_load(open(cfg_path, "r"))
    if cfg_override:
        # shallow override
        cfg.update(cfg_override)
    doc = json.load(open(doc_path, "r"))

    backend_name = (cfg.get("backend") or "llamaindex").lower()
    if backend_name == "ragatouille":
        backend = RAGatouilleBackend()
    else:
        backend = LlamaIndexBackend()

    backend.build(doc, cfg)

    q = query or "示例：怎么拆下连接器盖？"
    # For llamaindex backend, page_view=True triggers page collapse view if enabled.
    resp: Response = backend.query(q, page_view=True)
    print("Backend:", resp.backend)
    print("Answer:", resp.answer or "")
    for h in resp.hits[:5]:
        md = h.metadata or {}
        print(f"[{h.rank}] score={h.score:.3f} page={md.get('page_idx') or md.get('page_range')} path={md.get('section_path')}")

    backend.persist()


def main() -> None:
    ap = argparse.ArgumentParser(description="RAG over DocTree (embedding + FAISS + LlamaIndex)")
    ap.add_argument("--doc", required=True, help="Path to doctree.mm.json (or compatible tree JSON)")
    ap.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml"))
    ap.add_argument("--query", default=None, help="Optional query to run")
    ap.add_argument("--backend", default=None, help="Override backend: llamaindex | ragatouille")
    args = ap.parse_args()
    cfg_override = {}
    if args.backend:
        cfg_override["backend"] = args.backend
    run_build_and_query(args.doc, args.config, args.query, cfg_override=cfg_override)


if __name__ == "__main__":
    main()
