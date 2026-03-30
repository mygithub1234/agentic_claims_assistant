from __future__ import annotations
from typing import Dict, Any, List, Tuple

from app.rag.policy_chunking import chunk_policy_pages
from app.rag.ingest_common import ingest_pdf

def _policy_chunker(pages: List[Tuple[int, str]], base_md: Dict[str, Any], max_chars: int, overlap: int):
    return chunk_policy_pages(pages, base_metadata=base_md, max_chars=max_chars, overlap=overlap)

def ingest_policy_pdf(
    *,
    pdf_path: str,
    policy_metadata: Dict[str, Any],
    persist_dir: str,
    collection_name: str = "agentic_adjuster_assistant",
    embed_model: str = "text-embedding-3-small",
    max_chars: int = 1600,
    overlap: int = 200,
    reingest: bool = True,
) -> int:
    # For policies, reingest by doc_id + doc_version (if present)
    reingest_filter = None
    if reingest and policy_metadata.get("doc_id") and policy_metadata.get("doc_version"):
        reingest_filter = {"doc_id": policy_metadata["doc_id"], "doc_version": policy_metadata["doc_version"]}

    return ingest_pdf(
        pdf_path=pdf_path,
        base_metadata=policy_metadata,
        chunker=_policy_chunker,
        persist_dir=persist_dir,
        collection_name=collection_name,
        embed_model=embed_model,
        max_chars=max_chars,
        overlap=overlap,
        reingest_filter=reingest_filter,
    )
