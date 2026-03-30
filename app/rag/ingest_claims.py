from __future__ import annotations
from typing import Dict, Any, List, Tuple

from app.rag.claims_chunking import chunk_claim_document
from app.rag.ingest_common import ingest_pdf

def _claims_chunker(pages: List[Tuple[int, str]], base_md: Dict[str, Any], max_chars: int, overlap: int):
    return chunk_claim_document(pages=pages, base_metadata=base_md, max_chars=max_chars, overlap=overlap)

def ingest_claim_pdf(
    *,
    pdf_path: str,
    claim_metadata: Dict[str, Any],
    persist_dir: str,
    collection_name: str = "agentic_adjuster_assistant",
    embed_model: str = "text-embedding-3-small",
    max_chars: int = 1800,
    overlap: int = 200,
    reingest: bool = True,
) -> int:
    required = ["claim_id", "policy_id", "doc_type", "doc_id", "doc_version"]
    missing = [k for k in required if not claim_metadata.get(k)]
    if missing:
        raise ValueError(f"Missing required claim_metadata keys: {missing}")

    # Reingest by doc_id + doc_version
    reingest_filter = {"doc_id": claim_metadata["doc_id"], "doc_version": claim_metadata["doc_version"]} if reingest else None

    return ingest_pdf(
        pdf_path=pdf_path,
        base_metadata=claim_metadata,
        chunker=_claims_chunker,
        persist_dir=persist_dir,
        collection_name=collection_name,
        embed_model=embed_model,
        max_chars=max_chars,
        overlap=overlap,
        reingest_filter=reingest_filter,
    )
