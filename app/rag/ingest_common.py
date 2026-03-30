from __future__ import annotations
from typing import Callable, Dict, Any, List, Tuple, Protocol, Optional
from openai import OpenAI

from app.rag.loaders import load_pdf_pages
from app.rag.vectorstore import ChromaStore

class ChunkLike(Protocol):
    chunk_id: str
    text: str
    metadata: Dict[str, Any]

ChunkerFn = Callable[[List[Tuple[int, str]], Dict[str, Any], int, int], List[ChunkLike]]

def ingest_pdf(
    *,
    pdf_path: str,
    base_metadata: Dict[str, Any],
    chunker: ChunkerFn,
    persist_dir: str,
    collection_name: str = "agentic_adjuster_assistant",
    embed_model: str = "text-embedding-3-small",
    max_chars: int = 1600,
    overlap: int = 200,
    reingest_filter: Optional[Dict[str, Any]] = None,
) -> int:
    """
    Generic ingestion:
      1) load PDF pages
      2) chunk with provided chunker(pages, base_metadata, max_chars, overlap)
      3) embed chunks
      4) upsert to Chroma

    reingest_filter:
      If provided, delete existing vectors that match this metadata filter
      before upserting (useful for doc_id+doc_version).
    """
    pages = load_pdf_pages(pdf_path)
    page_tuples = [(p.page, p.text) for p in pages]

    chunks = chunker(page_tuples, base_metadata, max_chars, overlap)
    if not chunks:
        return 0

    store = ChromaStore(persist_dir=persist_dir, collection_name=collection_name)
    if reingest_filter:
        store.delete_by_filter(reingest_filter)

    client = OpenAI()
    texts = [c.text for c in chunks]
    resp = client.embeddings.create(model=embed_model, input=texts)
    vectors = [d.embedding for d in resp.data]

    store.upsert(
        ids=[c.chunk_id for c in chunks],
        embeddings=vectors,
        documents=[c.text for c in chunks],
        metadatas=[c.metadata for c in chunks],
    )
    return len(chunks)
