from __future__ import annotations
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings

def _normalize_where(where: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Chroma requires a single top-level operator in `where`.
    If caller passes multiple field filters, wrap them in $and.
    """
    if not where:
        return {}
    if len(where) == 1:
        return where
    return {"$and": [{k: v} for k, v in where.items()]}

class ChromaStore:
    def __init__(self, persist_dir: str, collection_name: str = "agentic_adjuster_assistant"):
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self.col = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(self, ids: List[str], embeddings: List[List[float]], documents: List[str], metadatas: List[Dict[str, Any]]):
        self.col.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)

    def query(self, query_embedding: List[float], where: Optional[Dict[str, Any]] = None, top_k: int = 8):
        return self.col.query(query_embeddings=[query_embedding], n_results=top_k, where=_normalize_where(where))

    def delete_by_filter(self, where: Dict[str, Any]):
        self.col.delete(where=_normalize_where(where))
