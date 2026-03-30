from __future__ import annotations
from typing import List
from openai import OpenAI

DEFAULT_EMBED_MODEL = "text-embedding-3-small"

class Embedder:
    def __init__(self, model: str = DEFAULT_EMBED_MODEL):
        self.client = OpenAI()
        self.model = model

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        resp = self.client.embeddings.create(
            model=self.model,
            input=texts,
        )
        return [d.embedding for d in resp.data]
