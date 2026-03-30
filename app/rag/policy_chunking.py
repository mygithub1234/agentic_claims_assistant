from __future__ import annotations
import re
from dataclasses import dataclass
from typing import List, Dict, Any, Iterable, Tuple

@dataclass
class Chunk:
    chunk_id: str
    text: str
    metadata: Dict[str, Any]

_HEADING_RE = re.compile(r"^(?:(\d+(\.\d+)*)\s+)?([A-Z][A-Z0-9 \-/:]{6,}|[A-Z][a-z][\w \-/:]{4,})\s*$")

def _split_by_headings(text: str) -> List[Tuple[str, str]]:
    lines = [ln.strip() for ln in text.splitlines()]
    blocks: List[Tuple[str, List[str]]] = []
    current_heading = "BODY"
    current: List[str] = []
    found = False
    for ln in lines:
        if not ln:
            continue
        if _HEADING_RE.match(ln) and len(ln) <= 90:
            found = True
            if current:
                blocks.append((current_heading, current))
            current_heading = ln
            current = []
        else:
            current.append(ln)
    if current:
        blocks.append((current_heading, current))
    if not found:
        return [("BODY", text)]
    return [(h, "\n".join(b).strip()) for h, b in blocks if b]

def _window_split(text: str, max_chars: int = 1600, overlap: int = 200) -> List[str]:
    text = text.strip()
    if len(text) <= max_chars:
        return [text]
    out = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        out.append(text[start:end].strip())
        if end == len(text):
            break
        start = max(0, end - overlap)
    return [x for x in out if x]

def chunk_policy_pages(
    pages: Iterable[Tuple[int, str]],
    base_metadata: Dict[str, Any],
    max_chars: int = 1600,
    overlap: int = 200,
) -> List[Chunk]:
    chunks: List[Chunk] = []
    idx = 0
    for page_num, page_text in pages:
        if not page_text.strip():
            continue
        blocks = _split_by_headings(page_text)
        for heading, body in blocks:
            parts = _window_split(body, max_chars=max_chars, overlap=overlap)
            for p in parts:
                idx += 1
                chunk_id = f"pol_{base_metadata.get('policy_id','unknown')}_p{page_num}_{idx}"
                md = dict(base_metadata)
                md.update({"page": page_num, "section": heading, "doc_type": "policy", "chunk_id": chunk_id})
                chunks.append(Chunk(chunk_id=chunk_id, text=p, metadata=md))
    return chunks
