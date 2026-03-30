from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Dict, Any, List, Iterable, Tuple

@dataclass
class Chunk:
    chunk_id: str
    text: str
    metadata: Dict[str, Any]

_HEADING_RE = re.compile(r"^(?:(\d+(\.\d+)*)\s+)?([A-Z][A-Z0-9 \-/:]{6,}|[A-Z][a-z][\w \-/:]{4,})\s*$")
_NOTE_TS_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4})"
    r"(\s+\d{1,2}:\d{2}(\s*[AP]M)?)?"
    r"(\s*[-–—]\s*)?.+",
    re.IGNORECASE
)

def _window_split(text: str, max_chars: int, overlap: int) -> List[str]:
    text = text.strip()
    if len(text) <= max_chars:
        return [text]
    out: List[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        out.append(text[start:end].strip())
        if end == len(text):
            break
        start = max(0, end - overlap)
    return [x for x in out if x]

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

def _split_claim_notes(text: str) -> List[str]:
    lines = text.splitlines()
    entries: List[List[str]] = []
    current: List[str] = []
    found = False
    for ln in lines:
        if _NOTE_TS_RE.match(ln.strip()):
            found = True
            if current:
                entries.append(current)
            current = [ln.strip()]
        else:
            if ln.strip():
                current.append(ln.strip())
    if current:
        entries.append(current)
    if not found:
        return [text.strip()]
    return ["\n".join(e).strip() for e in entries if e]

def chunk_claim_document(
    pages: Iterable[Tuple[int, str], Any],
    base_metadata: Dict[str, Any],
    max_chars: int = 1800,
    overlap: int = 200,
):
    doc_type = (base_metadata.get("doc_type") or "claim_doc").lower()
    claim_id = base_metadata.get("claim_id", "unknown")
    doc_id = base_metadata.get("doc_id", "doc_unknown")

    chunks: List[Chunk] = []
    idx = 0
    for page_num, page_text in pages:
        if not (page_text or "").strip():
            continue
        if doc_type in ("claim_note", "adjuster_note", "notes"):
            entries = _split_claim_notes(page_text)
            for entry in entries:
                for part in _window_split(entry, max_chars=max_chars, overlap=overlap):
                    idx += 1
                    chunk_id = f"clm_{claim_id}_{doc_id}_p{page_num}_{idx}"
                    md = dict(base_metadata)
                    md.update({"page": page_num, "section": "NOTE_ENTRY", "chunk_id": chunk_id, "doc_type": doc_type})
                    chunks.append(Chunk(chunk_id=chunk_id, text=part, metadata=md))
        else:
            blocks = _split_by_headings(page_text)
            for heading, body in blocks:
                for part in _window_split(body, max_chars=max_chars, overlap=overlap):
                    idx += 1
                    chunk_id = f"clm_{claim_id}_{doc_id}_p{page_num}_{idx}"
                    md = dict(base_metadata)
                    md.update({"page": page_num, "section": heading, "chunk_id": chunk_id, "doc_type": doc_type})
                    chunks.append(Chunk(chunk_id=chunk_id, text=part, metadata=md))
    return chunks
