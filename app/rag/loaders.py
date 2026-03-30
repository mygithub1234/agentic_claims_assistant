from __future__ import annotations
from dataclasses import dataclass
from typing import List
from pypdf import PdfReader

@dataclass
class PageText:
    page: int
    text: str

def load_pdf_pages(pdf_path: str) -> List[PageText]:
    reader = PdfReader(pdf_path)
    pages: List[PageText] = []
    for i, page in enumerate(reader.pages, start=1):
        txt = page.extract_text() or ""
        txt = txt.replace("\u00a0", " ").strip()
        pages.append(PageText(page=i, text=txt))
    return pages
