# agentic_adjuster_assistant — Adjuster Assistant (Commercial Property)
**Agentic RAG + Custom Chunking + Custom Embeddings + Persistent Chroma (Colab + Google Drive)**

A portfolio-grade **insurance claims & policy copilot** that helps adjusters:
- ingest **commercial property** policy docs (Dec Page, Base Form, Endorsements)
- ingest **claim docs** (FNOL, adjuster notes, estimates/invoices)
- apply **custom chunking** (policy clause/section-aware; claim doc-type aware)
- create **custom embeddings** (explicit embeddings API call)
- store and query everything via a **persistent Chroma vector store**
- retrieve **citation-ready** evidence using metadata filters (policy_id, claim_id, doc_id, version, page, section)

---

## Why this project is interview-ready
- ✅ **Custom chunking strategy** (domain-aware; not “default chunking”)
- ✅ **Custom embeddings pipeline** (explicit embedding calls; not managed embedding)
- ✅ **Metadata-first correctness** (versioning, jurisdiction/state, doc types)
- ✅ **Re-ingestion support** (delete old doc vectors by doc_id + doc_version)
- ✅ **Citations** (doc_id + page + section + chunk_id)

---

## Repo layout

---

## Quickstart (Google Colab + Google Drive)

### 1) Mount Drive
```python
from google.colab import drive
drive.mount("/content/drive")
