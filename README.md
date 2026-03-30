# Adjuster Assistant (Commercial Property) — Agentic RAG + Custom Chunking

A portfolio-grade **insurance claims & policy copilot** that helps adjusters **summarize claims**, **retrieve relevant policy clauses**, and **produce citation-ready answers** using **custom chunking + custom embeddings** stored in a persistent **Chroma** vector store.

This project is designed to run smoothly in **Google Colab** (with persistence via Google Drive) and later evolve into a deployable service (ECS/EKS/Lambda) with model backends like OpenAI now and Bedrock later.

---

## What it does

✅ **Ingest policy PDFs** (Declarations, Base Form, Endorsements)  
✅ **Ingest claim PDFs** (FNOL, adjuster notes, estimates/invoices)  
✅ Apply **custom chunking strategies**:
- Policies: **section/clause-aware chunking** (Definitions / Coverage / Exclusions / Conditions)
- Claims: **doc-type aware chunking** (notes split by entry, others by headings/tables)
✅ Create **OpenAI embeddings** (explicit call — not managed vector-store embedding)  
✅ Store chunks + metadata in **Chroma (persistent)**  
✅ Retrieve by semantic similarity + **metadata filters** for correctness and citations

---

## Why this is “enterprise-shaped”

- **Metadata-first correctness** (policy_id/state/version/effective_date; claim_id/doc_type)
- **Custom chunking** tuned for insurance documents
- **Re-ingestion support** (delete old chunks by doc_id + doc_version)
- **Citation-ready retrieval**: doc_id + page + section + chunk_id per result

---

## Repo layout


---

## Quickstart (Google Colab + Google Drive)

### 1) Mount Drive and set repo path

```python
from google.colab import drive
drive.mount("/content/drive")

REPO = "/content/drive/MyDrive/Agentic AI Projects/agentic_adjuster_assistant"
%cd "{REPO}"
