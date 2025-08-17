# AI Coding Agent Instructions

Concise, actionable guidance for working productively in this repo. Keep edits minimal, tenant-safe, and prompt-efficient.

## Core Architecture
- FastAPI entrypoint: `app/main.py` (routes for completion, chat, knowledge ops, website ingest, debug, health).
- Services: `LLMService` (prompt assembly + LM Studio REST) and `KnowledgeBase` (document IO, chunking, Chroma vector search) in `app/services/`.
- Utilities: personas (`personas.py`), prompt builder (`prompt_builder.py`), document helpers (`document_processor.py`), web crawler (`web_scraper.py`).
- Data layout (multi-tenant): `data/documents/<tenant>/` & `data/vectorstore/<tenant>/` with Chroma collection name `business_knowledge_<tenant>`.

## Multi-Tenancy Rules
- Tenant resolution precedence: body/form `tenant_id` > query `tenant_id` > header `X-Tenant-Id` > fallback `default`.
- Migration marker `.multitenant_migrated` prevents re-moving legacy single-tenant data (already handled automatically).
- Never mix tenant data paths; always derive paths via `KnowledgeBase` factory (cached per tenant).

## Operational Modes
- `OFFLINE_MODE=1`: Skip LM Studio calls; deterministic stub text (stable tests).
- `FAST_START=1`: Use cheap hash embeddings (no SentenceTransformer download) — rebuild later after disabling for real embeddings.
- Combine both for fastest local iteration; unset for production.

## Prompt & Persona Strategy
- System guardrails + persona traits (first 3 traits only) prepended once (see `prompt_builder.py`). Do NOT add duplicate system messages.
- Use `/completions` path when KB context exists; fallback to `/chat/completions` otherwise (handled in `LLMService`).
- Persona temperature overrides user-provided temperature if defined.
- Never surface chain-of-thought; keep answers concise & factual from supplied context.

## Knowledge Base Mechanics
- Ingestion: upload/document or website -> file saved -> text extracted -> paragraphs chunked with overlap -> embedded into Chroma.
- Query returns top 3 docs (fallback 1) and may return duplicate filenames; consider de-dup if adjusting response shape.
- Rebuild endpoint wipes & reprocesses tenant docs (use after toggling FAST_START off).

## Website Crawler (`web_scraper.py`)
- Depth=1 same-domain crawl, size caps: per page ~120KB, total ~250KB; duplicate pages skipped via normalized DOM hash.
- Boilerplate stripped; final aggregated file includes metadata header + page delimiters.
- Forbidden (403) raises `WebsiteScrapeForbidden` -> endpoint returns 400; preserve this behavior if modifying.

## Testing Workflow
- Integration tests: `tests/test_endpoints.py`, `tests/test_kb.py` (use `pytest`). Set `OFFLINE_MODE=1 FAST_START=1` for deterministic runs.
- Avoid starting uvicorn inside tests; they use FastAPI `TestClient` directly.
- When adding features: add minimal happy-path + one edge case test (offline stub OK).

## Extending Functionality
- New file type: extend `_extract_text_from_file` (in `knowledge_service.py`) & allow listing in `document_processor`.
- New persona: append entry in `personas.py`; traits auto-injected.
- Adjustable crawl params: add optional query/body fields and thread through to `crawl_website` (respect size guardrails).

## Logging & Error Patterns
- Use `logging.getLogger(__name__)`; log only short context slices (e.g., first 80 chars) to avoid leaking sensitive text.
- Raise domain-specific exceptions (e.g., `WebsiteScrapeForbidden`) and convert to HTTP errors in route handlers.

## Safe Change Checklist
1. Run tests (optionally with OFFLINE/FAST). 2. For KB-impacting changes, ingest a sample doc & verify `/api/knowledge/status`. 3. Confirm no duplicate system instructions in responses. 4. Maintain tenant isolation (no cross-tenant path usage). 5. Keep prompt size lean (< ~2000 chars context slice logic already trims—preserve trimming).

## Quick Commands (reference)
Dev (fast): `OFFLINE_MODE=1 FAST_START=1 uvicorn app.main:app --reload`
Prod-like: `unset OFFLINE_MODE FAST_START && uvicorn app.main:app --host 0.0.0.0 --port 8000`

Keep this file concise; document only proven patterns. Update if adding major capabilities (auth, deeper crawl depth, new LLM providers).
