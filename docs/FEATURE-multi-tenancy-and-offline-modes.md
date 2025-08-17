# Multi-Tenancy, Offline & Fast Start Feature Summary

## Objective
Implement multi-tenancy (per-company isolation) with migration of legacy single-tenant data into a default tenant, plus operational modes (offline, fast start) and end-to-end test coverage.

## Key Outcomes
### 1. Multi-Tenancy Architecture
- Per-tenant directories: `data/documents/{tenant}`, `data/vectorstore/{tenant}`
- Per-tenant Chroma collections: `business_knowledge_{tenant}`
- `tenant_id` added across request/response models; accepted via header/body/query/form
- In-memory cache of `KnowledgeBase` instances; lazy init
- Migration routine moves legacy flat data into `default/` and creates marker file `.multitenant_migrated` (idempotent)

### 2. Knowledge Base Enhancements
- Robust `_ensure_collection` with get_or_create fallback
- `HashEmbeddingFunction` for `FAST_START` mode (skips heavy model download)
- Rebuild endpoint re-processes all tenant docs
- Query returns top 3 docs (fallback to 1 if empty)

### 3. LLM Orchestration
- `generate_completion` vs `generate_chat_completion` unified persona + context flow
- Knowledge-based prompts call `/completions`; fallback to `/chat` if no context
- Persona temperature overrides user-specified temperature
- `OFFLINE_MODE` stubs deterministic responses for tests/dev

### 4. Operational Modes
- `OFFLINE_MODE`: Skips external LM Studio calls; returns stub text
- `FAST_START`: Skips SentenceTransformer load; uses hash embeddings
- Both toggles environment-driven (`.env` / shell export)

### 5. Testing & Reliability
- Added `tests/test_kb.py` (KB ingest + chat offline)
- Added `tests/test_endpoints.py` (health, completion, chat, upload, status, debug, rebuild, delete)
- Uses FastAPI `TestClient` (no running server needed)
- Deterministic offline responses ensure stable assertions

### 6. Dependency & Env Stabilization
- Pinned `huggingface_hub==0.16.4`, `transformers==4.33.3` to align with `sentence-transformers`
- Added `python-multipart`, `httpx`, `starlette` explicit
- Python 3.11 chosen (resolved pandas / build issues)
- Added `.venv` to `.gitignore`

### 7. Prompt & Persona Integrity
- System instructions suppress chain-of-thought
- Persona traits injected (trimmed to first 3 for token efficiency)
- Knowledge prompts concatenate context + question + traits

### 8. Migration Details
On first tenant KB access (default path):
1. Create `default/` directories if missing
2. Move legacy documents into default set
3. Rebuild default collection
4. Write marker `.multitenant_migrated` to prevent repeat

### 9. Current Runtime Status
- Server healthy in real mode (neither `OFFLINE_MODE` nor `FAST_START` set)
- Real embeddings load on demand (model download required once)

## Pending / Nice-to-Have
- Source list de-duplication in responses
- Test coverage for `/api/knowledge/add-website` (needs scrape mock)
- Validation that real embedding model loads (semantic quality check)
- Optional: eviction strategy for large tenant cache
- Optional: per-tenant rebuild endpoint variant (currently scoped by tenant param)

## Operational Playbook
### Development (fast, offline)
```bash
export OFFLINE_MODE=1 FAST_START=1
uvicorn app.main:app --reload
```

### Real mode
```bash
unset OFFLINE_MODE FAST_START
uvicorn app.main:app --host 0.0.0.0 --port 8000
```
Then rebuild per tenant for real embeddings:
```bash
curl -X POST "http://localhost:8000/api/knowledge/rebuild?tenant_id=default"
```

## Verification Steps
1. Upload document: `POST /api/knowledge/upload` (tenant=X)
2. Check status: `GET /api/knowledge/status` (tenant=X) -> doc_count increments
3. Chat with KB: `POST /api/chat` (`use_knowledge_base=true`, tenant=X) -> sources returned
4. Toggle OFFLINE/FAST modes to compare behavior

## Risks & Mitigations
| Risk | Mitigation |
|------|------------|
| Large number of tenants increases memory usage | Introduce LRU or timed eviction in `_kb_cache` |
| Embedding model download failures | Use `FAST_START` fallback; pre-warm on startup |
| Duplicate sources in response | De-duplicate before serializing response (planned) |

## Summary
Multi-tenancy, migration, operational toggles, and comprehensive test scaffolding are implemented. System is production-ready pending minor polish (source de-dup, website test, semantic verification).
