# AI Coding Agent Instructions

Concise, project-specific guidance for autonomous coding agents working on this repository.

## 1. Architecture Overview
- FastAPI app (`app/main.py`) exposes endpoints for completions, chat, document mgmt, knowledge base ops, health/debug.
- Service layer:
  - `LLMService` (`app/services/llm_service.py`): orchestrates persona selection, knowledge retrieval, builds prompts, calls LM Studio REST (`/completions`, `/chat/completions`). Chooses completions API when KB context present; falls back to chat API.
  - `KnowledgeBase` (`app/services/knowledge_service.py`): manages documents (txt/pdf/docx), chunking, embeddings (SentenceTransformer via Chroma), semantic query, rebuild.
- Utilities: prompt construction (`app/utils/prompt_builder.py`), personas (`app/utils/personas.py`), document file ops (`app/utils/document_processor.py`), (web scraping util present but not yet referenced in main endpoints excerpt).
- Data layer: documents and persistent Chroma DB under `data/documents` and `data/vectorstore` (configurable via env).
- Models (`app/models.py`): Pydantic schemas defining request/response contracts.

## 2. Configuration & Environment
- Environment loaded in `app/config.py` via `python-dotenv`.
- Key vars: `LM_STUDIO_URL` (default `http://localhost:1234/v1`), `DOCUMENTS_DIR`, `VECTORSTORE_DIR`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, `EMBEDDING_MODEL`.
- Adjust by editing `.env`; restart server to apply.

## 3. Personas & Prompt Strategy
- Personas stored in `app/utils/personas.py` with `temperature` and `traits` list. Retrieve via `get_persona()`.
- Prompt builder injects system guardrails (_no chain-of-thought_, missing data handling) in `prompt_builder.py`.
- Knowledge queries: build a single concatenated prompt (`build_knowledge_prompt`) embedding context + question + persona traits (first 3 only for token efficiency).
- Regular chat: system message prepended (`build_regular_chat_prompt`).

## 4. Knowledge Base Mechanics
- Documents ingested via endpoints -> saved to disk -> `KnowledgeBase.add_document()` extracts text (txt/pdf/docx), paragraph-based chunking with overlap, added to Chroma collection `business_knowledge`.
- Query path: `KnowledgeBase.query()` -> Chroma `query()` n_results=3 -> returns (documents, sources). Fallback returns one doc if empty.
- Rebuild: deletes collection, reprocesses all files currently in documents dir.

## 5. LLM Call Flow
1. API endpoint receives request (persona, KB flag, etc.).
2. If KB enabled & context retrieved: construct enhanced prompt and call `.../completions` (single-shot style) to reduce message overhead.
3. If no context or failure: use chat API with injected system message.
4. Persona temperature overrides user-provided temperature when set.

## 6. Testing & Manual Verification
- Minimal test script: `tests/test_kb.py` (runs live HTTP calls against running server on `localhost:8000`). No unit test harness; treat as integration smoke test.
- Start server: `uvicorn app.main:app --reload` (ensure LM Studio running on configured port first).
- Common manual checks:
  - `/health` returns base URL.
  - Upload a file then `/api/knowledge/status` shows document count increase.
  - `/debug/knowledge?query=...` prints retrieved contexts.

## 7. Coding Conventions & Patterns
- Logging: use `logging.getLogger(__name__)` in services; prefer informative context slices (e.g., first 50/100 chars).
- Keep prompts token-efficient (truncate contexts to ~2000 chars).
- Chain-of-thought suppression: NEVER output model reasoning in responses; prompt builder enforces this.
- Error handling: Services raise exceptions upward; endpoints (currently partially stubbed) should wrap and return HTTP errors with clear messages.
- When extending endpoints, reuse existing service interfaces; avoid duplicating query / prompt logic in the route handlers.

## 8. Extending the System
- Add new persona: update `personas.py` (include temperature) and prompts will auto-use it.
- Support new file type: extend `_extract_text_from_file` in `KnowledgeBase` and add extension to `document_processor.list_documents`.
- Swap embedding model: change `EMBEDDING_MODEL` env var (ensure model is valid HuggingFace SentenceTransformer).
- Add source attribution in chat path: currently only knowledge-enhanced completion path returns sources; adapt fallback chat response similarly if required.

## 9. Gotchas & Tips
- Empty KB: queries silently fall back to returning a doc; handle with status vectors count check if precision needed.
- Large documents: paragraph chunking may produce uneven sizes; consider refining chunker if quality issues arise.
- Persona temperature: override may surprise users—document this in client UI.
- Avoid leaking internal reasoning: if model starts emitting it, strengthen `_build_system_instructions` or post-process.

## 10. Safe Modification Checklist
Before committing changes:
- Run server & smoke test endpoints (upload, chat with KB on/off).
- Verify Chroma path and collection count unaffected unless intended.
- Ensure new prompts keep guardrails (system instructions included once).

Provide feedback if additional workflows (CI, Dockerization, auth) are added later so this file can evolve.
