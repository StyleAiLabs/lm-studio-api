# API Usage & Sample Requests

This document provides concrete examples for interacting with the Business AI API.

## Health Check
Simple readiness probe.
```bash
curl -s http://localhost:8000/health
```
Response:
```json
{"status":"healthy","lm_studio_url":"http://localhost:1234/v1"}
```

## Text Completion
POST `/api/completion` with optional knowledge base usage and persona.

Request body:
```json
{
  "prompt": "List three benefits of our safety cabinets.",
  "max_tokens": 256,
  "temperature": 0.7,
  "use_knowledge_base": true,
  "persona": "professional",
  "tenant_id": "default"
}
```
Curl:
```bash
curl -X POST http://localhost:8000/api/completion \
  -H 'Content-Type: application/json' \
  -d '{
        "prompt":"List three benefits of our safety cabinets.",
        "max_tokens":256,
        "temperature":0.7,
        "use_knowledge_base":true,
        "persona":"professional",
        "tenant_id":"default"
      }'
```
Response (shape):
```json
{
  "text": "...model answer...",
  "source_documents": ["safetycabinets.co.nz_homepage.txt", "sample.txt"]
}
```

## Chat Completion
POST `/api/chat` for multi-turn messages.

Request body:
```json
{
  "messages": [
    {"role": "user", "content": "Summarize our property services offering."}
  ],
  "max_tokens": 300,
  "temperature": 0.6,
  "use_knowledge_base": true,
  "persona": "professional",
  "tenant_id": "williamspropertyservices"
}
```
Curl:
```bash
curl -X POST http://localhost:8000/api/chat \
  -H 'Content-Type: application/json' \
  -H 'X-Tenant-Id: williamspropertyservices' \
  -d '{
        "messages":[{"role":"user","content":"Summarize our property services offering."}],
        "max_tokens":300,
        "temperature":0.6,
        "use_knowledge_base":true,
        "persona":"professional"
      }'
```
Response shape:
```json
{
  "message": {"role": "assistant", "content": "..."},
  "source_documents": ["williamspropertyservices.co.nz_homepage.txt"]
}
```

## Document Upload
`multipart/form-data` to `/api/knowledge/upload`.
```bash
curl -X POST http://localhost:8000/api/knowledge/upload \
  -H 'X-Tenant-Id: default' \
  -F 'file=@/path/to/local/file.pdf'
```
Response shape:
```json
{
  "status": "success",
  "filename": "file.pdf",
  "message": "Document uploaded and processed successfully",
  "tenant_id": "default"
}
```

## Add Website (Multi-Page Crawl)
Scrape and ingest a website (depth 1) for a tenant.
```bash
curl -X POST 'http://localhost:8000/api/knowledge/add-website?tenant_id=williamspropertyservices' \
  -H 'Content-Type: application/json' \
  -d '{"url":"https://williamspropertyservices.co.nz"}'
```
Response shape:
```json
{
  "status": "success",
  "filename": "williamspropertyservices.co.nz_homepage.txt",
  "message": "Website content from https://williamspropertyservices.co.nz added successfully",
  "tenant_id": "williamspropertyservices"
}
```

## Knowledge Base Status
```bash
curl -s 'http://localhost:8000/api/knowledge/status?tenant_id=default'
```
Response shape:
```json
{
  "status": "active",
  "document_count": 3,
  "vector_count": 120,
  "documents": ["doc1.txt","doc2.pdf"],
  "tenant_id": "default"
}
```

## Rebuild Knowledge Base
Forces full re-index of existing documents.
```bash
curl -X POST 'http://localhost:8000/api/knowledge/rebuild?tenant_id=default'
```
Response:
```json
{"status":"success","message":"Knowledge base rebuilt successfully"}
```

## Delete Document
```bash
curl -X DELETE 'http://localhost:8000/api/knowledge/documents/sample.txt?tenant_id=default'
```
Response:
```json
{"status":"success","message":"Document sample.txt removed successfully"}
```

## Debug Knowledge Retrieval
Directly inspect retrieved contexts.
```bash
curl -s 'http://localhost:8000/debug/knowledge?query=property%20services&tenant_id=williamspropertyservices'
```
Response shape:
```json
{
  "query": "property services",
  "found_matches": true,
  "contexts": ["...context snippet..."],
  "sources": ["williamspropertyservices.co.nz_homepage.txt"],
  "tenant_id": "williamspropertyservices"
}
```

## Headers vs Body Tenant Resolution
Priority: (1) explicit field `tenant_id` in body/form, (2) query param `tenant_id`, (3) header `X-Tenant-Id`, (4) fallback `default`.

Example using header only:
```bash
curl -X POST http://localhost:8000/api/completion \
  -H 'Content-Type: application/json' \
  -H 'X-Tenant-Id: acme' \
  -d '{"prompt":"Hello","max_tokens":64}'
```

## Error Handling Examples
- 400: invalid scrape or forbidden website
- 404: deleting non-existent document
- 500: unexpected server error

Example forbidden scrape response:
```json
{"detail":"Access forbidden (403) when scraping https://blocked.example"}
```

## Best Practices
- Provide `tenant_id` explicitly to avoid ambiguity.
- Use `use_knowledge_base=false` for purely generative prompts.
- Rebuild after bulk uploads for consistency.
- Keep documents concise; large PDFs may fragment into many chunks.

---
This guide can be extended as new endpoints are added.
