import os
import json
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from fastapi.testclient import TestClient

# Ensure fast startup (hash embeddings) for tests
os.environ.setdefault("FAST_START", "1")

from app.main import app  # noqa: E402

client = TestClient(app)

TENANT_HEADER = {"X-Tenant-Id": "default"}

def pretty(obj):
    return json.dumps(obj, indent=2)

def test_knowledge_status():
    resp = client.get("/api/knowledge/status", headers=TENANT_HEADER)
    assert resp.status_code == 200
    data = resp.json()
    print("Status:", pretty(data))
    assert "document_count" in data

def test_debug_queries():
    queries = [
        "What is the return policy?",
        "How long do I have to return items?",
        "Can I return electronics?",
        "What are the special conditions for returns?"
    ]
    for q in queries:
        r = client.get("/debug/knowledge", params={"query": q}, headers=TENANT_HEADER)
        assert r.status_code == 200
        payload = r.json()
        print(f"Query '{q}' found_matches={payload['found_matches']}")

def test_chat_with_kb():
    chat_payload = {
        "messages": [
            {"role": "user", "content": "What is our company's return policy?"}
        ],
        "use_knowledge_base": True,
        "tenant_id": "default"
    }
    r = client.post("/api/chat", json=chat_payload, headers=TENANT_HEADER)
    assert r.status_code == 200
    data = r.json()
    print("Chat response:", pretty(data))
    assert "message" in data

if __name__ == "__main__":
    # Run ad-hoc if executed directly
    test_knowledge_status()
    test_debug_queries()
    test_chat_with_kb()
    print("All ad-hoc tests completed")