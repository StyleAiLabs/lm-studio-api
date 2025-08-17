import os, io, sys, json
from fastapi.testclient import TestClient

# Fast startup & offline stubs
os.environ.setdefault("FAST_START", "1")
os.environ.setdefault("OFFLINE_MODE", "1")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.main import app  # noqa: E402

client = TestClient(app)
TENANT = "default"
HDR = {"X-Tenant-Id": TENANT}

def _p(obj):
    print(json.dumps(obj, indent=2))

def test_health():
    r = client.get('/health')
    assert r.status_code == 200
    data = r.json()
    assert data.get('status') == 'healthy'


def test_completion_offline():
    payload = {"prompt": "Say hi", "use_knowledge_base": False, "tenant_id": TENANT}
    r = client.post('/api/completion', json=payload, headers=HDR)
    assert r.status_code == 200
    assert 'OFFLINE' in r.json()['text']


def test_upload_and_status_cycle():
    # Prepare in-memory text file
    content = b"Return Policy:\nItems can be returned within 30 days."
    files = {"file": ("policy.txt", content, 'text/plain')}
    r = client.post('/api/knowledge/upload', files=files, data={"tenant_id": TENANT})
    assert r.status_code == 200
    up = r.json()
    assert up['status'] in ('success', 'warning')

    # Status should reflect document presence
    s = client.get('/api/knowledge/status', headers=HDR)
    assert s.status_code == 200
    data = s.json()
    assert any(doc.endswith('policy.txt') for doc in data['documents'])


def test_debug_query():
    r = client.get('/debug/knowledge', params={'query': 'return'}, headers=HDR)
    assert r.status_code == 200


def test_chat_offline():
    payload = {"messages": [{"role": "user", "content": "Hello"}], "use_knowledge_base": True, "tenant_id": TENANT}
    r = client.post('/api/chat', json=payload, headers=HDR)
    assert r.status_code == 200
    assert 'message' in r.json()


def test_rebuild():
    r = client.post('/api/knowledge/rebuild', headers=HDR)
    assert r.status_code == 200


def test_delete_document():
    # Delete policy.txt if exists
    r = client.delete('/api/knowledge/documents/policy.txt', headers=HDR)
    # Accept 200 (deleted) or 404 (already gone)
    assert r.status_code in (200, 404)

if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_') and callable(fn):
            print(f'Running {name}...')
            fn()
    print('All endpoint tests executed.')
