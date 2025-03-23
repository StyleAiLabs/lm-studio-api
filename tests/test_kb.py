import requests
import json

# Base URL - adjust if your API is running on a different port/host
BASE_URL = "http://localhost:8000"

# Step 1: Check knowledge base status
print("Checking knowledge base status...")
status_response = requests.get(f"{BASE_URL}/api/knowledge/status")
status_data = status_response.json()
print(f"Status: {json.dumps(status_data, indent=2)}")

# Step 2: Test direct knowledge query with debug endpoint
# Try various queries related to your sample.txt content
test_queries = [
    "What is the return policy?",
    "How long do I have to return items?",
    "Can I return electronics?",
    "What are the special conditions for returns?"
]

print("\nTesting direct knowledge retrieval with debug endpoint...")
for query in test_queries:
    debug_response = requests.get(f"{BASE_URL}/debug/knowledge", params={"query": query})
    debug_data = debug_response.json()
    print(f"\nQuery: {query}")
    print(f"Found matches: {debug_data['found_matches']}")
    if debug_data['found_matches']:
        print(f"Sources: {debug_data['sources']}")
        print(f"First context snippet: {debug_data['contexts'][0][:100]}...")
    else:
        print("No matches found")

# Step 3: Test chat endpoint with knowledge base
print("\nTesting chat endpoint with knowledge base...")
chat_payload = {
    "messages": [
        {"role": "user", "content": "What is our company's return policy?"}
    ],
    "use_knowledge_base": True
}

chat_response = requests.post(f"{BASE_URL}/api/chat", json=chat_payload)
chat_data = chat_response.json()
print(f"Chat response: {json.dumps(chat_data, indent=2)}")

# Check if sources were returned
if chat_data.get("source_documents"):
    print("Knowledge base was used - sources are included in response")
else:
    print("No source documents returned - knowledge base might not be used")