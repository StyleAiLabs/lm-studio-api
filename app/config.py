import os
from dotenv import load_dotenv

load_dotenv()

# API Configuration
LM_STUDIO_URL = os.getenv("LM_STUDIO_URL", "http://localhost:1234/v1")
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# Knowledge Base Configuration
DOCUMENTS_DIR = os.getenv("DOCUMENTS_DIR", "./data/documents")
VECTORSTORE_DIR = os.getenv("VECTORSTORE_DIR", "./data/vectorstore")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")