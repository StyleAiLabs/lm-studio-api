import os
import logging
import shutil
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
# from sentence_transformers import SentenceTransformer  # Removed to avoid heavy import at startup
import chromadb
# Lazy import of embedding_functions only when needed to avoid heavy deps during FAST_START
embedding_functions = None
import pypdf
import docx2txt
import hashlib
import math

# Optional fast-start mode to skip heavy model download (set FAST_START=1)
FAST_START = os.getenv("FAST_START", "0") == "1"

from app.config import DOCUMENTS_DIR, VECTORSTORE_DIR, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL
from app.utils.document_processor import list_documents

# Cache for tenant-specific knowledge bases
_kb_cache: Dict[str, "KnowledgeBase"] = {}

MIGRATION_FLAG_FILE = ".multitenant_migrated"

def _ensure_default_migration():
    """One-time migration: move flat documents/vectorstore into default/ subdirectories if not yet migrated."""
    root_docs = Path(DOCUMENTS_DIR)
    root_vec = Path(VECTORSTORE_DIR)
    flag_path = root_docs / MIGRATION_FLAG_FILE

    try:
        if flag_path.exists():
            return
        # Determine if migration needed: any files directly under documents (excluding directories & flag)
        direct_files = [p for p in root_docs.glob("*") if p.is_file() and p.name != MIGRATION_FLAG_FILE]
        if not direct_files and any(root_docs.iterdir()):
            # Already structured or empty
            flag_path.write_text("already structured")
            return

        default_docs = root_docs / "default"
        default_docs.mkdir(parents=True, exist_ok=True)
        for f in direct_files:
            shutil.move(str(f), str(default_docs / f.name))

        # Vectorstore migration: move existing contents into vectorstore/default if not already
        default_vec = root_vec / "default"
        default_vec.mkdir(parents=True, exist_ok=True)
        # If chroma.sqlite3 exists at root and default is empty, move it
        root_sqlite = root_vec / "chroma.sqlite3"
        if root_sqlite.exists() and not any(default_vec.iterdir()):
            shutil.move(str(root_sqlite), str(default_vec / root_sqlite.name))
        # Move uuid dirs
        for item in root_vec.glob("*"):
            if item.is_dir() and item.name != "default":
                target = default_vec / item.name
                if not target.exists():
                    shutil.move(str(item), str(target))
        flag_path.write_text("migrated")
    except Exception as e:
        logging.getLogger(__name__).error(f"Migration error: {e}")

_ensure_default_migration()

class KnowledgeBase:
    def __init__(self,
                 tenant_id: str = "default",
                 base_documents_dir: str = DOCUMENTS_DIR,
                 base_vectorstore_dir: str = VECTORSTORE_DIR,
                 embedding_model: str = EMBEDDING_MODEL,
                 chunk_size: int = CHUNK_SIZE,
                 chunk_overlap: int = CHUNK_OVERLAP):
        self.tenant_id = tenant_id or "default"
        self.documents_dir = os.path.join(base_documents_dir, self.tenant_id)
        self.vectorstore_dir = os.path.join(base_vectorstore_dir, self.tenant_id)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Configure logging
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initializing KnowledgeBase for tenant '{self.tenant_id}'")

        # Create directories
        os.makedirs(self.documents_dir, exist_ok=True)
        os.makedirs(self.vectorstore_dir, exist_ok=True)

        # Embeddings & client (with fallback hash embedding for fast start or import issues)
        self.embedding_function = None
        if FAST_START:
            self.logger.warning("FAST_START enabled: using lightweight hash embedding function (not semantic)")
        try:
            if not FAST_START:
                global embedding_functions
                if embedding_functions is None:
                    from chromadb.utils import embedding_functions as ef  # type: ignore
                    embedding_functions = ef
                self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name=embedding_model
                )
        except Exception as e:
            self.logger.error(f"Failed loading sentence transformer model '{embedding_model}': {e}. Falling back to hash embeddings.")

        if self.embedding_function is None:
            # Define simple deterministic hash embedding of fixed dim 384
            dim = 384

            def _hash_embed(texts):
                vectors = []
                for t in texts:
                    h = hashlib.sha256(t.encode('utf-8')).digest()
                    nums = []
                    while len(nums) < dim:
                        for b in h:
                            nums.append((b / 255.0) * 2 - 1)
                            if len(nums) >= dim:
                                break
                    norm = math.sqrt(sum(x * x for x in nums)) or 1.0
                    vectors.append([x / norm for x in nums])
                return vectors

            class HashEmbeddingFunction:
                """Lightweight Chroma embedding function fallback.

                Conforms to interface: __call__(self, input: List[str]) -> List[List[float]]
                """
                def __call__(self, input):  # Chroma expects parameter name 'input'
                    return _hash_embed(input)

            self.embedding_function = HashEmbeddingFunction()
            self.logger.warning(
                "Using non-semantic hash embeddings; enable real model by unsetting FAST_START and ensuring model download works."
            )
        self.client = chromadb.PersistentClient(path=self.vectorstore_dir)

        self.collection_name = f"business_knowledge_{self.tenant_id}"
        self.collection = None
        self._ensure_collection()

    def _ensure_collection(self):
        """Idempotently get or create the Chroma collection."""
        if self.collection is not None:
            return
        try:
            # Preferred helper if available
            if hasattr(self.client, 'get_or_create_collection'):
                self.collection = self.client.get_or_create_collection(
                    name=self.collection_name,
                    embedding_function=self.embedding_function,
                    metadata={"tenant": self.tenant_id}
                )
            else:
                try:
                    self.collection = self.client.get_collection(
                        name=self.collection_name,
                        embedding_function=self.embedding_function
                    )
                except ValueError:
                    self.collection = self.client.create_collection(
                        name=self.collection_name,
                        embedding_function=self.embedding_function,
                        metadata={"tenant": self.tenant_id}
                    )
            self.logger.info(
                f"Collection ready '{self.collection_name}' (count={self.collection.count()})"
            )
        except Exception as e:
            self.logger.error(f"Failed to ensure collection {self.collection_name}: {e}")
            raise
    
    def _extract_text_from_file(self, file_path: str) -> Optional[str]:
        """Extract text from various file types."""
        try:
            ext = os.path.splitext(file_path)[1].lower()
            
            if ext == '.txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            
            elif ext == '.pdf':
                text = ""
                with open(file_path, 'rb') as f:
                    pdf = pypdf.PdfReader(f)
                    for page in pdf.pages:
                        text += page.extract_text() + "\n"
                return text
            
            elif ext == '.docx':
                return docx2txt.process(file_path)
            
            else:
                self.logger.warning(f"Unsupported file type: {ext}")
                return None
        
        except Exception as e:
            self.logger.error(f"Error extracting text from {file_path}: {str(e)}")
            return None
    
    def _chunk_text(self, text: str, source: str) -> List[Dict[str, Any]]:
        """Split text into chunks with overlap."""
        chunks = []
        
        # Simple chunk by paragraph then by size
        paragraphs = text.split('\n\n')
        
        current_chunk = ""
        chunk_id = 0
        
        for paragraph in paragraphs:
            # Clean paragraph
            paragraph = paragraph.strip()
            if not paragraph:
                continue
                
            if len(current_chunk) + len(paragraph) <= self.chunk_size:
                current_chunk += paragraph + "\n\n"
            else:
                if current_chunk:
                    chunks.append({
                        "id": f"{os.path.basename(source)}-{chunk_id}",
                        "text": current_chunk.strip(), 
                        "source": source
                    })
                    chunk_id += 1
                    
                    # Create overlap by keeping some content
                    words = current_chunk.split()
                    if len(words) > self.chunk_overlap:
                        current_chunk = " ".join(words[-self.chunk_overlap:]) + "\n\n"
                    else:
                        current_chunk = ""
                
                current_chunk += paragraph + "\n\n"
        
        # Add the last chunk
        if current_chunk:
            chunks.append({
                "id": f"{os.path.basename(source)}-{chunk_id}",
                "text": current_chunk.strip(), 
                "source": source
            })
        
        self.logger.info(f"Split document into {len(chunks)} chunks")
        return chunks
    
    def add_document(self, file_path: str) -> bool:
        """Add a single document to the knowledge base."""
        try:
            text = self._extract_text_from_file(file_path)
            
            if not text:
                self.logger.warning(f"Could not extract text from {file_path}")
                return False
            
            chunks = self._chunk_text(text, file_path)
            
            if not chunks:
                self.logger.warning(f"No chunks created from {file_path}")
                return False
            
            # Prepare data for Chroma
            ids = [chunk["id"] for chunk in chunks]
            texts = [chunk["text"] for chunk in chunks]
            metadatas = [{"source": chunk["source"]} for chunk in chunks]
            
            # Add to collection
            self.collection.add(
                ids=ids,
                documents=texts,
                metadatas=metadatas
            )
            
            self.logger.info(f"Added {len(chunks)} chunks from {file_path} to knowledge base")
            return True
        
        except Exception as e:
            self.logger.error(f"Error adding document {file_path}: {str(e)}")
            return False
    
    def rebuild_knowledge_base(self) -> bool:
        """Rebuild the entire knowledge base from documents."""
        try:
            # Reset collection
            # Delete then recreate collection
            try:
                self.client.delete_collection(self.collection_name)
                self.logger.info(f"Deleted existing collection {self.collection_name}")
            except ValueError:
                self.logger.info("No existing collection to delete for rebuild")

            self.collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function,
                metadata={"tenant": self.tenant_id}
            )
            
            # Add all documents
            success = True
            documents = list_documents(self.documents_dir)
            self.logger.info(f"Found {len(documents)} documents to process")
            
            for file in documents:
                file_path = os.path.join(self.documents_dir, file)
                if not self.add_document(file_path):
                    self.logger.warning(f"Failed to process {file}")
                    success = False
            
            return success
        
        except Exception as e:
            self.logger.error(f"Error rebuilding knowledge base: {str(e)}")
            return False
    
    def query(self, query_text: str, k: int = 3) -> Tuple[List[str], List[str]]:
        """
        Retrieve relevant documents for a query.
        
        Returns:
            Tuple of (content_list, source_list)
        """
        try:
            # Try to ensure we get results - REMOVING include_distances
            self.logger.info(f"Querying knowledge base with: '{query_text}'")
            self._ensure_collection()
            results = self.collection.query(
                query_texts=[query_text],
                n_results=k
            )
            
            # Check if we have results
            if results and 'documents' in results and results['documents'] and results['documents'][0]:
                documents = results['documents'][0]  # First query result
                
                # Check if metadatas exists in results
                if 'metadatas' in results and results['metadatas'] and results['metadatas'][0]:
                    metadatas = results['metadatas'][0]
                    sources = [metadata.get('source', 'Unknown') for metadata in metadatas]
                else:
                    self.logger.warning("No metadata found in results")
                    sources = ["Unknown"] * len(documents)
                
                for i, doc in enumerate(documents):
                    self.logger.info(f"Match {i+1}: {doc[:50]}...")
                
                return documents, sources
            else:
                self.logger.warning("No matching documents found in knowledge base")
                self.logger.info(f"Results keys: {results.keys() if results else 'None'}")
                
                # Fallback: return a representative document anyway
                try:
                    # Try to get all documents
                    all_ids = self.collection.get(limit=1)
                    if all_ids and 'documents' in all_ids and all_ids['documents']:
                        self.logger.info("Using fallback document from knowledge base")
                        
                        if 'metadatas' in all_ids and all_ids['metadatas']:
                            source = all_ids['metadatas'][0].get('source', 'Unknown')
                        else:
                            source = "Unknown"
                            
                        return [all_ids['documents'][0]], [source]
                except Exception as e:
                    self.logger.error(f"Error getting fallback document: {str(e)}")
            
            return [], []
        
        except Exception as e:
            self.logger.error(f"Error querying knowledge base: {str(e)}")
            return [], []
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of the knowledge base."""
        document_count = len(list_documents(self.documents_dir))
        
        try:
            self._ensure_collection()
            vector_count = self.collection.count()
        except Exception as e:
            self.logger.error(f"Error getting vector count: {str(e)}")
            vector_count = 0
        
        return {
            "document_count": document_count,
            "vector_count": vector_count,
            "documents": list_documents(self.documents_dir)
        }

def get_knowledge_base(tenant_id: Optional[str]) -> KnowledgeBase:
    """Get or create a cached knowledge base for a tenant."""
    tid = tenant_id or "default"
    if tid not in _kb_cache:
        _kb_cache[tid] = KnowledgeBase(tenant_id=tid)
    return _kb_cache[tid]