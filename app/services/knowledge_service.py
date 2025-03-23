import os
import logging
import shutil
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions
import pypdf
import docx2txt

from app.config import DOCUMENTS_DIR, VECTORSTORE_DIR, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL
from app.utils.document_processor import list_documents

class KnowledgeBase:
    def __init__(self, 
                 documents_dir: str = DOCUMENTS_DIR,
                 vectorstore_dir: str = VECTORSTORE_DIR,
                 embedding_model: str = EMBEDDING_MODEL,
                 chunk_size: int = CHUNK_SIZE,
                 chunk_overlap: int = CHUNK_OVERLAP):
        
        self.documents_dir = documents_dir
        self.vectorstore_dir = vectorstore_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Configure logging
        logging.basicConfig(level=logging.INFO, 
                           format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Create directories if they don't exist
        os.makedirs(self.documents_dir, exist_ok=True)
        os.makedirs(self.vectorstore_dir, exist_ok=True)
        
        # Set up embeddings and chroma client
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )
        
        self.client = chromadb.PersistentClient(path=self.vectorstore_dir)
        
        # Create or get collection
        try:
            self.collection = self.client.get_collection(
                name="business_knowledge",
                embedding_function=self.embedding_function
            )
            logging.info(f"Loaded existing collection with {self.collection.count()} documents")
        except ValueError:
            self.collection = self.client.create_collection(
                name="business_knowledge",
                embedding_function=self.embedding_function
            )
            logging.info("Created new collection")
    
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
                logging.warning(f"Unsupported file type: {ext}")
                return None
        
        except Exception as e:
            logging.error(f"Error extracting text from {file_path}: {str(e)}")
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
        
        logging.info(f"Split document into {len(chunks)} chunks")
        return chunks
    
    def add_document(self, file_path: str) -> bool:
        """Add a single document to the knowledge base."""
        try:
            text = self._extract_text_from_file(file_path)
            
            if not text:
                logging.warning(f"Could not extract text from {file_path}")
                return False
            
            chunks = self._chunk_text(text, file_path)
            
            if not chunks:
                logging.warning(f"No chunks created from {file_path}")
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
            
            logging.info(f"Added {len(chunks)} chunks from {file_path} to knowledge base")
            return True
        
        except Exception as e:
            logging.error(f"Error adding document {file_path}: {str(e)}")
            return False
    
    def rebuild_knowledge_base(self) -> bool:
        """Rebuild the entire knowledge base from documents."""
        try:
            # Reset collection
            try:
                self.client.delete_collection("business_knowledge")
                logging.info("Deleted existing collection")
            except ValueError:
                logging.info("No existing collection to delete")
                
            self.collection = self.client.create_collection(
                name="business_knowledge",
                embedding_function=self.embedding_function
            )
            
            # Add all documents
            success = True
            documents = list_documents(self.documents_dir)
            logging.info(f"Found {len(documents)} documents to process")
            
            for file in documents:
                file_path = os.path.join(self.documents_dir, file)
                if not self.add_document(file_path):
                    logging.warning(f"Failed to process {file}")
                    success = False
            
            return success
        
        except Exception as e:
            logging.error(f"Error rebuilding knowledge base: {str(e)}")
            return False
    
    def query(self, query_text: str, k: int = 3) -> Tuple[List[str], List[str]]:
        """
        Retrieve relevant documents for a query.
        
        Returns:
            Tuple of (content_list, source_list)
        """
        try:
            # Try to ensure we get results
            results = self.collection.query(
                query_texts=[query_text],
                n_results=k,
                include_distances=True
            )
            
            logging.info(f"Query: '{query_text}'")
            
            # Check if we have results
            if results and 'documents' in results and results['documents'] and results['documents'][0]:
                documents = results['documents'][0]  # First query result
                metadatas = results['metadatas'][0]  # First query result
                distances = results['distances'][0] if 'distances' in results else None
                
                # Log distances if available
                if distances:
                    for i, (doc, dist) in enumerate(zip(documents, distances)):
                        logging.info(f"Match {i+1} (distance={dist:.4f}): {doc[:50]}...")
                
                sources = [metadata.get('source', 'Unknown') for metadata in metadatas]
                
                return documents, sources
            else:
                logging.warning("No matching documents found in knowledge base")
                
                # Fallback: return a representative document anyway
                try:
                    # Try to get all documents
                    all_ids = self.collection.get(limit=1)
                    if all_ids and 'documents' in all_ids and all_ids['documents']:
                        logging.info("Using fallback document from knowledge base")
                        return [all_ids['documents'][0]], [all_ids['metadatas'][0].get('source', 'Unknown')]
                except Exception as e:
                    logging.error(f"Error getting fallback document: {str(e)}")
            
            return [], []
        
        except Exception as e:
            logging.error(f"Error querying knowledge base: {str(e)}")
            return [], []
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of the knowledge base."""
        document_count = len(list_documents(self.documents_dir))
        
        try:
            # Get count from collection
            vector_count = self.collection.count()
        except Exception as e:
            logging.error(f"Error getting vector count: {str(e)}")
            vector_count = 0
        
        return {
            "document_count": document_count,
            "vector_count": vector_count,
            "documents": list_documents(self.documents_dir)
        }