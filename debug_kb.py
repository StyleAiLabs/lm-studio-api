import os
import sys
import logging
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions
import pypdf
import docx2txt

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Directory paths - adjust as needed
DOCUMENTS_DIR = "./data/documents"
VECTORSTORE_DIR = "./data/vectorstore"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

def list_documents(directory):
    """List all documents in the given directory."""
    if not os.path.exists(directory):
        logger.error(f"Directory doesn't exist: {directory}")
        return []
        
    all_files = []
    for ext in ["*.txt", "*.pdf", "*.docx"]:
        import glob
        all_files.extend(glob.glob(os.path.join(directory, ext)))
    
    filenames = [os.path.basename(file) for file in all_files]
    return filenames, all_files

def extract_text(file_path):
    """Extract text from a file."""
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
            logger.warning(f"Unsupported file type: {ext}")
            return None
    
    except Exception as e:
        logger.error(f"Error extracting text from {file_path}: {str(e)}")
        return None

def chunk_text(text, source, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
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
            
        if len(current_chunk) + len(paragraph) <= chunk_size:
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
                if len(words) > chunk_overlap:
                    current_chunk = " ".join(words[-chunk_overlap:]) + "\n\n"
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
    
    logger.info(f"Split document into {len(chunks)} chunks")
    return chunks

def initialize_chromadb():
    """Initialize and return a ChromaDB client and collection."""
    logger.info("Initializing ChromaDB...")
    
    # Create directories
    os.makedirs(DOCUMENTS_DIR, exist_ok=True)
    os.makedirs(VECTORSTORE_DIR, exist_ok=True)
    
    # Setup embedding function
    try:
        embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )
        logger.info(f"Created embedding function with model: {EMBEDDING_MODEL}")
    except Exception as e:
        logger.error(f"Error creating embedding function: {str(e)}")
        return None, None
    
    # Create client
    try:
        client = chromadb.PersistentClient(path=VECTORSTORE_DIR)
        logger.info(f"Created ChromaDB client with path: {VECTORSTORE_DIR}")
    except Exception as e:
        logger.error(f"Error creating ChromaDB client: {str(e)}")
        return None, None
    
    # Get or create collection
    try:
        try:
            collection = client.get_collection(
                name="business_knowledge",
                embedding_function=embedding_function
            )
            logger.info(f"Found existing collection with {collection.count()} documents")
        except ValueError:
            collection = client.create_collection(
                name="business_knowledge",
                embedding_function=embedding_function
            )
            logger.info("Created new collection")
        
        return client, collection
    except Exception as e:
        logger.error(f"Error getting/creating collection: {str(e)}")
        return client, None

def rebuild_collection(client, collection):
    """Delete and recreate the collection."""
    try:
        if collection:
            logger.info("Deleting existing collection...")
            client.delete_collection("business_knowledge")
        
        embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )
        
        collection = client.create_collection(
            name="business_knowledge",
            embedding_function=embedding_function
        )
        logger.info("Created fresh collection")
        return collection
    except Exception as e:
        logger.error(f"Error rebuilding collection: {str(e)}")
        return None

def add_document_to_collection(collection, file_path):
    """Add a single document to the collection."""
    try:
        # Extract text
        text = extract_text(file_path)
        if not text:
            logger.warning(f"Could not extract text from {file_path}")
            return False
        
        # Chunk text
        chunks = chunk_text(text, file_path)
        if not chunks:
            logger.warning(f"No chunks created from {file_path}")
            return False
        
        # Prepare data for Chroma
        ids = [chunk["id"] for chunk in chunks]
        texts = [chunk["text"] for chunk in chunks]
        metadatas = [{"source": chunk["source"]} for chunk in chunks]
        
        # Add to collection
        collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas
        )
        
        logger.info(f"Added {len(chunks)} chunks from {file_path} to collection")
        return True
    except Exception as e:
        logger.error(f"Error adding document {file_path}: {str(e)}")
        return False

def query_collection(collection, query_text, k=3):
    """Query the collection for relevant documents."""
    try:
        # Query the collection
        results = collection.query(
            query_texts=[query_text],
            n_results=k,
            include_distances=True
        )
        
        logger.info(f"Query: '{query_text}'")
        
        # Check if we have results
        if results and len(results['documents']) > 0 and len(results['documents'][0]) > 0:
            documents = results['documents'][0]  # First query result
            metadatas = results['metadatas'][0]  # First query result
            distances = results['distances'][0] if 'distances' in results else None
            
            # Log results
            logger.info(f"Query returned {len(documents)} results")
            
            if distances:
                for i, (doc, dist) in enumerate(zip(documents, distances)):
                    logger.info(f"Result {i+1} (distance={dist:.4f}): {doc[:100]}...")
                    logger.info(f"  Source: {metadatas[i].get('source', 'Unknown')}")
            
            sources = [metadata.get('source', 'Unknown') for metadata in metadatas]
            return documents, sources
        else:
            logger.warning("No results found")
            return [], []
    except Exception as e:
        logger.error(f"Error querying collection: {str(e)}")
        return [], []

def inspect_vectorstore():
    """Inspect the vectorstore directory structure."""
    logger.info(f"Inspecting vectorstore at: {VECTORSTORE_DIR}")
    
    if not os.path.exists(VECTORSTORE_DIR):
        logger.error(f"Vectorstore directory doesn't exist: {VECTORSTORE_DIR}")
        return
    
    try:
        # List files recursively
        for root, dirs, files in os.walk(VECTORSTORE_DIR):
            level = root.replace(VECTORSTORE_DIR, '').count(os.sep)
            indent = ' ' * 4 * level
            logger.info(f"{indent}{os.path.basename(root)}/")
            sub_indent = ' ' * 4 * (level + 1)
            for f in files:
                size = os.path.getsize(os.path.join(root, f))
                logger.info(f"{sub_indent}{f} ({size} bytes)")
    except Exception as e:
        logger.error(f"Error inspecting vectorstore: {str(e)}")

def inspect_chroma_collection(collection):
    """Inspect the collection contents."""
    try:
        # Get all IDs
        all_ids = collection.get()
        
        if all_ids and len(all_ids.get('ids', [])) > 0:
            logger.info(f"Collection contains {len(all_ids['ids'])} documents")
            
            # Show sample
            sample_size = min(5, len(all_ids['ids']))
            logger.info(f"Sample of {sample_size} documents:")
            
            for i in range(sample_size):
                doc_id = all_ids['ids'][i]
                doc_text = all_ids['documents'][i]
                doc_source = all_ids['metadatas'][i].get('source', 'Unknown')
                
                logger.info(f"Document {i+1}:")
                logger.info(f"  ID: {doc_id}")
                logger.info(f"  Source: {doc_source}")
                logger.info(f"  Text (first 100 chars): {doc_text[:100]}...")
        else:
            logger.warning("Collection appears to be empty")
    except Exception as e:
        logger.error(f"Error inspecting collection: {str(e)}")

def main():
    """Main function to debug the knowledge base."""
    logger.info("=== Knowledge Base Debug Script ===")
    
    # Check documents
    logger.info("\n=== Checking Documents ===")
    filenames, file_paths = list_documents(DOCUMENTS_DIR)
    if filenames:
        logger.info(f"Found {len(filenames)} documents: {', '.join(filenames)}")
        
        # Sample document contents
        if file_paths:
            sample_path = file_paths[0]
            sample_text = extract_text(sample_path)
            if sample_text:
                logger.info(f"Sample text from {os.path.basename(sample_path)} (first 200 chars):")
                logger.info(sample_text[:200] + "...")
    else:
        logger.warning("No documents found")
    
    # Initialize ChromaDB
    logger.info("\n=== Initializing ChromaDB ===")
    client, collection = initialize_chromadb()
    
    if not client or not collection:
        logger.error("Failed to initialize ChromaDB")
        return
    
    # Inspect current vectorstore
    logger.info("\n=== Inspecting Vectorstore Directory ===")
    inspect_vectorstore()
    
    # Inspect collection
    logger.info("\n=== Inspecting Collection ===")
    inspect_chroma_collection(collection)
    
    # Ask for user action
    action = input("\nChoose action:\n1. Rebuild collection from scratch\n2. Add missing documents to collection\n3. Query collection\n4. Exit\nChoice: ")
    
    if action == "1":
        logger.info("\n=== Rebuilding Collection ===")
        collection = rebuild_collection(client, collection)
        
        if collection:
            for file_path in file_paths:
                logger.info(f"Adding document: {os.path.basename(file_path)}")
                add_document_to_collection(collection, file_path)
            
            logger.info("\n=== Inspecting New Collection ===")
            inspect_chroma_collection(collection)
    
    elif action == "2":
        logger.info("\n=== Adding Missing Documents ===")
        if collection.count() == 0:
            logger.info("Collection is empty, adding all documents")
            for file_path in file_paths:
                add_document_to_collection(collection, file_path)
        else:
            # More sophisticated logic could be added here
            for file_path in file_paths:
                add_document_to_collection(collection, file_path)
        
        logger.info("\n=== Inspecting Updated Collection ===")
        inspect_chroma_collection(collection)
    
    elif action == "3":
        logger.info("\n=== Testing Queries ===")
        if collection.count() == 0:
            logger.warning("Collection is empty, cannot query")
            return
        
        test_queries = [
            "What is the return policy?",
            "How long do I have to return items?",
            "Can I return electronics?",
            "What are the special conditions for returns?"
        ]
        
        for query in test_queries:
            logger.info(f"\nTesting query: '{query}'")
            documents, sources = query_collection(collection, query)
            
            if documents and sources:
                logger.info(f"Found {len(documents)} matches")
            else:
                logger.warning("No matches found")
    
    logger.info("\n=== Debug Complete ===")

if __name__ == "__main__":
    main()