import os
import logging
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration values from your app
VECTORSTORE_DIR = "./data/vectorstore"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

def load_kb_and_query(query_text):
    """Load the knowledge base and query it directly."""
    logger.info(f"Loading knowledge base from: {VECTORSTORE_DIR}")
    
    try:
        # Set up embedding function
        embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )
        
        # Create client
        client = chromadb.PersistentClient(path=VECTORSTORE_DIR)
        
        # Check if collection exists
        try:
            collection = client.get_collection(
                name="business_knowledge",
                embedding_function=embedding_function
            )
            count = collection.count()
            logger.info(f"Found collection with {count} documents")
        except ValueError as e:
            logger.error(f"Collection not found: {str(e)}")
            return False
        
        # Query the collection - without include_distances
        logger.info(f"Querying with: '{query_text}'")
        results = collection.query(
            query_texts=[query_text],
            n_results=3
        )
        
        # Check results
        if results and 'documents' in results and results['documents'] and results['documents'][0]:
            documents = results['documents'][0]
            metadatas = results['metadatas'][0] if 'metadatas' in results and results['metadatas'] else []
            
            logger.info(f"Query returned {len(documents)} results")
            
            for i, doc in enumerate(documents):
                logger.info(f"Match {i+1}: {doc[:50]}...")
                if i < len(metadatas):
                    source = metadatas[i].get('source', 'Unknown')
                    logger.info(f"  Source: {os.path.basename(source)}")
            
            sources = [metadata.get('source', 'Unknown') for metadata in metadatas] if metadatas else []
            return documents, sources
        else:
            logger.warning("No matches found in result")
            logger.info(f"Raw result structure: {results.keys()}")
            return [], []
            
    except Exception as e:
        logger.error(f"Error querying knowledge base: {str(e)}")
        return [], []

def simulate_chat_endpoint():
    """Simulate how the chat endpoint uses the knowledge base."""
    logger.info("=== Simulating Chat Endpoint Knowledge Base Usage ===")
    
    # Test queries - including ones related to the sample document
    test_queries = [
        "What is our company's return policy?",
        "How many days do I have to return an item?",
        "Can I return electronics?",
        "What is the process for returns?",
        "Tell me about seasonal items returns"
    ]
    
    for query in test_queries:
        logger.info(f"\nTesting query: '{query}'")
        documents, sources = load_kb_and_query(query)
        
        if documents and sources:
            logger.info(f"SUCCESS: Found {len(documents)} matching documents")
            for source in sources:
                logger.info(f"Source: {os.path.basename(source)}")
        else:
            logger.warning("FAILURE: No matching documents found")
            
    logger.info("\n=== Simulation Complete ===")

if __name__ == "__main__":
    simulate_chat_endpoint()