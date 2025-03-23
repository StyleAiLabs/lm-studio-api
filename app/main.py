from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Body, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import os
from typing import List, Optional

from app.models import (
    CompletionRequest, 
    CompletionResponse, 
    ChatRequest, 
    ChatResponse, 
    ChatMessage,
    DocumentUploadResponse,
    KnowledgeBaseStatusResponse
)
from app.services.llm_service import LLMService
from app.services.knowledge_service import KnowledgeBase
from app.utils.document_processor import save_uploaded_file, delete_document, list_documents
from app.config import DOCUMENTS_DIR, LM_STUDIO_URL

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Business AI API",
    description="API for business knowledge-enhanced AI using LM Studio",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
# First create the knowledge base that will be shared
kb_service = KnowledgeBase()
# Then pass it to the LLM service
llm_service = LLMService(knowledge_base=kb_service)

logger.info("Services initialized with shared knowledge base")

@app.get("/health")
def health_check():
    """Check if the API is running and can connect to LM Studio."""
    return {"status": "healthy", "lm_studio_url": llm_service.base_url}

@app.post("/api/completion", response_model=CompletionResponse)
async def create_completion(request: CompletionRequest):
    """Generate text completion with optional knowledge base context."""
    try:
        response_text, sources = llm_service.generate_completion(
            request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            use_knowledge_base=request.use_knowledge_base
        )
        
        # Format source document paths for display
        formatted_sources = [os.path.basename(source) for source in sources] if sources else None
        
        return CompletionResponse(
            text=response_text, 
            source_documents=formatted_sources
        )
    except Exception as e:
        logging.error(f"Error in completion endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat", response_model=ChatResponse)
async def create_chat_completion(request: ChatRequest):
    """Generate chat completion with optional knowledge base context."""
    try:
        # Extract the last user message for knowledge base query
        last_user_message = next((msg.content for msg in reversed(request.messages) 
                       if msg.role == "user"), "")
        
        logging.info(f"Chat request received with use_knowledge_base={request.use_knowledge_base}")
        logging.info(f"Last user message: '{last_user_message[:50]}...'")
        
        sources = []
        
        # Direct access approach
        if request.use_knowledge_base and last_user_message:
            logging.info("Knowledge base enabled, accessing directly...")
            
            # Direct approach to query the knowledge base
            try:
                # Use the shared kb_service
                contexts, sources = kb_service.query(last_user_message)
                
                # Log found documents
                if contexts and len(contexts) > 0:
                    logging.info(f"Found {len(contexts)} matching documents")
                    for i, ctx in enumerate(contexts):
                        logging.info(f"Context {i+1} (first 100 chars): {ctx[:100]}...")
                        if i < len(sources):
                            logging.info(f"Source: {sources[i]}")
                    
                    # If we have contexts, use enhanced completion
                    context_text = "\n\n".join(contexts)
                    
                    # Create knowledge-enhanced prompt
                    prompt = f"""You are answering a question about company policies and information.
                    
IMPORTANT: You MUST ONLY use the information provided below to answer the question.
If the information doesn't contain the answer, say "I don't have that specific information in my knowledge base."

COMPANY INFORMATION:
{context_text}

QUESTION: {last_user_message}

YOUR ANSWER (using ONLY the provided company information):"""
                    
                    # Call completion API directly
                    url = f"{LM_STUDIO_URL}/completions"
                    payload = {
                        "prompt": prompt,
                        "max_tokens": request.max_tokens,
                        "temperature": 0.3,
                        "stream": False
                    }
                    
                    import requests
                    response = requests.post(url, json=payload)
                    
                    if response.status_code == 200:
                        answer_text = response.json()["choices"][0]["text"]
                        logging.info(f"Generated answer: {answer_text[:100]}...")
                        
                        # Format source document paths for display
                        formatted_sources = [os.path.basename(source) for source in sources]
                        
                        return ChatResponse(
                            message=ChatMessage(role="assistant", content=answer_text),
                            source_documents=formatted_sources
                        )
                else:
                    logging.warning("No contexts found for query")
                        
            except Exception as e:
                logging.error(f"Error with direct KB access: {str(e)}")
        
        # If we get here, either knowledge base is disabled or we didn't find matches
        # Fall back to regular chat API
        logging.info("Falling back to standard chat API (no knowledge context)")
        api_messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        url = f"{LM_STUDIO_URL}/chat/completions"
        
        payload = {
            "messages": api_messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "stream": False
        }
        
        import requests
        response = requests.post(url, json=payload)
        
        if response.status_code == 200:
            answer = response.json()["choices"][0]["message"]["content"]
            return ChatResponse(
                message=ChatMessage(role="assistant", content=answer),
                source_documents=None
            )
        else:
            logging.error(f"Error from chat API: {response.text}")
            raise Exception(f"Error calling LM Studio API: {response.text}")
            
    except Exception as e:
        logging.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/knowledge/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload a document to the knowledge base."""
    try:
        # Read file content
        file_content = await file.read()
        
        # Save the file
        file_path = save_uploaded_file(file_content, file.filename, DOCUMENTS_DIR)
        
        # Add to knowledge base
        logger.info(f"Adding document to knowledge base: {file.filename}")
        success = kb_service.add_document(file_path)
        
        if success:
            logger.info(f"Document successfully added: {file.filename}")
            return DocumentUploadResponse(
                status="success",
                filename=file.filename,
                message="Document uploaded and processed successfully"
            )
        else:
            logger.warning(f"Document saved but processing failed: {file.filename}")
            return DocumentUploadResponse(
                status="warning",
                filename=file.filename,
                message="Document saved but could not be processed"
            )
    except Exception as e:
        logging.error(f"Error uploading document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/knowledge/documents/{filename}")
async def remove_document(filename: str):
    """Remove a document from the knowledge base."""
    try:
        success = delete_document(filename, DOCUMENTS_DIR)
        
        if success:
            # Rebuild the knowledge base
            logger.info(f"Document deleted, rebuilding knowledge base: {filename}")
            kb_service.rebuild_knowledge_base()
            
            return {"status": "success", "message": f"Document {filename} removed successfully"}
        else:
            raise HTTPException(status_code=404, detail=f"Document {filename} not found")
    except Exception as e:
        logging.error(f"Error removing document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/knowledge/rebuild")
async def rebuild_knowledge_base():
    """Rebuild the entire knowledge base from documents."""
    try:
        logger.info("Rebuilding knowledge base")
        success = kb_service.rebuild_knowledge_base()
        
        if success:
            logger.info("Knowledge base rebuilt successfully")
            return {"status": "success", "message": "Knowledge base rebuilt successfully"}
        else:
            logger.warning("Knowledge base rebuilt with warnings")
            return {"status": "warning", "message": "Knowledge base rebuilt with warnings"}
    except Exception as e:
        logging.error(f"Error rebuilding knowledge base: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/knowledge/status", response_model=KnowledgeBaseStatusResponse)
async def get_knowledge_base_status():
    """Get status of the knowledge base."""
    try:
        status = kb_service.get_status()
        logger.info(f"Knowledge base status: {status['document_count']} documents, {status['vector_count']} vectors")
        
        return KnowledgeBaseStatusResponse(
            status="active" if status["vector_count"] > 0 else "empty",
            document_count=status["document_count"],
            vector_count=status["vector_count"],
            documents=status["documents"]
        )
    except Exception as e:
        logging.error(f"Error getting knowledge base status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/debug/knowledge")
async def debug_knowledge(query: str):
    """Debug endpoint to directly test knowledge retrieval."""
    contexts, sources = kb_service.query(query)
    
    return {
        "query": query,
        "found_matches": len(contexts) > 0,
        "contexts": contexts,
        "sources": sources
    }