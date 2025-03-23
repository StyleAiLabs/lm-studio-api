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
from app.config import DOCUMENTS_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

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
llm_service = LLMService()
kb_service = KnowledgeBase()

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
        response_text, sources = llm_service.generate_chat_completion(
            request.messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            use_knowledge_base=request.use_knowledge_base
        )
        
        # Format source document paths for display
        formatted_sources = [os.path.basename(source) for source in sources] if sources else None
        
        return ChatResponse(
            message=ChatMessage(role="assistant", content=response_text),
            source_documents=formatted_sources
        )
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
        success = kb_service.add_document(file_path)
        
        if success:
            return DocumentUploadResponse(
                status="success",
                filename=file.filename,
                message="Document uploaded and processed successfully"
            )
        else:
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
        success = kb_service.rebuild_knowledge_base()
        
        if success:
            return {"status": "success", "message": "Knowledge base rebuilt successfully"}
        else:
            return {"status": "warning", "message": "Knowledge base rebuilt with warnings"}
    except Exception as e:
        logging.error(f"Error rebuilding knowledge base: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/knowledge/status", response_model=KnowledgeBaseStatusResponse)
async def get_knowledge_base_status():
    """Get status of the knowledge base."""
    try:
        status = kb_service.get_status()
        
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