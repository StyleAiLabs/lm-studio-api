from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Body, Query, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import os
from typing import List, Optional
import requests

from app.models import (
    CompletionRequest, 
    CompletionResponse, 
    ChatRequest, 
    ChatResponse, 
    ChatMessage,
    DocumentUploadResponse,
    KnowledgeBaseStatusResponse,
    WebsiteUploadRequest
)
from app.services.llm_service import LLMService
from app.services.knowledge_service import get_knowledge_base
from app.utils.document_processor import save_uploaded_file, delete_document, list_documents
from app.utils.prompt_builder import build_knowledge_prompt, build_regular_chat_prompt
from app.utils.personas import get_persona
from app.utils.web_scraper import scrape_website, WebsiteScrapeForbidden
from app.config import DOCUMENTS_DIR, LM_STUDIO_URL

# Context size limits to prevent token overflow
MAX_CONTEXT_CHARS = 2000  # Approximately 500 tokens
ESTIMATED_CHARS_PER_TOKEN = 4  # Rough estimate

def truncate_contexts(contexts, max_total_chars=MAX_CONTEXT_CHARS):
    """Limit total context size to fit within token limits"""
    result = []
    total_chars = 0
    
    for ctx in contexts:
        # If adding this context would exceed our limit
        if total_chars + len(ctx) > max_total_chars:
            # If we have no contexts yet, add a truncated version of this one
            if not result:
                truncated = ctx[:max_total_chars]
                result.append(truncated)
                logging.info(f"Truncated single context to {len(truncated)} chars")
            break
        
        # Otherwise add the full context
        result.append(ctx)
        total_chars += len(ctx)
        
    logging.info(f"Reduced contexts from {len(contexts)} to {len(result)} to fit token limit")
    return result

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

# Initialize tenant-aware LLM service
llm_service = LLMService()
logger.info("Services initialized (tenant-aware)")

@app.get("/health")
def health_check():
    """Check if the API is running and can connect to LM Studio."""
    return {"status": "healthy", "lm_studio_url": llm_service.base_url}

@app.post("/api/completion", response_model=CompletionResponse)
async def create_completion(request: CompletionRequest, x_tenant_id: Optional[str] = Header(None)):
    """Generate text completion with optional knowledge base context."""
    try:
        tenant_id = request.tenant_id or x_tenant_id
        response_text, sources = llm_service.generate_completion(
            request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            use_knowledge_base=request.use_knowledge_base,
            persona=request.persona if hasattr(request, 'persona') else "default",
            tenant_id=tenant_id
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
async def create_chat_completion(request: ChatRequest, x_tenant_id: Optional[str] = Header(None)):
    """Generate chat completion with optional knowledge base context."""
    try:
        # Extract the last user message for knowledge base query
        last_user_message = next((msg.content for msg in reversed(request.messages) 
                       if msg.role == "user"), "")
        
        # Get persona if available (default to "default" if not specified)
        persona = getattr(request, 'persona', "default")
        
        logging.info(f"Chat request received with use_knowledge_base={request.use_knowledge_base}, persona={persona}")
        logging.info(f"Last user message: '{last_user_message[:50]}...'")
        
        tenant_id = request.tenant_id or x_tenant_id
        # Delegate to service (which will choose best path incl. KB+completion fallback)
        answer_text, sources = llm_service.generate_chat_completion(
            request.messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            use_knowledge_base=request.use_knowledge_base,
            persona=persona,
            tenant_id=tenant_id
        )

        formatted_sources = [os.path.basename(s) for s in sources] if sources else None
        return ChatResponse(
            message=ChatMessage(role="assistant", content=answer_text),
            source_documents=formatted_sources
        )
            
    except Exception as e:
        logging.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/knowledge/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...), tenant_id: Optional[str] = Form(None), x_tenant_id: Optional[str] = Header(None)):
    """Upload a document to the knowledge base."""
    try:
        # Read file content
        file_content = await file.read()

        # Save the file
        tid = tenant_id or x_tenant_id or "default"
        docs_dir = os.path.join(DOCUMENTS_DIR, tid)
        file_path = save_uploaded_file(file_content, file.filename, docs_dir)

        # Add to tenant knowledge base
        kb = get_knowledge_base(tid)
        logger.info(f"Adding document to knowledge base (tenant={tid}): {file.filename}")
        success = kb.add_document(file_path)
        
        if success:
            logger.info(f"Document successfully added: {file.filename}")
            return DocumentUploadResponse(
                status="success",
                filename=file.filename,
                message="Document uploaded and processed successfully",
                tenant_id=tid
            )
        else:
            logger.warning(f"Document saved but processing failed: {file.filename}")
            return DocumentUploadResponse(
                status="warning",
                filename=file.filename,
                message="Document saved but could not be processed",
                tenant_id=tid
            )
    except Exception as e:
        logging.error(f"Error uploading document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/knowledge/add-website", response_model=DocumentUploadResponse)
async def add_website(request: WebsiteUploadRequest, tenant_id: Optional[str] = Query(None), x_tenant_id: Optional[str] = Header(None)):
    """Add website content to the knowledge base."""
    try:
        url = request.url
        logger.info(f"Scraping website: {url}")
        tid = request.tenant_id or tenant_id or x_tenant_id or "default"
        docs_dir = os.path.join(DOCUMENTS_DIR, tid)
        file_path = scrape_website(url, docs_dir)
        
        if not file_path:
            raise HTTPException(status_code=400, detail="Failed to scrape website")
        # Add to knowledge base
        filename = os.path.basename(file_path)
        logger.info(f"Adding scraped content to knowledge base: {filename}")
        kb = get_knowledge_base(tid)
        success = kb.add_document(file_path)
        
        if success:
            logger.info(f"Website content successfully added: {filename}")
            return DocumentUploadResponse(
                status="success",
                filename=filename,
                message=f"Website content from {url} added successfully",
                tenant_id=tid
            )
        else:
            logger.warning(f"Website content saved but processing failed: {filename}")
            return DocumentUploadResponse(
                status="warning",
                filename=filename,
                message="Website content saved but could not be processed",
                tenant_id=tid
            )
    except WebsiteScrapeForbidden as fe:
        logging.warning(f"Forbidden scraping website: {str(fe)}")
        raise HTTPException(status_code=400, detail=str(fe))
    except HTTPException:
        # Already a proper HTTP error, propagate
        raise
    except Exception as e:
        logging.error(f"Error adding website: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/knowledge/documents/{filename}")
async def remove_document(filename: str, tenant_id: Optional[str] = Query(None), x_tenant_id: Optional[str] = Header(None)):
    """Remove a document from the knowledge base."""
    try:
        tid = tenant_id or x_tenant_id or "default"
        docs_dir = os.path.join(DOCUMENTS_DIR, tid)
        success = delete_document(filename, docs_dir)
        
        if success:
            # Rebuild the knowledge base
            logger.info(f"Document deleted, rebuilding knowledge base: {filename}")
            kb = get_knowledge_base(tid)
            kb.rebuild_knowledge_base()
            
            return {"status": "success", "message": f"Document {filename} removed successfully"}
        else:
            raise HTTPException(status_code=404, detail=f"Document {filename} not found")
    except Exception as e:
        logging.error(f"Error removing document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/knowledge/rebuild")
async def rebuild_knowledge_base(tenant_id: Optional[str] = Query(None), x_tenant_id: Optional[str] = Header(None)):
    """Rebuild the entire knowledge base from documents."""
    try:
        logger.info("Rebuilding knowledge base")
        tid = tenant_id or x_tenant_id or "default"
        kb = get_knowledge_base(tid)
        success = kb.rebuild_knowledge_base()
        
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
async def get_knowledge_base_status(tenant_id: Optional[str] = Query(None), x_tenant_id: Optional[str] = Header(None)):
    """Get status of the knowledge base."""
    try:
        tid = tenant_id or x_tenant_id or "default"
        kb = get_knowledge_base(tid)
        status = kb.get_status()
        logger.info(f"Knowledge base status: {status['document_count']} documents, {status['vector_count']} vectors")
        
        return KnowledgeBaseStatusResponse(
            status="active" if status["vector_count"] > 0 else "empty",
            document_count=status["document_count"],
            vector_count=status["vector_count"],
            documents=status["documents"],
            tenant_id=tid
        )
    except Exception as e:
        logging.error(f"Error getting knowledge base status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/debug/knowledge")
async def debug_knowledge(query: str, tenant_id: Optional[str] = Query(None), x_tenant_id: Optional[str] = Header(None)):
    """Debug endpoint to directly test knowledge retrieval."""
    tid = tenant_id or x_tenant_id or "default"
    kb = get_knowledge_base(tid)
    contexts, sources = kb.query(query)
    
    return {
        "query": query,
        "found_matches": len(contexts) > 0,
        "contexts": contexts,
    "sources": sources,
    "tenant_id": tid
    }