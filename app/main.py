from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Body, Query
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
    KnowledgeBaseStatusResponse
)
from app.services.llm_service import LLMService
from app.services.knowledge_service import KnowledgeBase
from app.utils.document_processor import save_uploaded_file, delete_document, list_documents
from app.utils.prompt_builder import build_knowledge_prompt, build_regular_chat_prompt
from app.utils.personas import get_persona
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
            use_knowledge_base=request.use_knowledge_base,
            persona=request.persona if hasattr(request, 'persona') else "default"
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
        
        # Get persona if available (default to "default" if not specified)
        persona = getattr(request, 'persona', "default")
        
        logging.info(f"Chat request received with use_knowledge_base={request.use_knowledge_base}, persona={persona}")
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
                    
                    # Apply context truncation to prevent token overflow
                    limited_contexts = truncate_contexts(contexts)
                    
                    # If we have contexts, use enhanced completion
                    context_text = "\n\n".join(limited_contexts)
                    logging.info(f"Total context size: {len(context_text)} characters")
                    
                    # Get persona configuration
                    persona_config = get_persona(persona)
                    persona_temp = persona_config.get("temperature", 0.7)
                    
                    # Create knowledge-enhanced prompt with prompt builder
                    prompt = build_knowledge_prompt(context_text, last_user_message, persona)
                    
                    # Log token estimate (rough approximation)
                    estimated_tokens = len(prompt) / ESTIMATED_CHARS_PER_TOKEN
                    logging.info(f"Estimated prompt tokens: {estimated_tokens}")
                    
                    # Call completion API directly
                    url = f"{LM_STUDIO_URL}/completions"
                    payload = {
                        "prompt": prompt,
                        "max_tokens": request.max_tokens,
                        "temperature": persona_temp,
                        "stream": False
                    }
                    
                    logging.info(f"Calling LM Studio API at {url}")
                    response = requests.post(url, json=payload)
                    
                    logging.info(f"LM Studio API response status: {response.status_code}")
                    if response.status_code != 200:
                        logging.error(f"LM Studio API error: {response.text}")
                    
                    if response.status_code == 200:
                        # Log the entire response structure to debug
                        logging.info(f"Response JSON keys: {list(response.json().keys())}")
                        
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
                logging.exception("Details:")
        
        # If we get here, either knowledge base is disabled or we didn't find matches
        # Fall back to regular chat API
        logging.info(f"Falling back to standard chat API with persona '{persona}'")
        
        # Convert ChatMessage objects to dicts for the API
        api_messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        
        # Use the prompt builder to enhance with persona
        enhanced_messages = build_regular_chat_prompt(api_messages, persona)
        
        url = f"{LM_STUDIO_URL}/chat/completions"
        
        # Get persona temperature
        persona_config = get_persona(persona)
        persona_temp = persona_config.get("temperature", request.temperature)
        
        payload = {
            "messages": enhanced_messages,
            "max_tokens": request.max_tokens,
            "temperature": persona_temp,
            "stream": False
        }
        
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