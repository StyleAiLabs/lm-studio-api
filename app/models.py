from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class CompletionRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = 200
    temperature: Optional[float] = 0.7
    use_knowledge_base: Optional[bool] = True
    persona: Optional[str] = "default"  # Added persona field
    
class CompletionResponse(BaseModel):
    text: str
    source_documents: Optional[List[str]] = None
    
class ChatMessage(BaseModel):
    role: str
    content: str
    
class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 200
    temperature: Optional[float] = 0.7
    use_knowledge_base: Optional[bool] = True
    persona: Optional[str] = "default"  # Added persona field
    
class ChatResponse(BaseModel):
    message: ChatMessage
    source_documents: Optional[List[str]] = None

class DocumentUploadResponse(BaseModel):
    status: str
    filename: str
    message: str

class KnowledgeBaseStatusResponse(BaseModel):
    status: str
    document_count: int
    vector_count: int
    documents: List[str]