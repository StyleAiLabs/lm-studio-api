from pydantic import BaseModel
from typing import List, Optional

class CompletionRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = 200
    temperature: Optional[float] = 0.7
    
class CompletionResponse(BaseModel):
    text: str
    
class ChatMessage(BaseModel):
    role: str
    content: str
    
class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 200
    temperature: Optional[float] = 0.7
    
class ChatResponse(BaseModel):
    message: ChatMessage