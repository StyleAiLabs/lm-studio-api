from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .models import CompletionRequest, CompletionResponse, ChatRequest, ChatResponse, ChatMessage
from .services.llm_service import LLMService

app = FastAPI(title="Small Business AI API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

llm_service = LLMService()

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/api/completion", response_model=CompletionResponse)
async def create_completion(request: CompletionRequest):
    try:
        response = llm_service.generate_completion(
            request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        return CompletionResponse(text=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat", response_model=ChatResponse)
async def create_chat_completion(request: ChatRequest):
    try:
        response = llm_service.generate_chat_completion(
            request.messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        return ChatResponse(message=ChatMessage(role="assistant", content=response))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))