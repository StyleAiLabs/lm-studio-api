# Business AI API

A knowledge-enhanced AI API that leverages LM Studio for generating responses based on custom business documents.

## Architecture

The Business AI System uses a layered architecture to provide AI-powered knowledge retrieval and generation services:

[Business AI System Architecture]

![image](https://github.com/user-attachments/assets/5e26e616-3262-4a52-930e-88954a251190)


### System Components

#### Client Applications
Various frontends and integrations can connect to the Business AI API:
- Web interface
- Mobile applications
- n8n workflow automation
- CRM systems integration

#### FastAPI Wrapper
The API layer exposing endpoints for:
- Chat interactions with business context
- Text completion with knowledge enhancement
- Document upload and management
- Knowledge base administration

#### Service Layer
Core business logic:
- **LLM Service**: Manages interactions with AI models, including context enhancement and prompt engineering
- **Knowledge Service**: Handles document processing, vector search, and retrieval

#### Data Layer
Storage components:
- **Vector Store**: Stores document embeddings for semantic search (ChromaDB/FAISS)
- **Document Storage**: Manages original document files and web content

#### LM Studio
Foundation model layer:
- Hosts open source models like Llama, Mistral
- Provides inference capabilities via API

### Key Features

- Knowledge-augmented responses using business-specific document base
- Persona-based response generation
- Document management (upload, delete, rebuild)
- Website content import
- Semantic search capabilities
- Extensible for various business use cases

## Getting Started

### Prerequisites
- Python 3.8+
- LM Studio running locally or accessible via URL
- Sufficient storage for vector database and documents

### Installation

1. Clone the repository:
```bash
git clone https://github.com/StyleAiLabs/lm-studio-api.git
cd business-ai-api
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables (see .env.example)
```bash
cp .env.example .env
# Edit .env with your configuration
```

4. Run the application:
```bash
uvicorn app.main:app --reload
```

5. Access the API Docs
```
http://localhost:8000/docs#/
```

## API Usage

### Chat Endpoint
```
POST /api/chat
{
  "messages": [
    {"role": "user", "content": "What is our return policy?"}
  ],
  "use_knowledge_base": true,
  "persona": "professional"
}
```

### Adding Website Content
```
POST /api/knowledge/add-website
{
  "url": "https://example.com/about-us"
}
```

### Document Upload
```
POST /api/knowledge/upload
# Form with file upload
```

## Extending the System

The modular architecture makes it easy to extend functionality:
- Add new document processors for additional file types
- Create custom personas for different use cases
- Implement additional data sources for the knowledge base
- Connect to different LLM providers beyond LM Studio
