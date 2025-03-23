import requests
import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from app.config import LM_STUDIO_URL
from app.services.knowledge_service import KnowledgeBase
from app.models import ChatMessage

class LLMService:
    def __init__(self, knowledge_base: KnowledgeBase):
        """
        Initialize LLM Service with a shared Knowledge Base instance.
        
        Args:
            knowledge_base: An instance of KnowledgeBase to use for context retrieval
        """
        self.base_url = LM_STUDIO_URL
        self.kb = knowledge_base  # Use the provided KB instance instead of creating a new one
        # Setup logger
        self.logger = logging.getLogger(__name__)
        self.logger.info("LLMService initialized with shared knowledge base instance")
    
    def generate_completion(
        self, 
        prompt: str, 
        max_tokens: int = 200, 
        temperature: float = 0.7,
        use_knowledge_base: bool = True
    ) -> Tuple[str, Optional[List[str]]]:
        """
        Generate text completion with optional knowledge base context.
        
        Returns:
            Tuple of (generated_text, source_documents)
        """
        context = ""
        sources = []
        
        # Retrieve context from knowledge base if requested
        if use_knowledge_base:
            self.logger.info(f"Knowledge base enabled for completion, querying with: '{prompt[:50]}...'")
            contexts, sources = self.kb.query(prompt)
            if contexts:
                context = "\n\n".join(contexts)
                self.logger.info(f"Retrieved {len(contexts)} contexts for completion")
            else:
                self.logger.warning("Knowledge base query returned no contexts despite being enabled")
        else:
            self.logger.info("Knowledge base disabled for completion request")
        
        # Build the prompt with context if available
        enhanced_prompt = prompt
        if context:
            enhanced_prompt = f"""You are answering a question about company policies and information.
            
IMPORTANT: You MUST ONLY use the information provided below to answer the question.
If the information doesn't contain the answer, say "I don't have that specific information in my knowledge base."

COMPANY INFORMATION:
{context}

QUESTION: {prompt}

YOUR ANSWER (using ONLY the provided company information):"""
        
        # Call LM Studio API
        url = f"{self.base_url}/completions"
        
        payload = {
            "prompt": enhanced_prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False
        }
        
        try:
            response = requests.post(url, json=payload)
            
            if response.status_code == 200:
                return response.json()["choices"][0]["text"], sources
            else:
                self.logger.error(f"Error from LM Studio API: {response.text}")
                raise Exception(f"Error calling LM Studio API: {response.text}")
        except Exception as e:
            self.logger.error(f"Error generating completion: {str(e)}")
            raise
    
    def generate_chat_completion(
        self, 
        messages: List[ChatMessage], 
        max_tokens: int = 200, 
        temperature: float = 0.7,
        use_knowledge_base: bool = True
    ) -> Tuple[str, Optional[List[str]]]:
        """
        Generate chat completion with knowledge base context.
        
        Returns:
            Tuple of (generated_text, source_documents)
        """
        # Extract the last user message
        last_user_message = next((msg.content for msg in reversed(messages) 
                       if msg.role == "user"), "")
        
        self.logger.info(f"Chat completion requested with use_knowledge_base={use_knowledge_base}")
        self.logger.info(f"Last user message: '{last_user_message[:50]}...'")
        
        # Check knowledge base status
        kb_status = self.kb.get_status()
        self.logger.info(f"Knowledge base status: {kb_status['document_count']} documents, {kb_status['vector_count']} vectors")
        
        context = ""
        sources = []
        contexts = []
        
        # Only attempt to retrieve from knowledge base if it's enabled AND there are documents
        if use_knowledge_base and last_user_message and kb_status['vector_count'] > 0:
            self.logger.info("Knowledge base is enabled and has content, retrieving context...")
            
            # Try to get context from knowledge base
            contexts, sources = self.kb.query(last_user_message)
            
            if contexts and len(contexts) > 0:
                self.logger.info(f"Successfully retrieved {len(contexts)} contexts")
                context = "\n\n".join(contexts)
                for i, ctx in enumerate(contexts):
                    self.logger.info(f"Context {i+1} (first 100 chars): {ctx[:100]}...")
            else:
                self.logger.warning("Knowledge base query returned no contexts")
        else:
            if not use_knowledge_base:
                self.logger.info("Knowledge base usage disabled by request parameter")
            elif not last_user_message:
                self.logger.warning("No user message to query knowledge base with")
            elif kb_status['vector_count'] <= 0:
                self.logger.warning("Knowledge base is empty (no vectors)")
            else:
                self.logger.warning("Unknown reason for not using knowledge base")
        
        # If we successfully retrieved context, use the completion approach with knowledge
        if context:
            self.logger.info("Using context-enhanced completion")
            # Create a direct, forceful prompt
            prompt = f"""You are answering a question about company policies and information.
            
IMPORTANT: You MUST ONLY use the information provided below to answer the question.
If the information doesn't contain the answer, say "I don't have that specific information in my knowledge base."

COMPANY INFORMATION:
{context}

QUESTION: {last_user_message}

YOUR ANSWER (using ONLY the provided company information):"""
            
            # Use completions API for a more direct approach
            url = f"{self.base_url}/completions"
            
            payload = {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0.3,  # Lower temperature for more deterministic responses
                "stream": False
            }
            
            try:
                self.logger.info("Calling completions API with knowledge context")
                response = requests.post(url, json=payload)
                
                if response.status_code == 200:
                    answer_text = response.json()["choices"][0]["text"]
                    self.logger.info(f"Generated answer with knowledge context (first 100 chars): {answer_text[:100]}...")
                    return answer_text, sources
                else:
                    self.logger.error(f"Error from completions API: {response.text}")
                    # Fall through to regular chat if this fails
            except Exception as e:
                self.logger.error(f"Error with completion approach: {str(e)}")
                # Fall through to regular chat if this fails
        
        # Standard chat approach as fallback
        self.logger.info("Using standard chat approach (no knowledge context applied)")
        api_messages = [{"role": msg.role, "content": msg.content} for msg in messages]
        url = f"{self.base_url}/chat/completions"
        
        payload = {
            "messages": api_messages,
            "max_tokens": max_tokens, 
            "temperature": temperature,
            "stream": False
        }
        
        try:
            response = requests.post(url, json=payload)
            
            if response.status_code == 200:
                answer = response.json()["choices"][0]["message"]["content"]
                return answer, []
            else:
                self.logger.error(f"Error from chat API: {response.text}")
                raise Exception(f"Error calling LM Studio API: {response.text}")
        except Exception as e:
            self.logger.error(f"Error generating chat completion: {str(e)}")
            raise