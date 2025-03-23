import requests
import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from app.config import LM_STUDIO_URL
from app.services.knowledge_service import KnowledgeBase
from app.models import ChatMessage

class LLMService:
    def __init__(self):
        self.base_url = LM_STUDIO_URL
        self.kb = KnowledgeBase()
    
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
            contexts, sources = self.kb.query(prompt)
            if contexts:
                context = "\n\n".join(contexts)
                logging.info(f"Retrieved {len(contexts)} contexts for completion")
        
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
                logging.error(f"Error from LM Studio API: {response.text}")
                raise Exception(f"Error calling LM Studio API: {response.text}")
        except Exception as e:
            logging.error(f"Error generating completion: {str(e)}")
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
        
        context = ""
        sources = []
        
        # Retrieve context from knowledge base if requested
        if use_knowledge_base and last_user_message:
            contexts, sources = self.kb.query(last_user_message)
            if contexts:
                context = "\n\n".join(contexts)
                logging.info(f"Retrieved {len(contexts)} contexts for query: {last_user_message}")
                for i, ctx in enumerate(contexts):
                    logging.info(f"Context {i+1} (first 100 chars): {ctx[:100]}...")
        
        # If we have context, use it with a completion approach
        if context:
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
                logging.info("Using completions API with knowledge context")
                response = requests.post(url, json=payload)
                
                if response.status_code == 200:
                    answer_text = response.json()["choices"][0]["text"]
                    logging.info(f"Generated answer using knowledge (first 100 chars): {answer_text[:100]}...")
                    return answer_text, sources
                else:
                    logging.error(f"Error from completions API: {response.text}")
                    # Fall through to regular chat if this fails
            except Exception as e:
                logging.error(f"Error with completion approach: {str(e)}")
                # Fall through to regular chat if this fails
        
        # Standard chat approach as fallback
        logging.info("Using standard chat approach (no knowledge context)")
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
                logging.error(f"Error from chat API: {response.text}")
                raise Exception(f"Error calling LM Studio API: {response.text}")
        except Exception as e:
            logging.error(f"Error generating chat completion: {str(e)}")
            raise