import requests
import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from app.config import LM_STUDIO_URL
from app.services.knowledge_service import get_knowledge_base
from app.models import ChatMessage
from app.utils.prompt_builder import build_knowledge_prompt, build_regular_chat_prompt
from app.utils.personas import get_persona  # Updated import path

class LLMService:
    def __init__(self):
        """Initialize LLM Service (knowledge bases resolved per tenant)."""
        self.base_url = LM_STUDIO_URL
        self.logger = logging.getLogger(__name__)
        self.logger.info("LLMService initialized (tenant-aware mode)")
        self.offline = os.getenv("OFFLINE_MODE", "0") == "1"
        if self.offline:
            self.logger.warning("OFFLINE_MODE enabled: external LM Studio calls will be stubbed.")

    def _kb(self, tenant_id: Optional[str]):
        return get_knowledge_base(tenant_id)
    
    def generate_completion(
        self, 
        prompt: str, 
        max_tokens: int = 200, 
        temperature: float = 0.7,
    use_knowledge_base: bool = True,
        persona: str = "default",
        tenant_id: Optional[str] = None
    ) -> Tuple[str, Optional[List[str]]]:
        """
        Generate text completion with optional knowledge base context.
        
        Returns:
            Tuple of (generated_text, source_documents)
        """
        context = ""
        sources = []
        
        # Get persona configuration
        persona_config = get_persona(persona)
        persona_temp = persona_config.get("temperature", temperature)
        
        # Retrieve context from knowledge base if requested
        if use_knowledge_base:
            self.logger.info(f"Knowledge base enabled for completion, querying with: '{prompt[:50]}...'")
            contexts, sources = self._kb(tenant_id).query(prompt)
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
            # Use the prompt builder with persona and few-shot examples
            enhanced_prompt = build_knowledge_prompt(context, prompt, persona)
            self.logger.info(f"Using persona '{persona}' for knowledge-enhanced prompt")
        
        if self.offline:
            stub = f"[OFFLINE RESPONSE] Persona={persona}. Prompt='{prompt[:60]}...'" + (" With context." if context else "")
            return stub, sources

        # Call LM Studio API
        url = f"{self.base_url}/completions"
        
        payload = {
            "prompt": enhanced_prompt,
            "max_tokens": max_tokens,
            "temperature": persona_temp,
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
    use_knowledge_base: bool = True,
    persona: str = "default",
    tenant_id: Optional[str] = None
    ) -> Tuple[str, Optional[List[str]]]:
        """Generate chat completion, optionally enriched with knowledge base context."""
        # 1. Extract last user message
        last_user_message = next((m.content for m in reversed(messages) if m.role == "user"), "")
        self.logger.info(
            f"Chat completion requested (use_kb={use_knowledge_base}, persona={persona}) last_user='{last_user_message[:50]}...'"
        )

        # 2. Persona configuration
        persona_config = get_persona(persona)
        persona_temp = persona_config.get("temperature", temperature)

        # 3. Knowledge base status & optional retrieval
        kb_status = self._kb(tenant_id).get_status()
        self.logger.info(
            f"KB status: docs={kb_status['document_count']} vectors={kb_status['vector_count']}"
        )
        context = ""
        sources: List[str] = []

        if (
            use_knowledge_base
            and last_user_message
            and kb_status.get("vector_count", 0) > 0
        ):
            contexts, sources = self._kb(tenant_id).query(last_user_message)
            if contexts:
                context = "\n\n".join(contexts)
                self.logger.info(f"Retrieved {len(contexts)} context chunk(s)")
            else:
                self.logger.warning("KB query returned no contexts")
        else:
            if not use_knowledge_base:
                self.logger.info("KB usage disabled by request")
            elif not last_user_message:
                self.logger.warning("No user message to query KB with")
            elif kb_status.get("vector_count", 0) <= 0:
                self.logger.warning("KB empty (no vectors)")

        # 4. If we have context, attempt knowledge-enhanced completion path
        if context:
            prompt = build_knowledge_prompt(context, last_user_message, persona)
            url = f"{self.base_url}/completions"
            payload = {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": persona_temp,
                "stream": False,
            }
            if self.offline:
                return f"[OFFLINE CHAT COMPLETION WITH CONTEXT] {last_user_message[:80]}...", sources
            try:
                resp = requests.post(url, json=payload)
                if resp.status_code == 200:
                    answer = resp.json()["choices"][0]["text"]
                    return answer, sources
                else:
                    self.logger.error(
                        f"Completions API (with context) error {resp.status_code}: {resp.text}"
                    )
            except Exception as e:
                self.logger.error(f"Knowledge-enhanced completion failed: {e}")
                # Fall through to chat approach

        # 5. Fallback to standard chat
        api_messages = [{"role": m.role, "content": m.content} for m in messages]
        enhanced_messages = build_regular_chat_prompt(api_messages, persona)
        url = f"{self.base_url}/chat/completions"
        payload = {
            "messages": enhanced_messages,
            "max_tokens": max_tokens,
            "temperature": persona_temp,
            "stream": False,
        }
        if self.offline:
            return f"[OFFLINE CHAT RESPONSE] {last_user_message[:80]}...", []
        try:
            resp = requests.post(url, json=payload)
            if resp.status_code == 200:
                answer = resp.json()["choices"][0]["message"]["content"]
                return answer, []  # sources empty because chat path
            else:
                raise Exception(f"Chat API error {resp.status_code}: {resp.text}")
        except Exception as e:
            self.logger.error(f"Error generating chat completion: {e}")
            raise