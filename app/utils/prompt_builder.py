# app/utils/prompt_builder.py

from app.utils.personas import get_persona

def build_knowledge_prompt(context, user_question, persona_key="default"):
    """
    Build a prompt with knowledge context and specific persona.
    Optimized for token efficiency.
    """
    persona = get_persona(persona_key)
    
    # Shorten the traits to save tokens
    traits_str = "\n".join([f"- {trait}" for trait in persona["traits"][:3]])  # Only use first 3 traits
    
    # Limit context size - estimate context tokens and truncate if needed
    # Rough estimate: 1 token ≈ 4 characters
    max_context_chars = 2000  # Approximately 500 tokens
    
    if len(context) > max_context_chars:
        context = context[:max_context_chars] + "...[truncated for length]"
    
    # Simplified prompt with no examples to save tokens
    prompt = f"""You are {persona['name']}, a {persona['style']}. 
    
PERSONALITY: {traits_str}

INSTRUCTIONS: Answer based ONLY on the company information below. If info isn't provided, say you don't have that detail.

COMPANY INFORMATION:
{context}

QUESTION: {user_question}

YOUR RESPONSE:"""

    return prompt

def build_regular_chat_prompt(messages, persona_key="default"):
    """
    Build a system message to prepend to the chat history.
    Optimized for token efficiency.
    """
    persona = get_persona(persona_key)
    
    # Simplified system message
    system_message = {
        "role": "system",
        "content": f"You are {persona['name']}, a {persona['style']}. Be conversational and helpful."
    }
    
    # Insert system message at the beginning
    enhanced_messages = [system_message] + messages
    
    return enhanced_messages