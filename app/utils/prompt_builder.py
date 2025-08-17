# app/utils/prompt_builder.py

from app.utils.personas import get_persona

# Centralized minimal system safeguard / style instructions
def _build_system_instructions(persona_key: str) -> str:
    """
    Returns concise system instructions (token‑efficient) enforcing:
    - No chain-of-thought exposure
    - Final answer only
    - Handling missing data
    """
    base = (
        "Provide only the final answer in a concise, conversational professional tone. "
        "Do NOT output internal reasoning, chain-of-thought, analysis steps, or tags like <think>. "
        "If required data is absent, explicitly say it's not available and suggest a validated next step."
    )
    if persona_key == "safety_officer":
        # Slightly specialized safety angle, still short
        return (
            base +
            " Prioritize accuracy, regulatory clarity, and safe practice. "
            "Do not fabricate figures or regulations—state uncertainty plainly."
        )
    return base

def build_knowledge_prompt(context, user_question, persona_key="default"):
    """
    Build a prompt with knowledge context and specific persona.
    Optimized for token efficiency.
    """
    persona = get_persona(persona_key)
    system_instructions = _build_system_instructions(persona_key)

    # Use only first 3 traits to save tokens
    traits_str = " ".join(persona["traits"][:3])

    # Truncate context (≈500 tokens)
    max_context_chars = 2000
    if len(context) > max_context_chars:
        context = context[:max_context_chars] + "...[truncated]"

    prompt = (
        f"You are {persona['name']}, a {persona['style']}.\n"
        f"PERSONA TRAITS: {traits_str}\n"
        f"SYSTEM INSTRUCTIONS: {system_instructions}\n\n"
        "Answer ONLY using the COMPANY INFORMATION. If the specific answer is not present, say so.\n\n"
        "COMPANY INFORMATION:\n"
        f"{context}\n\n"
        f"QUESTION: {user_question}\n\n"
        "FINAL ANSWER:"
    )
    return prompt

def build_regular_chat_prompt(messages, persona_key="default"):
    """
    Build a system message to prepend to the chat history.
    Optimized for token efficiency.
    """
    persona = get_persona(persona_key)
    system_instructions = _build_system_instructions(persona_key)

    system_message = {
        "role": "system",
        "content": (
            f"You are {persona['name']}, a {persona['style']}. "
            f"{system_instructions}"
        )
    }

    # Insert system message at the beginning
    enhanced_messages = [system_message] + messages
    
    return enhanced_messages