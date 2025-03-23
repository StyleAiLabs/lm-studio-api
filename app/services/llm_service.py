import requests
import os
from dotenv import load_dotenv

load_dotenv()

LM_STUDIO_URL = os.getenv("LM_STUDIO_URL", "http://localhost:1234/v1")

class LLMService:
    def __init__(self):
        self.base_url = LM_STUDIO_URL
    
    def generate_completion(self, prompt, max_tokens=200, temperature=0.7):
        url = f"{self.base_url}/completions"
        
        payload = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False
        }
        
        response = requests.post(url, json=payload)
        
        if response.status_code == 200:
            return response.json()["choices"][0]["text"]
        else:
            raise Exception(f"Error calling LM Studio API: {response.text}")
    
    def generate_chat_completion(self, messages, max_tokens=200, temperature=0.7):
        url = f"{self.base_url}/chat/completions"
        
        payload = {
            "messages": [{"role": msg.role, "content": msg.content} for msg in messages],
            "max_tokens": max_tokens, 
            "temperature": temperature,
            "stream": False
        }
        
        response = requests.post(url, json=payload)
        
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            raise Exception(f"Error calling LM Studio API: {response.text}")