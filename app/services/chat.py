import requests
from app.config import get_settings
from app.models import Message, ChatResponse
from typing import List

settings = get_settings()

async def get_chat_response(messages: List[Message]) -> ChatResponse:
    """
    Get a response from the Ollama chat model.
    """
    # Convert messages to the format expected by Ollama
    formatted_messages = [{"role": msg['role'], "content": msg['content']} for msg in messages]
    
    # Prepare the request to Ollama
    response = requests.post(
        f"{settings.ollama_base_url}/api/chat",
        json={
            "model": settings.model_name,
            "messages": formatted_messages,
            "stream": False,
            "options": {
                "temperature": settings.temperature,
                "num_predict": settings.max_tokens
            }
        }
    )
    
    if response.status_code != 200:
        raise Exception(f"Ollama API error: {response.text}")

    response_data = response.json()
    
    message = Message(
        role="assistant",
        content=response_data["message"]["content"]
    )

    return ChatResponse(
        message=message,
        usage={
            "prompt_tokens": response_data.get("prompt_eval_count", 0),
            "completion_tokens": response_data.get("eval_count", 0),
            "total_tokens": response_data.get("prompt_eval_count", 0) + response_data.get("eval_count", 0)
        }
    ) 