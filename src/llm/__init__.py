"""LLM integration package for Spy vs AI."""

from src.llm.clients.base import ModelCapabilities, ModelResponse
from src.llm.clients.mock_client import MockModelClient
from src.llm.clients.ollama_client import OllamaClient
from src.llm.clients.openai_compatible_client import OpenAICompatibleClient

__all__ = [
    "ModelCapabilities",
    "ModelResponse",
    "MockModelClient",
    "OllamaClient",
    "OpenAICompatibleClient",
]
