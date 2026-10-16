import json

from src.llm.clients.base import ModelResponse
from src.llm.clients.ollama_client import OllamaClient


def test_ollama_client_normalizes_chat_response():
    def fake_request(payload):
        assert payload["model"] == "gemma4:latest"
        assert payload["stream"] is False
        return {
            "model": "gemma4:latest",
            "created_at": "2026-09-14T00:00:00Z",
            "message": {
                "role": "assistant",
                "content": json.dumps({"action": "inspect", "target_id": "desk_01", "reason": "Inspect the desk"}),
            },
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 42,
            "eval_count": 9,
        }

    client = OllamaClient(
        model_name="gemma4:latest",
        base_url="http://localhost:11434",
        timeout=30.0,
        request_fn=fake_request,
    )

    response = client.generate_action("system", "user", {"type": "object"})

    assert isinstance(response, ModelResponse)
    assert response.provider == "ollama"
    assert response.model == "gemma4:latest"
    assert response.payload["action"] == "inspect"
    assert response.payload["target_id"] == "desk_01"
    assert response.input_tokens == 42
    assert response.output_tokens == 9
    assert response.finish_reason == "stop"


def test_ollama_client_handles_markdown_json_wrapping():
    def fake_request(payload):
        return {
            "message": {"role": "assistant", "content": "```json\n{\"action\": \"wait\"}\n```"},
            "done": True,
            "done_reason": "stop",
        }

    client = OllamaClient(
        model_name="qwen:latest",
        base_url="http://localhost:11434",
        timeout=30.0,
        request_fn=fake_request,
    )

    response = client.generate_action("system", "user", {"type": "object"})

    assert response.payload["action"] == "wait"
    assert response.provider == "ollama"


def test_ollama_client_malformed_json_yields_empty_payload_not_fabricated_wait():
    """M5B.1: a real failure mode from the M5B smoke test - Qwen returning
    truncated/invalid JSON must never silently become {"action": "wait"}
    at the client layer. The strict validator in llm_controller.py is what
    decides how to classify and safely resolve an empty payload."""

    def fake_request(payload):
        return {
            "message": {"role": "assistant", "content": '{"door": "door_1801, unterminated'},
            "done": True,
            "done_reason": "stop",
        }

    client = OllamaClient(
        model_name="qwen:latest",
        base_url="http://localhost:11434",
        timeout=30.0,
        request_fn=fake_request,
    )

    response = client.generate_action("system", "user", {"type": "object"})

    assert response.payload == {}
    assert "action" not in response.payload
