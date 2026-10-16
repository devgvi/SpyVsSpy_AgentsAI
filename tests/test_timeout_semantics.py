"""M5E.1.1 spec sections 18-21, 47: controlled, network-free (localhost
only, no external network) proof of the Native-vs-Strands timeout
asymmetry M5E.1 first observed empirically. A tiny fake Ollama-shaped
HTTP server (this file only) answers both request shapes real callers
make:

- non-streaming (`stream: false`, Native's `OllamaClient`): sleeps the
  full configured delay, then returns one JSON body - simulates a slow
  model turn.
- streaming (`stream: true`, Strands' real `ollama.AsyncClient` path):
  sends several small newline-delimited JSON chunks, each gap shorter
  than the configured timeout, but whose *total* duration exceeds it -
  simulates real token-by-token generation that never goes idle long
  enough to trip a per-read timeout.

This lets both wrappers' *actual* timeout enforcement be measured
directly, rather than asserted from reading library source alone.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from src.llm.clients.ollama_client import OllamaClient


def _make_slow_server(*, stream_mode: bool, total_delay: float, chunk_gap: float = 0.15):
    """A minimal local server answering exactly one request the way a
    slow Ollama `/api/chat` would, either as one delayed non-streaming
    body or as several small streamed chunks spread across `total_delay`
    seconds with `chunk_gap`-second gaps between them."""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # keep test output quiet
            pass

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            self.rfile.read(length)  # drain the request body
            if stream_mode:
                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson")
                self.end_headers()
                elapsed = 0.0
                while elapsed < total_delay:
                    time.sleep(chunk_gap)
                    elapsed += chunk_gap
                    chunk = {
                        "model": "gemma4:latest",
                        "message": {"role": "assistant", "content": "."},
                        "done": False,
                    }
                    self.wfile.write((json.dumps(chunk) + "\n").encode("utf-8"))
                    self.wfile.flush()
                final = {
                    "model": "gemma4:latest",
                    "message": {"role": "assistant", "content": ""},
                    "done": True,
                    "done_reason": "stop",
                    "total_duration": int(total_delay * 1e9),
                    "prompt_eval_count": 10,
                    "eval_count": 5,
                }
                self.wfile.write((json.dumps(final) + "\n").encode("utf-8"))
                self.wfile.flush()
            else:
                time.sleep(total_delay)
                body = json.dumps(
                    {
                        "model": "gemma4:latest",
                        "message": {"role": "assistant", "content": '{"action": "wait"}'},
                        "done": True,
                        "done_reason": "stop",
                        "total_duration": int(total_delay * 1e9),
                        "prompt_eval_count": 10,
                        "eval_count": 5,
                    }
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.handle_request, daemon=True)
    thread.start()
    return server, thread


# -- Native: a slow *non-streaming* response is bounded by `timeout` -----


def test_native_ollama_client_hard_bounds_total_latency_to_configured_timeout():
    server, thread = _make_slow_server(stream_mode=False, total_delay=3.0)
    base_url = f"http://127.0.0.1:{server.server_port}"
    client = OllamaClient(model_name="gemma4:latest", base_url=base_url, timeout=1.0)

    start = time.monotonic()
    with pytest.raises(Exception) as excinfo:
        client.generate_action("sys", "user", {})
    elapsed = time.monotonic() - start

    # Bounded near the configured 1.0s ceiling, not anywhere near the
    # server's real 3.0s delay - this is Native's documented hard-timeout
    # behavior (spec section 18), now proven rather than assumed.
    assert elapsed < 2.0
    assert "time" in str(excinfo.value).lower() or "timed out" in str(excinfo.value).lower()
    thread.join(timeout=5.0)


# -- Strands: a slow *streaming* response is NOT bounded the same way ----


def test_strands_streaming_response_is_not_bounded_by_configured_timeout_like_native():
    strands = pytest.importorskip("strands")
    from strands.models.ollama import OllamaModel

    server, thread = _make_slow_server(stream_mode=True, total_delay=2.0, chunk_gap=0.3)
    base_url = f"http://127.0.0.1:{server.server_port}"

    # Same `timeout` value/shape Strands' adapter actually passes
    # (`ollama_client_args={"timeout": X}` -> straight into
    # `ollama.AsyncClient(host, timeout=X)` -> httpx's per-I/O-operation
    # timeout, not a total-request-duration bound - see CHANGELOG.md's
    # M5E.1.1 entry for the full source-level explanation).
    model = OllamaModel(host=base_url, model_id="gemma4:latest", temperature=0.0, ollama_client_args={"timeout": 1.0})

    async def _consume():
        chunks = []
        async for event in model.stream(messages=[{"role": "user", "content": [{"text": "hi"}]}]):
            chunks.append(event)
        return chunks

    import asyncio

    start = time.monotonic()
    asyncio.run(_consume())
    elapsed = time.monotonic() - start
    thread.join(timeout=5.0)

    # The server took the full ~2.0s (in gap-sized chunks, each well under
    # the 1.0s configured timeout) - if Strands' timeout bounded *total*
    # call duration the way Native's does, this would have raised well
    # before 2s. It doesn't: the call completes at roughly the server's
    # real total delay, proving the asymmetry empirically, not just from
    # reading httpx's source.
    assert elapsed >= 1.5


# -- post-fix reproduction: StrandsAgentAdapter's own watchdog (strands-adapter-v2) --


def test_strands_agent_adapter_watchdog_now_bounds_a_slow_call_to_configured_timeout():
    """The asymmetry above is real at the raw OllamaModel/httpx level, but
    `StrandsAgentAdapter._invoke_strands` (M5E.1.1, strands-adapter-v2) no
    longer inherits it: it stops *waiting* on a slow call once
    `llm_timeout` elapses, regardless of why the underlying call is slow.
    Uses the same delayed-fake-agent fixture test_strands_agent.py's own
    fairness tests already rely on - a synchronous `time.sleep` inside the
    fake agent, which is exactly what a real slow/streaming Strands call
    looks like from this adapter's perspective (it blocks the calling
    thread until done, streaming or not)."""
    import sys

    sys.path.insert(0, "tests")
    from test_strands_agent import QueuedStrandsFactory  # noqa: E402

    from src.agents.strands_agent import StrandsAgentAdapter

    factory = QueuedStrandsFactory([[{"action": "wait"}]], delay=3.0)
    adapter = StrandsAgentAdapter("white_spy", agent_factory=factory, llm_timeout=1.0)

    start = time.monotonic()
    invocation = adapter._invoke_strands({"legal_actions": []}, repair_instruction=None)
    elapsed = time.monotonic() - start

    assert elapsed < 2.0  # bounded near 1.0s, not the fake agent's real 3.0s delay
    # M5E.2: distinct from Native's own "provider_error" (a real,
    # provider-reported timeout) so watchdog_terminations can be counted
    # separately from provider_timeouts (spec section 15) - still
    # classifies into the same provider_timeout *category* because the
    # message carries "timed out" (see classify_provider_call_error).
    assert invocation.error_type == "watchdog_timeout"
    from src.llm_controller import classify_provider_call_error

    assert classify_provider_call_error(invocation.error_type, invocation.error_message) == "provider_timeout"
    assert "timed out" in invocation.error_message
