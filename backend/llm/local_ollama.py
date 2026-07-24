import json
from collections.abc import AsyncIterator

import httpx

from config import OLLAMA_HOST, OLLAMA_MODEL, OLLAMA_NUM_PREDICT, OLLAMA_TEMPERATURE
from llm.base import LLMEngine

# Per-request generation options. num_predict is a hard cap on output tokens, so
# the interviewer can't ramble — which keeps turns short AND bounds latency.
_OPTIONS = {"num_predict": OLLAMA_NUM_PREDICT, "temperature": OLLAMA_TEMPERATURE}


class LocalOllamaLLM(LLMEngine):
    """Local LLM via Ollama (https://ollama.com). No API key, runs on your box.

    Start it separately: `ollama serve`, then `ollama pull <model>`. We talk to
    its HTTP API. It's already streaming-native, so `stream` is real here.
    """

    async def reply(
        self,
        messages: list[dict],
        max_tokens: int | None = None,
        format_schema: dict | None = None,
        reasoning_effort: str | None = None,
    ) -> str:
        # Kept for interface parity with hosted reasoning models; Ollama's
        # current adapter has no equivalent per-call reasoning knob.
        del reasoning_effort
        options = _OPTIONS if max_tokens is None else {**_OPTIONS, "num_predict": max_tokens}
        payload = {"model": OLLAMA_MODEL, "messages": messages,
                   "stream": False, "options": options}
        if format_schema is not None:
            # Ollama's structured outputs: the schema is compiled into a grammar
            # that constrains decoding, so the reply is guaranteed-valid JSON.
            payload["format"] = format_schema
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(f"{OLLAMA_HOST}/api/chat", json=payload)
            resp.raise_for_status()
            return resp.json()["message"]["content"].strip()

    async def stream(self, messages: list[dict]) -> AsyncIterator[str]:
        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream(
                "POST",
                f"{OLLAMA_HOST}/api/chat",
                json={"model": OLLAMA_MODEL, "messages": messages,
                      "stream": True, "options": _OPTIONS},
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    chunk = json.loads(line)
                    piece = chunk.get("message", {}).get("content", "")
                    if piece:
                        yield piece
