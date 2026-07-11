from collections.abc import AsyncIterator

import httpx

from config import OPENAI_API_KEY, OPENAI_MODEL
from llm.base import LLMEngine

OPENAI_URL = "https://api.openai.com/v1/chat/completions"


class OpenAILLM(LLMEngine):
    """Hosted LLM via OpenAI's chat completions API. Needs OPENAI_API_KEY.
    Same interface as the local Ollama engine, so it's a flag swap away."""

    def __init__(self) -> None:
        if not OPENAI_API_KEY:
            raise RuntimeError("LLM_ENGINE=openai but OPENAI_API_KEY is not set")

    async def reply(
        self,
        messages: list[dict],
        max_tokens: int | None = None,
        format_schema: dict | None = None,
    ) -> str:
        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
        payload = {"model": OPENAI_MODEL, "messages": messages}
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if format_schema is not None:
            # OpenAI's equivalent of constrained decoding ("structured outputs").
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "action", "strict": True, "schema": format_schema},
            }
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(OPENAI_URL, headers=headers, json=payload)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()

    async def stream(self, messages: list[dict]) -> AsyncIterator[str]:
        import json

        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
        payload = {"model": OPENAI_MODEL, "messages": messages, "stream": True}
        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream(
                "POST", OPENAI_URL, headers=headers, json=payload
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[len("data: ") :]
                    if data == "[DONE]":
                        break
                    delta = json.loads(data)["choices"][0]["delta"]
                    piece = delta.get("content", "")
                    if piece:
                        yield piece
