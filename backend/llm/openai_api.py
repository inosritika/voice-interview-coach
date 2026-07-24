from collections.abc import AsyncIterator

import httpx

from config import OPENAI_API_KEY, OPENAI_MODEL, OPENAI_REASONING_EFFORT
from llm.base import LLMEngine

OPENAI_URL = "https://api.openai.com/v1/responses"


class OpenAILLM(LLMEngine):
    """Hosted interviewer through the OpenAI Responses API.

    The rest of the app still talks to the same LLMEngine interface. This adapter
    converts its chat-history format into Responses input, retains streaming for
    quick text-to-speech, and retains strict JSON output for the director loop.
    """

    def __init__(self) -> None:
        if not OPENAI_API_KEY:
            raise RuntimeError("LLM_ENGINE=openai but OPENAI_API_KEY is not set")

    @staticmethod
    def _input(messages: list[dict]) -> list[dict]:
        """Map the app's old chat roles to the Responses API role names.

        The project's pinned system prompt is application guidance, so it becomes
        a developer message. Candidate and interviewer turns stay user/assistant.
        """
        role_map = {"system": "developer", "user": "user", "assistant": "assistant"}
        return [
            {"role": role_map.get(message["role"], "user"), "content": message["content"]}
            for message in messages
        ]

    @staticmethod
    def _output_text(response: dict) -> str:
        """Collect text from every message item; Responses output can also
        include reasoning and other non-message items before the final answer."""
        texts: list[str] = []
        for item in response.get("output", []):
            if item.get("type") != "message":
                continue
            for content in item.get("content", []):
                if content.get("type") == "output_text":
                    texts.append(content.get("text", ""))
        return "".join(texts).strip()

    def _payload(
        self,
        messages: list[dict],
        max_tokens: int | None = None,
        format_schema: dict | None = None,
        stream: bool = False,
        reasoning_effort: str | None = None,
    ) -> dict:
        payload: dict = {
            "model": OPENAI_MODEL,
            "input": self._input(messages),
            "reasoning": {"effort": reasoning_effort or OPENAI_REASONING_EFFORT},
            # Interview transcripts and resumes are sensitive user data. Do
            # not retain Responses objects for later retrieval by default.
            "store": False,
            "stream": stream,
        }
        if max_tokens is not None:
            payload["max_output_tokens"] = max_tokens
        if format_schema is not None:
            payload["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": "interview_director_action",
                    "strict": True,
                    "schema": format_schema,
                }
            }
        return payload

    async def reply(
        self,
        messages: list[dict],
        max_tokens: int | None = None,
        format_schema: dict | None = None,
        reasoning_effort: str | None = None,
    ) -> str:
        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                OPENAI_URL,
                headers=headers,
                json=self._payload(messages, max_tokens, format_schema, reasoning_effort=reasoning_effort),
            )
            resp.raise_for_status()
            return self._output_text(resp.json())

    async def stream(self, messages: list[dict]) -> AsyncIterator[str]:
        import json

        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream(
                "POST", OPENAI_URL, headers=headers, json=self._payload(messages, stream=True)
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[len("data: ") :]
                    if data == "[DONE]":
                        break
                    event = json.loads(data)
                    if event.get("type") != "response.output_text.delta":
                        continue
                    piece = event.get("delta", "")
                    if piece:
                        yield piece
