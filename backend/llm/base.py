from abc import ABC, abstractmethod
from collections.abc import AsyncIterator


class LLMEngine(ABC):
    """The interviewer's brain. Takes the running chat history, returns the
    next interviewer turn.

    `reply` returns the whole turn at once — that's all step 1 needs. `stream`
    yields tokens as they arrive; step 3 will feed those into streaming TTS to
    cut latency. Local engines can implement stream as a thin wrapper if needed.
    """

    @abstractmethod
    async def reply(
        self,
        messages: list[dict],
        max_tokens: int | None = None,
        format_schema: dict | None = None,
        reasoning_effort: str | None = None,
    ) -> str:
        """messages is a list of {"role": "system"|"user"|"assistant", "content": str}.
        Returns the assistant's full reply text.

        `max_tokens` overrides the engine's default output cap for this one call.
        The interviewer keeps replies short (a low cap), but the end-of-interview
        debrief needs room for a full scored review — pass a larger cap there.

        `format_schema` is a JSON schema for CONSTRAINED DECODING: the engine
        masks token choices during generation so only schema-valid JSON can come
        out. This is stronger than asking nicely in the prompt — invalid output
        becomes impossible, not just unlikely. Used by the director's action loop."""
        ...

    async def stream(self, messages: list[dict]) -> AsyncIterator[str]:
        """Yield reply text incrementally. Default: fall back to non-streaming
        and emit the whole thing as one chunk, so step 1 works without every
        engine implementing true streaming yet."""
        yield await self.reply(messages)
