from abc import ABC, abstractmethod


class TTSEngine(ABC):
    """Text-to-speech. Give it text, get back a complete audio clip.

    Returns WAV bytes (16-bit PCM) so the browser can play it directly from a
    Blob. Step 3 will add a streaming variant that yields audio chunks as the
    LLM produces tokens; this whole-clip method stays as the simple path.
    """

    @abstractmethod
    async def synthesize(self, text: str) -> bytes:
        """Returns a complete WAV audio clip for `text`."""
        ...
