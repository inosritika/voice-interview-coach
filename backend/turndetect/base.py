"""Turn-completeness checking — the semantic half of endpointing.

Silero VAD answers "is this frame speech?". A turn checker answers a smarter
question about the WHOLE utterance so far: "does this sound like a finished
thought, or someone mid-sentence?" — from prosody: pitch contour, phrase-final
lengthening, trailing intonation. It's the difference between

    "My biggest achievement was leading the…"   (falling? no — keep waiting)
    "…and that's why I left."                   (final — end the turn NOW)

Same adapter discipline as stt/tts/llm/vad: an abstract interface here, a
concrete model behind it, chosen by the factory in engines.py.
"""

from abc import ABC, abstractmethod


class TurnChecker(ABC):
    @abstractmethod
    async def completeness(self, pcm: bytes) -> float:
        """Probability [0..1] that the utterance (raw 16 kHz mono s16le PCM)
        is a COMPLETE turn. Must be fast (tens of ms) — it runs inside a live
        pause while the user might still be mid-thought."""
        ...
