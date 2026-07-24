from abc import ABC, abstractmethod


class STTEngine(ABC):
    """Speech-to-text. Give it a full audio clip, get back a transcript.

    As of step 2, the server hands over one complete utterance's worth of audio
    once the endpointer decides the turn is over. In step 3 we'll add a streaming
    method that yields partial transcripts; this whole-utterance method stays as
    the simple path.
    """

    @abstractmethod
    async def transcribe(
        self, audio_bytes: bytes, initial_prompt: str | None = None, fast: bool = False
    ) -> str:
        """audio_bytes is raw PCM: 16 kHz, mono, 16-bit signed little-endian —
        the utterance the VAD/endpointer captured. Returns the recognized text.

        `initial_prompt` is an optional domain-vocabulary hint (the current topic
        pack's jargon) so technical terms transcribe correctly. Engines that can't
        use it may ignore it.

        `fast` favors speed over accuracy (used for live PARTIAL transcripts while
        the user is still talking) — the accurate pass runs on the final utterance."""
        ...
