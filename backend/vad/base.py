from abc import ABC, abstractmethod

import numpy as np


class VADEngine(ABC):
    """Voice Activity Detection: given one short frame of audio, how likely is
    it that someone is speaking?

    A VAD is a *gatekeeper only* — it knows nothing about words or meaning, just
    sound-vs-speech. Its output (a probability per frame) is the raw signal that
    the Endpointer turns into "the person is done talking". Keeping VAD behind an
    interface means we could swap Silero for another detector without touching
    the endpointing or pipeline code — the same adapter idea as STT/TTS/LLM.
    """

    @abstractmethod
    def speech_prob(self, frame: np.ndarray) -> float:
        """frame is float32 PCM in [-1, 1], exactly VAD_FRAME_SAMPLES long.
        Returns P(speech) in [0, 1]."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Clear any internal state between utterances. Silero is a recurrent
        model — it remembers recent frames — so we reset it at the start of each
        new turn to avoid one turn bleeding into the next."""
        ...
