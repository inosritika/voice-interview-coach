import asyncio

import numpy as np
from faster_whisper import WhisperModel

from config import (
    WHISPER_BEAM_SIZE,
    WHISPER_COMPUTE_TYPE,
    WHISPER_DEVICE,
    WHISPER_LANGUAGE,
    WHISPER_MODEL,
)
from stt.base import STTEngine


class LocalWhisperSTT(STTEngine):
    """faster-whisper running locally. No API key, no network.

    As of step 2 the audio arrives as raw PCM (16 kHz, mono, 16-bit signed) that
    the server already assembled from the VAD-detected utterance — so there's no
    container to decode anymore. We just reinterpret the bytes as samples and
    hand whisper a float32 array directly (faster, no temp file, no ffmpeg).
    """

    def __init__(self) -> None:
        self._model: WhisperModel | None = None

    def _get_model(self) -> WhisperModel:
        if self._model is None:
            self._model = WhisperModel(
                WHISPER_MODEL,
                device=WHISPER_DEVICE,
                compute_type=WHISPER_COMPUTE_TYPE,
            )
        return self._model

    async def transcribe(self, audio_bytes: bytes) -> str:
        # faster-whisper is blocking CPU work — keep it off the event loop.
        return await asyncio.to_thread(self._transcribe_sync, audio_bytes)

    def _transcribe_sync(self, audio_bytes: bytes) -> str:
        # int16 PCM -> float32 in [-1, 1], which is what whisper expects.
        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        # Latency knobs — this is on the critical path to first audio:
        #   beam_size=1  greedy decode, ~2-3x faster than the beam_size=5 default,
        #                negligible accuracy loss on clean single-utterance speech.
        #   language     pinned (interviews are in English), so whisper skips a
        #                whole detect-language forward pass on every turn.
        #   condition_on_previous_text=False  each utterance is independent here;
        #                turning it off avoids cross-utterance hallucination drift.
        segments, _info = self._get_model().transcribe(
            audio,
            beam_size=WHISPER_BEAM_SIZE,
            language=WHISPER_LANGUAGE or None,
            condition_on_previous_text=False,
        )
        return " ".join(seg.text.strip() for seg in segments).strip()
