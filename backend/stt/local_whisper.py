import asyncio
import threading

import numpy as np
from faster_whisper import WhisperModel

from config import (
    WHISPER_BEAM_SIZE,
    WHISPER_COMPUTE_TYPE,
    WHISPER_DEVICE,
    WHISPER_INITIAL_PROMPT,
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
        # A live partial and the final transcription can be requested at nearly the
        # same instant (a partial fires just before the utterance ends). WhisperModel
        # isn't safe to call from two threads at once, so serialize model access.
        self._lock = threading.Lock()

    def _get_model(self) -> WhisperModel:
        if self._model is None:
            self._model = WhisperModel(
                WHISPER_MODEL,
                device=WHISPER_DEVICE,
                compute_type=WHISPER_COMPUTE_TYPE,
            )
        return self._model

    async def transcribe(
        self, audio_bytes: bytes, initial_prompt: str | None = None, fast: bool = False
    ) -> str:
        # faster-whisper is blocking CPU work — keep it off the event loop.
        return await asyncio.to_thread(self._transcribe_sync, audio_bytes, initial_prompt, fast)

    def _transcribe_sync(
        self, audio_bytes: bytes, initial_prompt: str | None = None, fast: bool = False
    ) -> str:
        # int16 PCM -> float32 in [-1, 1], which is what whisper expects.
        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        # Decoding knobs — accuracy first (the interviewer is only as good as the
        # transcript it's fed; garbled input made even a strong LLM incoherent):
        #   beam_size          higher = more accurate search (config, now 5).
        #   language           pinned to English, skips a detect-language pass.
        #   condition_on_previous_text=False   each utterance is independent —
        #                      stops cross-utterance hallucination drift.
        #   temperature=0      no sampling fallback ladder -> deterministic, fewer
        #                      invented words on hard/quiet audio.
        # Hallucination guards — Whisper notoriously invents fluent text ("thank
        # you", "cheese provided at the door") from the trailing endpoint SILENCE
        # and near-silent frames. These suppress it:
        #   vad_filter=True    drop non-speech spans before decoding, so silence
        #                      never gets "transcribed" into a phantom sentence.
        #   no_speech_threshold / log_prob_threshold   discard segments the model
        #                      itself is unsure about instead of emitting garbage.
        #   initial_prompt     a short domain hint so proper nouns land ("Postgres",
        #                      not "posters") — optional, via WHISPER_INITIAL_PROMPT.
        # Partials (fast=True) run mid-utterance and just need to be quick and
        # roughly right: greedy decode, and skip the VAD filter (the audio is live
        # speech, not a clip with trailing silence). The final pass stays accurate.
        with self._lock:  # one transcription at a time (partial vs final overlap)
            segments, _info = self._get_model().transcribe(
                audio,
                beam_size=1 if fast else WHISPER_BEAM_SIZE,
                language=WHISPER_LANGUAGE or None,
                condition_on_previous_text=False,
                temperature=0.0,
                vad_filter=not fast,
                vad_parameters={"min_silence_duration_ms": 500},
                no_speech_threshold=0.6,
                log_prob_threshold=-1.0,
                # Per-topic jargon (packs.py stt_hint) when provided, else the generic
                # config hint — biases decoding toward the terms actually in play.
                initial_prompt=initial_prompt or WHISPER_INITIAL_PROMPT or None,
            )
            # `segments` is lazy — iterate it INSIDE the lock so the actual decode
            # (which happens on iteration) is what's serialized.
            return " ".join(seg.text.strip() for seg in segments).strip()
