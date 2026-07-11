"""smart-turn v3 (pipecat-ai) — an open, tiny (~8 MB) ONNX turn-detection model.

Verified before adoption (the project's rule 5): repo `pipecat-ai/smart-turn-v3`
on Hugging Face, BSD-2-Clause, not gated; measured ~40 ms per check on this
machine's CPU — fast enough to run inside a live pause.

How it works: the last (up to) 8 seconds of the utterance are converted to a
Whisper log-mel spectrogram (the same front-end whisper STT uses) and fed to a
small classifier that outputs P(turn complete). It listens to *prosody* —
trailing intonation, phrase-final lengthening — not words.

Download (one-time, ~8 MB):
    curl -L -o models/smart-turn-v3.2-cpu.onnx \
      https://huggingface.co/pipecat-ai/smart-turn-v3/resolve/main/smart-turn-v3.2-cpu.onnx
"""

import asyncio
import logging
import pathlib

import numpy as np

import config
from turndetect.base import TurnChecker

log = logging.getLogger("interview-coach")

_MAX_SECS = 8  # the model's context: the last 8s of audio decide the verdict


class SmartTurnChecker(TurnChecker):
    def __init__(self) -> None:
        path = pathlib.Path(__file__).resolve().parent.parent / config.SMART_TURN_MODEL
        if not path.exists():
            raise FileNotFoundError(
                f"smart-turn model not found at {path} — download it with:\n"
                "  curl -L -o models/smart-turn-v3.2-cpu.onnx "
                "https://huggingface.co/pipecat-ai/smart-turn-v3/resolve/main/smart-turn-v3.2-cpu.onnx"
            )
        import onnxruntime as ort  # already a dependency (silero VAD is ONNX too)
        from transformers import WhisperFeatureExtractor

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(str(path), sess_options=so)
        self._input_name = self._session.get_inputs()[0].name
        self._extract = WhisperFeatureExtractor(chunk_length=_MAX_SECS)
        log.info("smart-turn checker ready (%s)", path.name)

    async def completeness(self, pcm: bytes) -> float:
        # Inference is blocking CPU work — same rule as whisper/piper: off the
        # event loop, or the receive loop goes deaf while we think (§3 lesson).
        return await asyncio.to_thread(self._predict, pcm)

    def _predict(self, pcm: bytes) -> float:
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        audio = audio[-_MAX_SECS * config.SAMPLE_RATE :]
        feats = self._extract(
            audio,
            sampling_rate=config.SAMPLE_RATE,
            return_tensors="np",
            padding="max_length",
            max_length=_MAX_SECS * config.SAMPLE_RATE,
            truncation=True,
            do_normalize=True,
        ).input_features.astype(np.float32)
        out = self._session.run(None, {self._input_name: feats})
        return float(out[0].flatten()[0])
