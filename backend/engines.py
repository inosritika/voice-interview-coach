"""Engine factory. Pipeline code (main.py) calls these — it never imports a
concrete engine or checks a flag itself. Flip the flags in config.py / .env and
the whole pipeline swaps underneath. Engines are built once and cached.
"""

import config
from llm.base import LLMEngine
from stt.base import STTEngine
from tts.base import TTSEngine
from vad.base import VADEngine

_stt: STTEngine | None = None
_tts: TTSEngine | None = None
_llm: LLMEngine | None = None
_vad: VADEngine | None = None
_pipeline = None


def get_stt() -> STTEngine:
    global _stt
    if _stt is None:
        if config.STT_ENGINE == "local":
            from stt.local_whisper import LocalWhisperSTT

            _stt = LocalWhisperSTT()
        elif config.STT_ENGINE == "deepgram":
            from stt.deepgram import DeepgramSTT

            _stt = DeepgramSTT()
        else:
            raise ValueError(f"Unknown STT_ENGINE: {config.STT_ENGINE!r}")
    return _stt


def get_tts() -> TTSEngine:
    global _tts
    if _tts is None:
        if config.TTS_ENGINE == "local":
            from tts.local_piper import LocalPiperTTS

            _tts = LocalPiperTTS()
        elif config.TTS_ENGINE == "cartesia":
            from tts.cartesia import CartesiaTTS

            _tts = CartesiaTTS()
        else:
            raise ValueError(f"Unknown TTS_ENGINE: {config.TTS_ENGINE!r}")
    return _tts


def get_pipeline():
    """The turn strategy — cascaded (STT+LLM) or fused (Gemma). This is the
    top-level 'which pipeline' swap; the strategies below still use the per-stage
    engines above where relevant."""
    global _pipeline
    if _pipeline is None:
        if config.PIPELINE == "cascaded":
            from pipeline.cascaded import CascadedStrategy

            _pipeline = CascadedStrategy()
        elif config.PIPELINE == "fused":
            from pipeline.fused import FusedStrategy

            _pipeline = FusedStrategy()
        else:
            raise ValueError(f"Unknown PIPELINE: {config.PIPELINE!r}")
    return _pipeline


def get_vad() -> VADEngine:
    global _vad
    if _vad is None:
        # Only Silero for now; kept behind the factory so a future swap is a
        # one-line change here, exactly like the other stages.
        from vad.silero import SileroVAD

        _vad = SileroVAD()
    return _vad


_turn_checker = None
_turn_checker_failed = False


def get_turn_checker():
    """The semantic turn-completeness model (turndetect/), or None when
    ENDPOINT_MODE=silence or the model can't load — the caller falls back to
    plain silence endpointing, so a missing 8MB file never breaks the app."""
    global _turn_checker, _turn_checker_failed
    if config.ENDPOINT_MODE != "semantic" or _turn_checker_failed:
        return _turn_checker
    if _turn_checker is None:
        try:
            from turndetect.smart_turn import SmartTurnChecker

            _turn_checker = SmartTurnChecker()
        except Exception as exc:  # noqa: BLE001 — degrade to silence mode
            _turn_checker_failed = True
            import logging

            logging.getLogger("interview-coach").warning(
                "semantic endpointing unavailable (%s) — using silence mode", exc
            )
    return _turn_checker


def get_llm() -> LLMEngine:
    global _llm
    if _llm is None:
        if config.LLM_ENGINE == "local":
            from llm.local_ollama import LocalOllamaLLM

            _llm = LocalOllamaLLM()
        elif config.LLM_ENGINE == "openai":
            from llm.openai_api import OpenAILLM

            _llm = OpenAILLM()
        else:
            raise ValueError(f"Unknown LLM_ENGINE: {config.LLM_ENGINE!r}")
    return _llm
