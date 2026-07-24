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
    """The PROSODY turn-completeness model (turndetect/smart_turn.py), or None
    when ENDPOINT_MODE=silence or the model can't load — the caller falls back to
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
                "prosody endpointing unavailable (%s) — using silence mode", exc
            )
    return _turn_checker


_semantic_checker = None


def get_semantic_checker():
    """The TEXT semantic turn checker (turndetect/semantic.py), or None unless
    ENDPOINT_MODE=semantic AND ENDPOINT_SEMANTIC_TEXT is on. It reuses the STT +
    LLM engines, so it has nothing of its own to fail loading."""
    global _semantic_checker
    if config.ENDPOINT_MODE != "semantic" or not config.ENDPOINT_SEMANTIC_TEXT:
        return None
    if _semantic_checker is None:
        from turndetect.semantic import SemanticTurnChecker

        _semantic_checker = SemanticTurnChecker()
    return _semantic_checker


def _build_llm(name: str) -> LLMEngine:
    if name == "local":
        from llm.local_ollama import LocalOllamaLLM

        return LocalOllamaLLM()
    if name == "openai":
        from llm.openai_api import OpenAILLM

        return OpenAILLM()
    if name == "claude":
        from llm.claude_code import ClaudeCodeLLM

        return ClaudeCodeLLM()
    raise ValueError(f"Unknown LLM engine: {name!r}")


def get_llm() -> LLMEngine:
    """The PRIMARY interviewer brain — the spoken reply and the debrief. This is
    what the user hears and reads, so it's the quality-first engine (LLM_ENGINE)."""
    global _llm
    if _llm is None:
        _llm = _build_llm(config.LLM_ENGINE)
    return _llm


_utility_llm: LLMEngine | None = None


def get_utility_llm() -> LLMEngine:
    """The UTILITY brain for internal bookkeeping the user never sees: the
    director's JSON decide-loop and context compaction. These fire several times
    per turn, so on a high per-call-overhead engine like Claude (~4s/call, mostly
    CLI-spawn) they dominate latency. UTILITY_LLM_ENGINE lets them run on a fast
    local model (Ollama, ~0.8s, no spawn) while the SPOKEN reply stays on the
    quality engine. "main" reuses get_llm() (everything on one engine)."""
    global _utility_llm
    if config.UTILITY_LLM_ENGINE == "main":
        return get_llm()
    if _utility_llm is None:
        _utility_llm = _build_llm(config.UTILITY_LLM_ENGINE)
    return _utility_llm
