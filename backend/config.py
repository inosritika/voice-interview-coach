"""Central configuration. All engine choices are flags read from the environment.

The whole point of the adapter pattern (see stt/base.py, tts/base.py, llm/base.py)
is that pipeline code never hardcodes an engine. It asks the factory in engines.py
for "the STT engine" and gets whichever one these flags select.
"""

import os

from dotenv import load_dotenv

load_dotenv()

# ---- Pipeline: how a whole turn is processed --------------------------------
# "cascaded": speech -> STT -> LLM -> reply. Two separate models (below), full
#             visibility into each stage. This is the hand-built learning path.
# "fused":    speech + history -> Gemma 4 -> reply, in ONE model call. Gemma does
#             both the speech-understanding and the interviewer reasoning, deleting
#             the STT->LLM handoff. Needs transformers + Gemma 4 weights.
PIPELINE = os.getenv("PIPELINE", "cascaded")     # "cascaded" | "fused"

# ---- Engine selection flags (used by the cascaded pipeline) -----------------
# Defaults are all-local so the MVP runs with no API keys.
STT_ENGINE = os.getenv("STT_ENGINE", "local")   # "local" (faster-whisper) | "deepgram"
TTS_ENGINE = os.getenv("TTS_ENGINE", "local")   # "local" (piper)          | "cartesia"
LLM_ENGINE = os.getenv("LLM_ENGINE", "local")   # "local" (ollama)         | "openai"

# ---- Fused pipeline: Gemma 4 audio model ------------------------------------
# Gemma 4 wants audio as 16 kHz mono float32 in [-1, 1] — exactly what we already
# feed whisper, so no conversion. E2B is the smallest audio-capable size.
# E4B (instruction-tuned) is the sweet spot for a 16+ GB machine: enough
# reasoning for relevant follow-ups, still light enough to stay responsive.
# E2B = lower latency / weaker reasoning; 12B = best quality but slow on MPS.
# Note the capitalization: it's "E4B", not "e4b" (case-sensitive HF repo id).
GEMMA_MODEL = os.getenv("GEMMA_MODEL", "google/gemma-4-E4B-it")
# NOT "auto": measured on this project's own hardware, transformers' device_map=
# "auto" mis-judged available memory and silently offloaded some layers to disk
# (visible as "Some parameters are on the meta device..." in the logs), which
# made every forward pass page weights back in — a large, silent latency tax.
# "auto" here means "let fused.py's own detection pick mps/cuda/cpu", which pins
# every parameter to one real device with no disk offload.
GEMMA_DEVICE = os.getenv("GEMMA_DEVICE", "auto")   # "auto" | "cpu" | "mps" | "cuda"
GEMMA_MAX_NEW_TOKENS = int(os.getenv("GEMMA_MAX_NEW_TOKENS", "160"))

# ---- Local STT: faster-whisper ----------------------------------------------
# Model size trades accuracy for speed/VRAM: tiny < base < small < medium < large-v3.
# "base" is a good CPU-friendly starting point.
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")        # "cpu" | "cuda"
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")  # int8 is fast on CPU
# STT is on the critical path to first audio. beam_size=1 (greedy) is ~2-3x
# faster than the library's default of 5, for negligible accuracy loss on a
# clean single utterance. Pin the language so whisper skips detect-language.
WHISPER_BEAM_SIZE = int(os.getenv("WHISPER_BEAM_SIZE", "1"))
WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "en")   # "" to auto-detect

# ---- Local TTS: piper -------------------------------------------------------
# Path to a downloaded piper voice .onnx (its .onnx.json must sit beside it).
# Download voices from https://github.com/rhasspy/piper/blob/master/VOICES.md
PIPER_VOICE = os.getenv("PIPER_VOICE", "voices/en_US-lessac-medium.onnx")

# ---- Local LLM: Ollama ------------------------------------------------------
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
# Hard ceiling on reply length (tokens). A spoken interviewer turn is short, so
# capping this both keeps answers tight AND bounds worst-case generation time.
OLLAMA_NUM_PREDICT = int(os.getenv("OLLAMA_NUM_PREDICT", "80"))
# Lower temperature = less rambly, more on-task. 0.7 is a reasonable middle.
OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.7"))

# ---- Step 2: streaming audio + VAD + endpointing ----------------------------
# The browser now streams raw PCM continuously instead of one clip on a button.
# Contract on the wire: 16 kHz, mono, 16-bit signed little-endian PCM.
SAMPLE_RATE = 16000
# Silero VAD requires exactly 512 samples per frame at 16 kHz = 32 ms.
VAD_FRAME_SAMPLES = 512
# Speech if the model's probability for a frame is >= this. Higher = stricter
# (fewer false triggers from noise, but may clip very quiet speech).
VAD_THRESHOLD = float(os.getenv("VAD_THRESHOLD", "0.5"))
# Endpointing: how long a pause must last before we decide "they're done".
# This is the patience-vs-snappiness knob — the heart of step 2. Too low and it
# guillotines thinking pauses; too high and every reply feels laggy.
ENDPOINT_SILENCE_MS = int(os.getenv("ENDPOINT_SILENCE_MS", "700"))

# ---- Semantic turn detection (two-stage endpointing) --------------------------
# "silence":  the fixed-pause rule above — simple, predictable, the default.
# "semantic": two-stage — at MIN silence, ask the smart-turn model whether the
#             utterance SOUNDS finished (prosody: trailing intonation, final
#             lengthening). Finished -> end the turn early (snappier than 700ms);
#             unfinished -> extend patience up to MAX (mid-thought pauses stop
#             getting guillotined). Requires the ~8MB model (see turndetect/).
ENDPOINT_MODE = os.getenv("ENDPOINT_MODE", "silence")   # "silence" | "semantic"
ENDPOINT_MIN_SILENCE_MS = int(os.getenv("ENDPOINT_MIN_SILENCE_MS", "450"))
ENDPOINT_MAX_SILENCE_MS = int(os.getenv("ENDPOINT_MAX_SILENCE_MS", "1400"))
SMART_TURN_MODEL = os.getenv("SMART_TURN_MODEL", "models/smart-turn-v3.2-cpu.onnx")
# P(complete) at/above this ends the turn at the MIN checkpoint.
SEMANTIC_TURN_THRESHOLD = float(os.getenv("SEMANTIC_TURN_THRESHOLD", "0.5"))
# Ignore speech blips shorter than this (a cough, a "um" click) so they don't
# fire a whole empty turn.
MIN_SPEECH_MS = int(os.getenv("MIN_SPEECH_MS", "250"))
# Keep a little audio from just before VAD triggered, so the first phoneme isn't
# clipped (VAD always fires a hair late).
SPEECH_PREROLL_MS = int(os.getenv("SPEECH_PREROLL_MS", "160"))

# ---- Step 4: barge-in (interrupting the interviewer mid-sentence) ------------
# While the AI is speaking we keep listening; if we detect the user talking, we
# cancel the reply and hand the floor back. To avoid false triggers from the AI's
# own voice leaking past echo cancellation, barge-in uses a STRICTER speech bar
# and needs sustained speech (not one stray frame) before it fires.
BARGEIN_THRESHOLD = float(os.getenv("BARGEIN_THRESHOLD", "0.55"))
BARGEIN_MIN_SPEECH_MS = int(os.getenv("BARGEIN_MIN_SPEECH_MS", "240"))
# Quiet frames tolerated *inside* a barge-in run before it's abandoned. Real
# speech dips under the VAD bar between syllables; without this tolerance the
# run only completes if you shout without pausing (measured: interrupting felt
# broken). Raise it if the AI's own voice leaks past echo cancellation and
# interrupts itself; lower it if interruptions feel too eager.
BARGEIN_GAP_MS = int(os.getenv("BARGEIN_GAP_MS", "200"))

# ---- Spoken-output guard (speech_filter.py) ---------------------------------
# The persona asks for exactly ONE question per turn. When on, the harness makes
# it true by ending the turn at the first "?" instead of trusting the model.
# (Bracketed stage directions are always stripped — that isn't optional.)
ONE_QUESTION_PER_TURN = os.getenv("ONE_QUESTION_PER_TURN", "on").lower() in (
    "on", "true", "1", "yes",
)

# ---- The interview director: an agentic tool-use loop -----------------------
# When on, each turn runs TWO LLM calls instead of one: first a "decide" loop
# where the model picks structured actions (note evidence, probe deeper, switch
# topic, end) as schema-constrained JSON, then the normal streamed "speak" call
# guided by the chosen move. Off = the original single-call chatbot behavior.
DIRECTOR = os.getenv("DIRECTOR", "on").lower() in ("on", "true", "1", "yes")
# Safety cap on the decide loop: a confused model can't spin forever.
DIRECTOR_MAX_STEPS = int(os.getenv("DIRECTOR_MAX_STEPS", "4"))
# Decide-call output cap. Actions are one short JSON object; keep it tight so
# the extra call costs as little latency as possible.
DIRECTOR_NUM_PREDICT = int(os.getenv("DIRECTOR_NUM_PREDICT", "120"))

# ---- Context compaction ------------------------------------------------------
# Keeps the LLM's view of the conversation under budget: system prompt pinned,
# oldest exchanges rolled into an incremental summary, recent turns verbatim.
# The session's FULL history is untouched (debrief + saved transcripts see all).
COMPACTION = os.getenv("COMPACTION", "on").lower() in ("on", "true", "1", "yes")
# ~4 chars/token; 24000 chars ≈ 6000 tokens, leaving llama3:8b's 8192 window
# room for the reply, the director's context, and template overhead.
COMPACTION_BUDGET_CHARS = int(os.getenv("COMPACTION_BUDGET_CHARS", "24000"))
COMPACTION_KEEP_RECENT = int(os.getenv("COMPACTION_KEEP_RECENT", "8"))

# ---- Hosted engines (only needed if you flip a flag above) ------------------
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")
CARTESIA_API_KEY = os.getenv("CARTESIA_API_KEY", "")
CARTESIA_VOICE_ID = os.getenv("CARTESIA_VOICE_ID", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
