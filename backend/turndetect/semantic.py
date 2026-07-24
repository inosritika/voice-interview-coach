"""Semantic turn completion — the third endpointing signal.

The other two signals are acoustic: Silero VAD hears *silence*, and smart-turn
(turndetect/smart_turn.py) hears *prosody* — the melody of the voice. Neither
knows what was actually SAID. This one does: it transcribes the utterance so far
and asks the LLM whether the words form a complete thought or trail off
mid-sentence.

    "My biggest achievement was leading the"   -> incomplete (dangling verb)
    "and that's why I left that role."          -> complete (finished clause)

Why a separate model at all, when prosody already guesses completion? Because
they fail on different inputs. Prosody misreads flat/uptalk speakers; semantics
misreads someone who pauses mid-clause to think. Fusing an acoustic and a
linguistic signal is exactly how production turn-taking is built — each covers
the other's blind spot.

Same TurnChecker interface as the prosody model (completeness(pcm) -> float), so
the endpointer fuses them without caring which is which. It's expensive
(transcribe + LLM), so the caller CASCADES: only invoked when prosody is
genuinely uncertain (see main.py `_utterance_completeness`).
"""

import json
import logging

from turndetect.base import TurnChecker

log = logging.getLogger("interview-coach")

# Tiny constrained output: a small local model does a clean boolean far more
# reliably than free-form text (the director/debrief lesson, again).
_SCHEMA = {
    "type": "object",
    "properties": {"complete": {"type": "boolean"}},
    "required": ["complete"],
    "additionalProperties": False,
}

_SYSTEM = (
    "You decide, from a partial transcript, whether a person speaking in an "
    "interview has FINISHED their thought or is still mid-sentence and about to "
    "continue. Judge only from the words and grammar. Reply complete=true only if "
    "the text reads as a finished statement or question; reply complete=false if "
    "it ends on a dangling word (a conjunction, article, preposition, or an "
    "unfinished clause) that clearly expects more to follow."
)


class SemanticTurnChecker(TurnChecker):
    """P(complete) from the MEANING of the utterance-so-far: STT -> LLM judgment."""

    async def completeness(self, pcm: bytes) -> float:
        # Lazy import to avoid an engines <-> turndetect import cycle at module load.
        import engines

        try:
            text = (await engines.get_stt().transcribe(pcm)).strip()
        except Exception:  # noqa: BLE001 — a failed transcription just abstains
            log.exception("semantic turn: transcription failed")
            return 0.5
        if not text:
            return 0.0  # nothing intelligible yet — treat as "keep waiting"

        try:
            raw = await engines.get_llm().reply(
                [
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": f'Partial transcript: "{text}"\n\nIs the thought complete?'},
                ],
                max_tokens=16,
                format_schema=_SCHEMA,
                reasoning_effort="none",  # a snap judgment; no deliberation needed
            )
            complete = bool(json.loads(raw).get("complete"))
            log.info("semantic turn: %s -> complete=%s", text[-60:], complete)
            return 1.0 if complete else 0.0
        except Exception:  # noqa: BLE001 — abstain (0.5) so prosody decides alone
            log.exception("semantic turn: judgment failed")
            return 0.5
