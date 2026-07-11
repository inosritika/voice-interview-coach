"""A *turn strategy* — how one conversational turn is produced.

This is the swap point the user asked for: choose the whole pipeline shape with a
flag, without touching anything else. Everything around a turn — VAD, endpointing,
the floor state, chunking the reply, TTS, streaming to the browser, the latency
HUD — lives in main.py and is *shared*. Only "given the user's audio + the
conversation so far, produce the transcript and the reply" varies:

  - CascadedStrategy: STT model -> LLM model (two stages, full visibility)
  - FusedStrategy:    Gemma 4 does both in one call (deletes the STT->LLM handoff)

A strategy emits a stream of typed events. The Transcript (what the user said) is
used for the on-screen log and the conversation history; the ReplyToken pieces are
the interviewer's reply, streamed so speech can start early. Cascaded emits the
Transcript first (it transcribes, then thinks); fused may emit it last (it speaks
first for lower latency). The caller doesn't care about the order.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass


@dataclass
class Transcript:
    """What the user said — for display and for the conversation history."""

    text: str


@dataclass
class ReplyToken:
    """One piece of the interviewer's reply, as it streams out."""

    text: str


@dataclass
class DirectorAction:
    """One action the interview director took while deciding this turn — e.g.
    "note_evidence: quantified the migration's impact". Surfaced so the UI can
    show the agent thinking; never spoken."""

    text: str


TurnEvent = Transcript | ReplyToken | DirectorAction


class TurnStrategy(ABC):
    async def warmup(self) -> None:
        """Optional: do the expensive one-time setup (load weights, compile GPU
        kernels) at server startup instead of on the user's first turn. Default is
        a no-op — cheap pipelines don't need it. Overridden by the fused (Gemma)
        strategy, whose first inference is very slow cold."""
        return

    @abstractmethod
    def run(
        self,
        utterance_pcm: bytes | None,
        history: list[dict],
        director_state=None,
    ) -> AsyncIterator[TurnEvent]:
        """Stream one turn as events.

        `utterance_pcm` is the captured user audio (raw 16 kHz mono s16le), or
        None for the opening greeting (no user turn yet). `history` is the prior
        conversation (system + past turns), which the strategy reads but does not
        mutate — the caller owns history.

        `director_state` is the session's DirectorState when the agentic
        director is enabled (cascaded only, for now). Strategies that don't use
        it just ignore it.
        """
        ...
