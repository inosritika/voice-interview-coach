"""The cascaded turn strategy: STT model, then LLM model — two separate stages.

This is just the original step-1..3 flow, lifted behind the TurnStrategy interface
so it can sit next to the fused Gemma strategy and be chosen by a flag. It uses the
existing engine factory, so STT and LLM are each still independently swappable
(whisper/Deepgram, Ollama/OpenAI) underneath.

With DIRECTOR=on, the single LLM call becomes the agentic two-phase turn:
decide (director.py's action loop, tiny schema-constrained JSON calls) and then
the same streamed speak call as before, guided by the director's chosen move —
so step 3's token streaming and time-to-first-audio are untouched.
"""

import logging
from collections.abc import AsyncIterator

import config
import engines
from pipeline.base import DirectorAction, ReplyToken, Transcript, TurnEvent, TurnStrategy
from prompts import DIDNT_CATCH, DIRECTIVE_INSTRUCTIONS

log = logging.getLogger("interview-coach")


async def reply_events(
    turn_history: list[dict], director_state=None, opening_text: str | None = None
) -> AsyncIterator[TurnEvent]:
    """The interviewer's BRAIN, audio-free: given the context for this turn
    (ending with the user's message, or just the system prompt for a greeting),
    run the director's decide loop and stream the spoken reply.

    Split out of CascadedStrategy so the eval harness (evals/run_eval.py) can
    interview the exact same brain over text — same director, same prompts,
    same directive plumbing — without STT or TTS in the loop.
    """
    # The calibrated opening is deterministic so the selected area and
    # difficulty are guaranteed before adaptive LLM follow-ups begin.
    if opening_text is not None and turn_history[-1]["role"] == "system":
        yield ReplyToken(opening_text)
        return

    # Coding round: honor a spoken "give me a different problem / a medium one /
    # something harder" DETERMINISTICALLY — pick a new problem and present it, so
    # the editor and the interviewer stay in sync. Otherwise the model would just
    # improvise a problem that doesn't match what's loaded on screen. The session
    # (main.py) reads director_state.pending_problem after the turn to swap the
    # editor + reset the per-question timer.
    if (
        director_state is not None
        and getattr(director_state, "problem", None) is not None
        and turn_history[-1]["role"] == "user"
    ):
        import problems

        nxt = problems.match_switch(
            turn_history[-1]["content"],
            director_state.problem,
            exclude=getattr(director_state, "shown_problems", None),
        )
        if nxt is not None:
            director_state.problem = nxt
            director_state.pending_problem = nxt
            getattr(director_state, "shown_problems", set()).add(nxt.id)
            yield ReplyToken(problems.coding_switch_line(nxt))
            return

    # The director decides HOW to respond (only when there's a user answer to
    # analyze). Its terminal move is folded into the tail of the user message
    # for the speak call — NOT appended as a trailing system message: llama3's
    # chat template mishandles a conversation that ends with a system turn
    # (verified live: it echoes instead of replying).
    if config.DIRECTOR and director_state is not None and turn_history[-1]["role"] == "user":
        import director as director_mod

        directive = None
        async for action in director_mod.decide(
            engines.get_utility_llm(), turn_history, director_state
        ):
            yield DirectorAction(f"{action['action']}: {action['detail']}")
            directive = action  # the last action is always the terminal move
        instruction = DIRECTIVE_INSTRUCTIONS.get(directive["action"])
        if instruction is not None:
            directed = dict(turn_history[-1])  # copy — the caller owns history
            directed["content"] += (
                "\n\n[" + instruction.format(detail=directive["detail"]) + "]"
            )
            turn_history = turn_history[:-1] + [directed]

    async for token in engines.get_llm().stream(turn_history):
        yield ReplyToken(token)


class CascadedStrategy(TurnStrategy):
    async def warmup(self) -> None:
        """Pre-load STT, LLM and TTS at server boot so the user's FIRST turn isn't
        a cold-start. Measured: the first greeting was ~11s otherwise — Ollama
        pages llama3 into RAM, piper loads its voice, whisper loads its weights,
        all lazily on turn one. Each engine is warmed independently and
        failure-tolerant: a warmup miss (e.g. Ollama not up yet) logs and moves
        on — it must never stop the server from booting."""
        import time

        t0 = time.perf_counter()
        # STT: transcribing ~0.3s of silence loads the whisper model.
        try:
            silence = b"\x00" * (config.SAMPLE_RATE // 3 * 2)  # int16 = 2 bytes/sample
            await engines.get_stt().transcribe(silence)
        except Exception:  # noqa: BLE001
            log.exception("warmup: STT load failed (continuing)")
        # LLM: a 1-token reply pages an Ollama model into memory ("local"), or
        # pays the one-time `claude` CLI subprocess spawn ("claude"). Hosted
        # OpenAI has no local state to warm, and an artificially tiny Responses
        # budget can yield an incomplete response — so it's skipped. We warm the
        # PRIMARY (spoken reply) and UTILITY (director/compaction) engines both,
        # de-duped, since with the hybrid they differ (Claude speak + Ollama util).
        seen = set()
        for engine in (engines.get_llm(), engines.get_utility_llm()):
            if id(engine) in seen or type(engine).__name__ == "OpenAILLM":
                continue
            seen.add(id(engine))
            try:
                await engine.reply([{"role": "user", "content": "hi"}], max_tokens=1)
            except Exception:  # noqa: BLE001
                log.exception("warmup: LLM warm failed (continuing — is the engine reachable?)")
        # TTS: synthesizing one short word loads the piper voice.
        try:
            await engines.get_tts().synthesize("Ready.")
        except Exception:  # noqa: BLE001
            log.exception("warmup: TTS load failed (continuing)")
        log.info("cascaded warm (%.1fs) — first turn will be fast", time.perf_counter() - t0)

    async def run(
        self,
        utterance_pcm: bytes | None,
        history: list[dict],
        director_state=None,
        opening_text: str | None = None,
    ) -> AsyncIterator[TurnEvent]:
        turn_history = history
        if utterance_pcm is not None:
            # Real user turn: transcribe first, surface it, add it to the context
            # we hand the LLM (but not to the caller's history — the caller owns that).
            # Feed STT the current topic's jargon so technical terms land
            # ("priority queue", not "priority cube") — topic comes off the
            # director state the session already threads through.
            from packs import get_pack

            hint = get_pack(director_state.interview_type).stt_hint if director_state else None
            transcript = await engines.get_stt().transcribe(utterance_pcm, initial_prompt=hint)
            yield Transcript(transcript)
            if not transcript:
                # Nothing intelligible: don't run the LLM on an empty turn, but
                # don't go silent either — ask them to repeat (main.py records
                # this as the interviewer's turn, so the conversation continues).
                yield ReplyToken(DIDNT_CATCH)
                return
            turn_history = history + [{"role": "user", "content": transcript}]

        # Greeting (no user message) or reply: the shared brain does the rest.
        # (reply_events itself skips the director unless the context ends with
        # a user message, so the greeting stays a single fast call.)
        state = director_state if utterance_pcm is not None else None
        async for event in reply_events(turn_history, state, opening_text=opening_text):
            yield event
