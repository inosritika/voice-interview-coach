"""Context compaction — the rolling summary that closes the project's oldest
open item (learning-guide §12: unbounded history vs llama3's 8K window).

The problem: `Session.messages` grows every turn and is sent whole to the LLM.
Past the context window, Ollama truncates FROM THE TOP — eating the system
prompt first, so the interviewer silently forgets the job and its rules.

The fix is the same mechanism every production agent harness uses on long
conversations: keep the full record, but give the model a *view* of it —
   [ system prompt (pinned) ]
   [ one summary block covering the oldest exchanges ]
   [ the most recent turns, verbatim ]

Design points worth noticing:
  - The Session keeps FULL history (the debrief and saved transcript must never
    see a lossy view). Compaction only shapes what the LLM reads.
  - Summarization is INCREMENTAL: each compaction folds only the newly-old
    turns into the existing summary (one small LLM call), instead of
    re-summarizing the whole interview every turn.
  - The summary is a system-role message at index 1 — early system messages
    are fine; it's a TRAILING system message that breaks llama3's template
    (found in the director build).
  - Budgets are in characters (~4 chars/token) because exact token counting
    would drag in a tokenizer dependency for a number we only need roughly.
"""

from __future__ import annotations

import logging

import config

log = logging.getLogger("interview-coach")

_SUMMARIZE_SYSTEM = """You maintain the running summary of a mock job interview. \
Fold the NEW EXCHANGES into the EXISTING SUMMARY and return the updated summary.

Rules:
- Third person, dense, factual: topics asked about, what the candidate claimed
  (skills, numbers, outcomes), notable strengths/weaknesses shown.
- Keep every concrete detail the interviewer might need to refer back to.
- No commentary, no advice, no formatting — one compact paragraph, max ~150 words."""


def _size(messages: list[dict]) -> int:
    return sum(len(m["content"]) for m in messages)


class Compactor:
    """One per Session. Call `view(messages)` right before an LLM call: returns
    `messages` untouched while it fits the budget, or the compacted view once it
    doesn't. Never raises — if the summary call fails, it falls back to a plain
    truncated view (system + recent turns), which is still better than letting
    the runtime eat the system prompt."""

    def __init__(self, llm) -> None:
        self._llm = llm
        self._summary: str = ""
        self._covered = 1  # messages[1:_covered] are folded into _summary

    async def view(self, messages: list[dict]) -> list[dict]:
        if not config.COMPACTION or _size(messages) <= config.COMPACTION_BUDGET_CHARS:
            return messages

        # Everything except the system prompt and the last N turns gets folded.
        cut = max(self._covered, len(messages) - config.COMPACTION_KEEP_RECENT)
        if cut > self._covered:
            new_turns = messages[self._covered : cut]
            try:
                self._summary = await self._summarize(new_turns)
                self._covered = cut
            except Exception:  # noqa: BLE001 — degrade, never break the turn
                log.exception("compaction: summary call failed; using truncated view")
                # Without a summary we can still protect the system prompt.
                return [messages[0]] + messages[cut:]

        compacted = [
            messages[0],
            {
                "role": "system",
                "content": "Summary of the interview so far (older exchanges, "
                f"condensed): {self._summary}",
            },
            *messages[self._covered :],
        ]
        log.info(
            "compaction: %d msgs / %d chars -> %d msgs / %d chars",
            len(messages), _size(messages), len(compacted), _size(compacted),
        )
        return compacted

    async def _summarize(self, new_turns: list[dict]) -> str:
        rendered = "\n".join(
            f"{'INTERVIEWER' if m['role'] == 'assistant' else 'CANDIDATE'}: {m['content']}"
            for m in new_turns
            if m["role"] in ("user", "assistant")
        )
        prompt = (
            f"EXISTING SUMMARY:\n{self._summary or '(none yet — this is the start)'}\n\n"
            f"NEW EXCHANGES:\n{rendered}\n\nReturn the updated summary."
        )
        return await self._llm.reply(
            [
                {"role": "system", "content": _SUMMARIZE_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            # Responses caps include reasoning as well as visible output.
            max_tokens=512,
        )
