"""The interview director — a hand-rolled agentic tool-use loop.

This is the project's first real *agent*: instead of one LLM call that goes
straight from transcript to spoken words, each turn now runs a small loop where
the model chooses structured ACTIONS and this harness validates and executes
them. The pieces map one-to-one onto what any agent framework would give you —
built by hand here so each is visible:

  - a tool menu with a typed schema        (ACTION_SCHEMA)
  - constrained decoding                   (format_schema= on the LLM call —
                                            invalid JSON is impossible, not
                                            just discouraged)
  - validation + bounded retries           (_parse_action / the retry branch)
  - an iteration cap                       (config.DIRECTOR_MAX_STEPS)
  - state OUTSIDE the model                (DirectorState — the evidence
                                            notebook survives regardless of
                                            what the model remembers)

Two calls per turn, one job each (the §16 lesson from the fused experiment):
call 1..N = decide (JSON, tiny, this module), then ONE streamed speak call
(pipeline/cascaded.py) guided by the chosen move — so token streaming and the
low time-to-first-audio from step 3 are preserved.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

import config
from prompts import build_director_prompt

log = logging.getLogger("interview-coach")

# The tool menu. Deliberately FLAT (one "detail" string instead of per-action
# fields): flat schemas survive schema->grammar compilation reliably and give
# a small model less structure to get wrong.
ACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": [
                "note_evidence",
                "note_red_flag",
                "probe_deeper",
                "switch_topic",
                "end_interview",
            ],
        },
        "detail": {"type": "string"},
    },
    "required": ["action", "detail"],
}

# Actions that end the decide loop and shape the speak call.
TERMINAL_ACTIONS = {"probe_deeper", "switch_topic", "end_interview"}
# When the loop must be cut short (cap reached, model kept taking notes), fall
# back to the safest sensible move.
_FALLBACK = {"action": "probe_deeper", "detail": "follow up on the candidate's last answer"}


@dataclass
class DirectorState:
    """The interview's memory OUTSIDE the model. The model is consulted; this
    object is the record. The debrief reads it directly — structured judgment
    accumulated during the interview instead of one giant retrospective glance."""

    evidence: list[str] = field(default_factory=list)
    red_flags: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)  # from switch_topic moves
    ended: bool = False

    # Which topic pack / company overlay this interview is running (packs.py,
    # companies.py) — set once at setup (main.py) and read here so the director
    # prompt gets domain-appropriate judgment guidance without changing the
    # ACTION_SCHEMA or the `decide()` call signature at all.
    interview_type: str = "behavioral"
    company: str = "generic"

    def apply(self, action: dict) -> None:
        kind, detail = action["action"], action["detail"]
        if kind == "note_evidence":
            self.evidence.append(detail)
        elif kind == "note_red_flag":
            self.red_flags.append(detail)
        elif kind == "switch_topic":
            self.topics.append(detail)
        elif kind == "end_interview":
            self.ended = True

    def render(self) -> str:
        """The state as the director prompt sees it."""

        def block(title: str, items: list[str]) -> str:
            body = "\n".join(f"- {i}" for i in items) if items else "- (none yet)"
            return f"{title}:\n{body}"

        parts = [
            block("EVIDENCE NOTES SO FAR", self.evidence),
            block("RED FLAGS SO FAR", self.red_flags),
            block("TOPICS ALREADY COVERED", self.topics),
        ]
        if self.ended:
            parts.append("STATUS: you already chose to end this interview.")
        return "\n\n".join(parts)

    def notes_for_debrief(self) -> str | None:
        """Evidence + red flags as a block for the debrief judge, or None."""
        if not self.evidence and not self.red_flags:
            return None
        lines = [f"- {n}" for n in self.evidence]
        lines += [f"- (red flag) {n}" for n in self.red_flags]
        return "\n".join(lines)


def _parse_action(raw: str) -> dict | None:
    """Validate one model output against the tool menu. Constrained decoding
    should make failures impossible with Ollama — but the harness must not
    TRUST that (other engines, other models), so validation stays."""
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    action = obj.get("action")
    detail = obj.get("detail")
    schema_actions = ACTION_SCHEMA["properties"]["action"]["enum"]
    if action not in schema_actions or not isinstance(detail, str) or not detail.strip():
        return None
    return {"action": action, "detail": detail.strip()}


async def decide(
    llm, history: list[dict], state: DirectorState
) -> AsyncIterator[dict]:
    """The decide loop. Yields each validated action as it happens (so the UI
    can show the agent thinking); the LAST yielded action is always terminal —
    that's the directive for the speak call. Applies actions to `state` as a
    side effect. Never raises past itself: any model failure degrades to the
    fallback move, because a broken director must not break the interview."""
    transcript_lines = [
        f"{'INTERVIEWER' if m['role'] == 'assistant' else 'CANDIDATE'}: {m['content']}"
        for m in history
        if m["role"] in ("user", "assistant")
    ]
    messages = [
        {"role": "system", "content": build_director_prompt(
            state.render(), state.interview_type, state.company
        )},
        {
            "role": "user",
            "content": "INTERVIEW SO FAR:\n" + "\n".join(transcript_lines[-16:])
            + "\n\nChoose your ONE next action as JSON.",
        },
    ]

    retried = False
    for _ in range(config.DIRECTOR_MAX_STEPS):
        try:
            raw = await llm.reply(
                messages,
                max_tokens=config.DIRECTOR_NUM_PREDICT,
                format_schema=ACTION_SCHEMA,
            )
        except Exception:  # noqa: BLE001 — degrade, don't break the turn
            log.exception("director: decide call failed")
            break

        action = _parse_action(raw)
        if action is None:
            log.warning("director: invalid action output: %.120s", raw)
            if retried:
                break  # two strikes — stop burning calls, use the fallback
            retried = True
            messages.append({"role": "assistant", "content": raw})
            messages.append({
                "role": "user",
                "content": "That was not a valid action. Reply with ONE JSON object "
                "matching the schema — nothing else.",
            })
            continue

        state.apply(action)
        yield action
        if action["action"] in TERMINAL_ACTIONS:
            return

        # Non-terminal (a note): acknowledge and ask for the next action.
        messages.append({"role": "assistant", "content": json.dumps(action)})
        messages.append({"role": "user", "content": "Noted. Choose your ONE next action as JSON."})

    # Cap reached / model kept note-taking / call failed: force a safe move so
    # the interview always continues.
    log.info("director: forcing fallback move")
    state.apply(_FALLBACK)
    yield dict(_FALLBACK)
