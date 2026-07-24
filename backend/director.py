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
import re
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
    "additionalProperties": False,
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

    # Consecutive probe_deeper moves on the CURRENT thread. Notes don't touch it;
    # a switch_topic resets it to 0. The hard loop-breaker in decide() reads this
    # to force a switch once the same thread has been probed DIRECTOR_MAX_PROBES
    # times — which is the root fix for the interviewer re-asking one question
    # forever (the model chose probe_deeper every turn; nothing overrode it).
    times_probed: int = 0

    # Which topic pack / company overlay this interview is running (packs.py,
    # companies.py) — set once at setup (main.py) and read here so the director
    # prompt gets domain-appropriate judgment guidance without changing the
    # ACTION_SCHEMA or the `decide()` call signature at all.
    interview_type: str = "behavioral"
    company: str = "generic"

    # Coding round: the problem currently on the candidate's screen, and — when a
    # spoken "give me a different problem" is detected — the one to switch TO. The
    # session (main.py) reads `pending_problem` after the turn to update the editor.
    problem: object = None
    pending_problem: object = None
    # Every problem id shown this session, so a switch never hands back one the
    # candidate has already worked ("no, we've already been through this").
    shown_problems: set = field(default_factory=set)

    def apply(self, action: dict) -> None:
        kind, detail = action["action"], action["detail"]
        if kind == "note_evidence":
            self.evidence.append(detail)
        elif kind == "note_red_flag":
            self.red_flags.append(detail)
        elif kind == "probe_deeper":
            self.times_probed += 1
        elif kind == "switch_topic":
            self.topics.append(detail)
            self.times_probed = 0  # fresh thread — the probe counter resets
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


# Candidate meta-requests we honor DETERMINISTICALLY — the model doesn't get a
# vote, because live it repeatedly ignored "can we move on to the next question?"
# and re-asked the same thing. Anchored to phrases (not bare words like "stop")
# so an answer that merely mentions stopping a service doesn't end the interview.
# NOTE: no BARE "move on" — that false-fires on an algorithm explanation ("…and
# move on to the next element"). Require an explicit control lead-in or an actual
# topic noun.
_MOVE_ON_RE = re.compile(
    r"\b(next question|another question|a different question|new question|"
    r"skip (?:this|that|it)|(?:can we|let'?s|let us|shall we|could we) move on|"
    r"move on to (?:the next|a|another) (?:question|problem|topic))\b",
    re.IGNORECASE,
)
_STOP_RE = re.compile(
    r"\b(stop the interview|end the interview|can we stop|let'?s stop|"
    r"i'?m done|that'?s all|wrap (?:this|it)? ?up|finish the interview)\b",
    re.IGNORECASE,
)
# The candidate wants to go BACK to / revisit a specific earlier problem. The
# interviewer kept deflecting ("we'll get there", "let's build up to it") and
# ignoring this — so honor it deterministically: re-pose what they asked for.
_GO_BACK_RE = re.compile(
    r"\b(go back|previous (?:problem|question|one)|earlier (?:problem|question)|"
    r"that (?:question|problem|one) only|i want that|revisit|come back to|"
    r"the (?:one|problem) (?:you|we) (?:asked|mentioned|had) (?:earlier|before))\b",
    re.IGNORECASE,
)
# The candidate wants to STAY on the current problem (don't switch away).
_STAY_RE = re.compile(
    r"\b(stay on|keep going|keep (?:this|working)|don'?t (?:move on|switch|change)|"
    r"not (?:that|another) one|listen to me|hold on)\b",
    re.IGNORECASE,
)


def _meta_request_action(text: str) -> dict | None:
    """A forced director move when the candidate explicitly STEERS the interview,
    or None. Checked before the model runs so it CANNOT be overruled — the whole
    point is that the candidate's stated wishes beat the model's own agenda."""
    if not text:
        return None
    if _STOP_RE.search(text):
        return {"action": "end_interview", "detail": "the candidate asked to stop"}
    # "go back" is checked before "move on"/"stay": it's the most specific intent
    # and "go back to the previous question" shouldn't read as "move on".
    if _GO_BACK_RE.search(text):
        return {
            "action": "switch_topic",
            "detail": "the candidate asked to RETURN to an earlier problem — re-pose "
            "the specific problem they referenced and work through it with them; do "
            "NOT introduce a brand-new topic",
        }
    if _STAY_RE.search(text):
        return {
            "action": "probe_deeper",
            "detail": "the candidate asked to STAY on the current problem — keep "
            "working it with them (a hint if they're stuck); do NOT switch topics",
        }
    if _MOVE_ON_RE.search(text):
        return {
            "action": "switch_topic",
            "detail": "the candidate asked to move on — switch to a fresh, relevant "
            "topic not yet covered",
        }
    return None


def _loop_break(action: dict, state: DirectorState) -> dict:
    """Enforce the 'don't probe the same thread forever' rule the model keeps
    breaking: once the current thread has been probed DIRECTOR_MAX_PROBES times,
    a further probe_deeper is rewritten into a switch_topic."""
    if action["action"] == "probe_deeper" and state.times_probed >= config.DIRECTOR_MAX_PROBES:
        return {
            "action": "switch_topic",
            "detail": "this thread has been probed enough — move to a fresh, relevant "
            "area not yet covered",
        }
    return action


def _end_guard(action: dict, exchanges: int) -> dict:
    """Refuse a premature end_interview. The small utility model sometimes picks
    it two turns in — observed live: the candidate asked for background on a system
    design prompt and the round was closed out at Q2. An explicit "let's stop" is
    handled BEFORE the model ever runs (_meta_request_action), so the model itself
    never has a legitimate reason to end this early."""
    if action["action"] == "end_interview" and exchanges < config.DIRECTOR_MIN_EXCHANGES:
        log.info("director: blocked premature end_interview (%d exchanges)", exchanges)
        return {
            "action": "probe_deeper",
            "detail": "the interview has barely started — stay with the current thread "
            "and help them engage with it",
        }
    return action


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
    last_user_text = (
        history[-1]["content"] if history and history[-1]["role"] == "user" else ""
    )
    # How far in are we? Used to block a premature model-initiated end_interview.
    exchanges = sum(1 for m in history if m.get("role") == "user")

    # (D) A candidate meta-request ("can we move on?", "let's stop") is honored
    # deterministically, BEFORE the model runs — so it can't be ignored the way
    # the LLM ignored it live. This is a terminal move on its own.
    forced = _meta_request_action(last_user_text)
    if forced is not None:
        log.info("director: candidate meta-request -> %s", forced["action"])
        state.apply(forced)
        yield forced
        return

    # The director reads the same compacted view the speaker does: user/assistant
    # turns verbatim (last 16), PLUS the compaction summary when there is one — a
    # system message the OLD filter dropped, so on long interviews the director
    # was blind to everything past the recent window and re-opened covered ground.
    summary = next(
        (m["content"] for i, m in enumerate(history) if i > 0 and m["role"] == "system"),
        None,
    )
    convo = [
        f"{'INTERVIEWER' if m['role'] == 'assistant' else 'CANDIDATE'}: {m['content']}"
        for m in history
        if m["role"] in ("user", "assistant")
    ]
    interview_so_far = ("EARLIER (summary): " + summary + "\n\n" if summary else "") + \
        "\n".join(convo[-16:])
    messages = [
        {"role": "system", "content": build_director_prompt(
            state.render(), state.interview_type, state.company
        )},
        {
            "role": "user",
            "content": "INTERVIEW SO FAR:\n" + interview_so_far
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
                reasoning_effort=config.OPENAI_DIRECTOR_REASONING_EFFORT,
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

        # (C) Enforce the probe cap and the no-premature-ending rule before we commit.
        action = _end_guard(_loop_break(action, state), exchanges)
        state.apply(action)
        yield action
        if action["action"] in TERMINAL_ACTIONS:
            return

        # Non-terminal (a note): acknowledge and ask for the next action.
        messages.append({"role": "assistant", "content": json.dumps(action)})
        messages.append({"role": "user", "content": "Noted. Choose your ONE next action as JSON."})

    # Cap reached / model kept note-taking / call failed: force a safe move so
    # the interview always continues — still subject to the loop-breaker, so a
    # forced fallback can't itself extend a probe loop.
    fallback = _loop_break(dict(_FALLBACK), state)
    log.info("director: forcing fallback move (%s)", fallback["action"])
    state.apply(fallback)
    yield fallback
