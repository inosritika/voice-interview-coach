"""Agent-vs-agent evals — regression tests for the interviewer's BEHAVIOR.

Every prompt change so far was judged by vibes: talk to the interviewer, feel
whether it got better. That doesn't scale past two rules. This harness turns
"did my change help?" into a number:

    simulated candidate  <-- interviews -->  the REAL interviewer stack
        (an LLM playing                    (same system prompt, same director
         a persona)                          loop, same directive plumbing —
                                             via pipeline.cascaded.reply_events)
                       |
                       v
                 judge model reads the transcript and scores the
                 INTERVIEWER (not the candidate) against a checklist,
                 as schema-constrained JSON

Personas are chosen to hit the failure modes we've actually seen: an evasive
candidate (tests the don't-move-on rule), a rambler (tests brevity + focus),
and a strong candidate (tests probing for numbers instead of coasting).

Text-only on purpose: STT/TTS don't change what the interviewer *decides*, and
skipping them makes a run cheap enough to use after every prompt edit.

Usage (from backend/, venv active):
    python -m evals.run_eval                       # all personas, 4 exchanges
    python -m evals.run_eval --personas evasive --turns 3
Results print as a table and are saved to data/evals/<timestamp>.json.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import engines  # noqa: E402
from director import DirectorState  # noqa: E402
from pipeline.base import DirectorAction, ReplyToken  # noqa: E402
from pipeline.cascaded import reply_events  # noqa: E402
from prompts import build_system_prompt  # noqa: E402

EVALS_DIR = pathlib.Path(__file__).resolve().parent.parent / "data" / "evals"

_JD = "Senior Backend Engineer: distributed systems, Postgres, high-traffic APIs, on-call ownership."
_RESUME = "5 years backend. Led a payments migration. Python/Go. Some Kubernetes."

PERSONAS = {
    "strong": (
        "You are a STRONG candidate in a mock behavioral interview. Answer in 2-4 spoken "
        "sentences: concrete stories, real numbers (latency, revenue, team size), clear "
        "STAR structure. Stay in character; never mention being an AI."
    ),
    "rambler": (
        "You are a RAMBLING candidate in a mock behavioral interview. Answer every question "
        "with 6-8 sentences that wander across several half-finished anecdotes, heavy on "
        "filler ('you know', 'kind of'), light on numbers. You mean well but never land the "
        "point. Stay in character; never mention being an AI."
    ),
    "evasive": (
        "You are an EVASIVE candidate in a mock behavioral interview. Never actually answer "
        "what was asked: deflect with generalities ('teamwork is really important to me'), "
        "change the subject, or give one vague sentence. Never provide concrete examples or "
        "numbers, even when pressed. Stay in character; never mention being an AI."
    ),
}

# What we score the INTERVIEWER on — each criterion traces to a rule in
# prompts.py or a failure mode we actually shipped and fixed.
JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "one_question_per_turn": {"type": "integer", "minimum": 1, "maximum": 5},
        "brevity": {"type": "integer", "minimum": 1, "maximum": 5},
        "builds_on_answers": {"type": "integer", "minimum": 1, "maximum": 5},
        "probes_vague_answers": {"type": "integer", "minimum": 1, "maximum": 5},
        "no_fabrication": {"type": "integer", "minimum": 1, "maximum": 5},
        "comment": {"type": "string"},
    },
    "required": [
        "one_question_per_turn", "brevity", "builds_on_answers",
        "probes_vague_answers", "no_fabrication", "comment",
    ],
}
CRITERIA = [k for k in JUDGE_SCHEMA["properties"] if k != "comment"]

JUDGE_SYSTEM = """You evaluate the INTERVIEWER in a mock-interview transcript (the \
candidate is a test dummy — do not score them). Score each criterion 1-5 (1=clearly \
violated, 3=mixed, 5=consistently satisfied):

- one_question_per_turn: every interviewer turn asks exactly ONE question.
- brevity: interviewer turns are 1-2 sentences, spoken-style, no lists/markdown.
- builds_on_answers: questions react to what the candidate actually said.
- probes_vague_answers: vague/evasive/unquantified answers get a sharp follow-up
  instead of the interviewer moving on.
- no_fabrication: the interviewer never asserts details (numbers, tech, projects)
  the candidate didn't state.

Judge only from the transcript. Reply as JSON matching the schema."""


async def interviewer_turn(history: list[dict], state: DirectorState, verbose: bool) -> str:
    """One turn of the real interviewer brain; returns the spoken text."""
    parts: list[str] = []
    async for ev in reply_events(history, state if history[-1]["role"] == "user" else None):
        if isinstance(ev, DirectorAction) and verbose:
            print(f"      [director] {ev.text[:90]}")
        elif isinstance(ev, ReplyToken):
            parts.append(ev.text)
    return "".join(parts).strip()


async def candidate_turn(persona_prompt: str, history: list[dict]) -> str:
    """The simulated candidate answers. Roles flip: interviewer words become
    'user' from the candidate model's point of view."""
    flipped = [{"role": "system", "content": persona_prompt}]
    for m in history:
        if m["role"] == "assistant":
            flipped.append({"role": "user", "content": m["content"]})
        elif m["role"] == "user":
            flipped.append({"role": "assistant", "content": m["content"]})
    return (await engines.get_llm().reply(flipped, max_tokens=512)).strip()


async def judge(transcript: str) -> dict:
    raw = await engines.get_llm().reply(
        [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": f"TRANSCRIPT:\n{transcript}\n\nReturn the JSON scores."},
        ],
        max_tokens=768,
        format_schema=JUDGE_SCHEMA,
    )
    return json.loads(raw)


async def run_persona(name: str, turns: int, verbose: bool) -> dict:
    print(f"\n=== persona: {name} ({turns} exchanges) ===")
    history = [{"role": "system", "content": build_system_prompt(_JD, _RESUME)}]
    state = DirectorState()

    greeting = await interviewer_turn(history, state, verbose)
    history.append({"role": "assistant", "content": greeting})
    print(f"  INTERVIEWER: {greeting}")

    for _ in range(turns):
        answer = await candidate_turn(PERSONAS[name], history)
        history.append({"role": "user", "content": answer})
        print(f"  CANDIDATE:   {answer[:120]}{'…' if len(answer) > 120 else ''}")

        reply = await interviewer_turn(history, state, verbose)
        history.append({"role": "assistant", "content": reply})
        print(f"  INTERVIEWER: {reply}")

    transcript = "\n".join(
        f"{'INTERVIEWER' if m['role'] == 'assistant' else 'CANDIDATE'}: {m['content']}"
        for m in history[1:]
    )
    scores = await judge(transcript)
    return {"persona": name, "scores": scores, "transcript": transcript,
            "director_notes": state.notes_for_debrief()}


def print_table(results: list[dict]) -> None:
    short = {c: c.replace("_", " ")[:18] for c in CRITERIA}
    width = max(len(v) for v in short.values())
    print("\n" + " " * (width + 2) + "  ".join(f"{r['persona']:>8}" for r in results))
    for c in CRITERIA:
        row = "  ".join(f"{r['scores'][c]:>8}" for r in results)
        print(f"{short[c]:<{width}}  {row}")
    avg = ["%8.1f" % (sum(r["scores"][c] for c in CRITERIA) / len(CRITERIA)) for r in results]
    print(f"{'AVERAGE':<{width}}  " + "  ".join(avg))
    for r in results:
        print(f"\n[{r['persona']}] judge: {r['scores']['comment']}")


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--personas", default=",".join(PERSONAS), help="comma-separated subset")
    ap.add_argument("--turns", type=int, default=4, help="exchanges per persona")
    ap.add_argument("--verbose", action="store_true", help="show director actions")
    args = ap.parse_args()

    t0 = time.perf_counter()
    results = []
    for name in [p.strip() for p in args.personas.split(",") if p.strip()]:
        if name not in PERSONAS:
            raise SystemExit(f"unknown persona {name!r} — choose from {list(PERSONAS)}")
        results.append(await run_persona(name, args.turns, args.verbose))

    print_table(results)

    EVALS_DIR.mkdir(parents=True, exist_ok=True)
    out = EVALS_DIR / (time.strftime("%Y%m%d-%H%M%S") + ".json")
    out.write_text(json.dumps({"turns": args.turns, "results": results}, indent=2))
    print(f"\nsaved {out.relative_to(EVALS_DIR.parent.parent)} · {time.perf_counter() - t0:.0f}s total")


if __name__ == "__main__":
    asyncio.run(main())
