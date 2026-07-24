"""Step 5 — the post-interview debrief.

Two independent signals get combined into one review:

  1. Delivery metrics — cheap, deterministic numbers we compute ourselves from
     data we already captured during the interview (transcript words + how long
     the candidate actually spoke). No model, no cost: words-per-minute, filler
     rate, average answer length, and how much of the airtime was theirs.

  2. Content scoring — the LLM acting as an *evaluator* (a different job from the
     conversational interviewer): it reads the whole transcript and scores it
     against a fixed rubric, returning strict JSON we can render.

Keeping these separate is deliberate — the metrics are trustworthy because they
are arithmetic, and the rubric scores are clearly labelled as a model's opinion.
"""

from __future__ import annotations

import json
import logging
import re

import engines
from packs import get_pack

log = logging.getLogger("interview-coach")

# Words we count as fillers. Multi-word phrases are matched first so "you know"
# isn't double-counted as "you" + "know".
#
# Deliberately excludes "right", "actually", "literally", "honestly": these are
# far more often legitimate content ("the RIGHT approach", "it ACTUALLY failed")
# than verbal filler, and counting them inflated the debrief's filler verdict on
# perfectly clean answers. We keep the high-signal ones — the true hesitation
# sounds plus the classic discourse fillers. "like"/"basically" still over-count
# a little (they have real uses too), but they're genuinely the most common
# interview fillers, so the trade favors keeping them.
_FILLERS = [
    "you know", "kind of", "sort of", "i mean",
    "um", "uh", "erm", "uhm", "hmm", "like", "basically",
]

# The rubric is now pack-driven (packs.py `rubric`) rather than hardcoded here —
# each topic (behavioral, dsa, ml, system_design) has its own 4 dimensions.
# `get_pack("behavioral").rubric` is exactly the four axes this used to be.

SCORING_SYSTEM = """You are an expert interview coach reviewing a completed {label} \
mock interview. You are given the full transcript (INTERVIEWER and CANDIDATE turns).

Score ONLY the candidate's performance, fairly and specifically, using this rubric \
(each 1-5, where 1 = poor, 3 = adequate, 5 = excellent):
{rubric}

Rules:
- Base every judgement on what the candidate ACTUALLY said. Do not invent details.
- If the interview is very short or the candidate gave non-answers, score low and say so.
- Strengths and improvements must be concrete and reference the transcript.
- Improvements must be actionable ("quantify your results" not "be better").

Respond with STRICT JSON and nothing else — no markdown, no prose around it — in exactly this shape:
{{
  "overall": <integer 0-100>,
  "headline": "<one-sentence overall verdict>",
  "dimensions": [
{dims_example}
  ],
  "strengths": ["<specific strength>", "<specific strength>"],
  "improvements": ["<specific, actionable fix>", "<specific, actionable fix>"]
}}"""


def build_transcript(messages: list[dict]) -> str:
    """Render the chat history as a readable INTERVIEWER/CANDIDATE script (the
    system prompt is dropped — the evaluator judges the conversation, not the
    instructions we gave the interviewer)."""
    lines: list[str] = []
    for m in messages:
        role = m.get("role")
        if role == "assistant":
            lines.append(f"INTERVIEWER: {m['content']}")
        elif role == "user":
            lines.append(f"CANDIDATE: {m['content']}")
    return "\n".join(lines)


def _count_fillers(text: str) -> int:
    low = f" {text.lower()} "
    n = 0
    for f in _FILLERS:
        # word-boundary match so "like" doesn't fire inside "unlikely"
        n += len(re.findall(rf"(?<![a-z]){re.escape(f)}(?![a-z])", low))
    return n


def delivery_metrics(turn_stats: list[dict], ai_speak_secs: float) -> dict:
    """Deterministic delivery stats from captured data. `turn_stats` is one entry
    per candidate answer: {"words": int, "speech_secs": float, "text": str}."""
    answers = [t for t in turn_stats if t["words"] > 0]
    total_words = sum(t["words"] for t in answers)
    speak_secs = sum(t["speech_secs"] for t in answers)
    fillers = sum(_count_fillers(t["text"]) for t in answers)

    wpm = round(total_words / (speak_secs / 60)) if speak_secs > 1 else None
    total_voice = speak_secs + ai_speak_secs
    talk_ratio = round(100 * speak_secs / total_voice) if total_voice > 0 else None

    avg_answer = round(total_words / len(answers)) if answers else 0
    fpm = round(fillers / (speak_secs / 60), 1) if speak_secs > 1 else None
    metrics = {
        "answers": len(answers),
        "total_words": total_words,
        "speaking_secs": round(speak_secs),
        "wpm": wpm,
        "fillers": fillers,
        "fillers_per_min": fpm,
        "avg_answer_words": avg_answer,
        "talk_ratio": talk_ratio,  # % of speaking time that was the candidate
    }
    # Turn the raw numbers into meaning: each metric gets a plain-English verdict
    # and a tone (good/warn/bad), plus a short coaching summary. This is computed,
    # not a model's guess — and it's what makes the debrief useful even when the
    # LLM's scored review fails.
    metrics["assessments"] = _assess_delivery(metrics)
    metrics["summary"] = _delivery_summary(metrics)
    return metrics


# Interview-delivery reference ranges (conversational-speech norms). Each entry is
# ordered; the first matching band wins. tone: "good" | "warn" | "bad".
def _assess_delivery(m: dict) -> dict:
    a: dict = {}

    wpm = m["wpm"]
    if wpm is not None:
        if wpm < 110:
            a["wpm"] = ("Measured, even a little slow — fine if it reads as calm, not hesitant.", "warn")
        elif wpm <= 165:
            a["wpm"] = ("A clear, easy-to-follow pace. Right in the pocket.", "good")
        elif wpm <= 190:
            a["wpm"] = ("A touch fast — brief pauses between ideas help the listener keep up.", "warn")
        else:
            a["wpm"] = ("Quite fast — you're likely losing the interviewer. Slow down and breathe.", "bad")

    avg = m["avg_answer_words"]
    if m["answers"]:
        if avg < 12:
            a["avg_answer_words"] = ("Very brief — answers this short rarely show your reasoning. Add the how and why.", "bad")
        elif avg <= 30:
            a["avg_answer_words"] = ("Concise. Good for rapport; make sure the technical ones go deeper.", "good")
        elif avg <= 90:
            a["avg_answer_words"] = ("Substantial, well-developed answers. Nice detail.", "good")
        else:
            a["avg_answer_words"] = ("Long — strong content, but watch for rambling. Land the point sooner.", "warn")

    fpm = m["fillers_per_min"]
    if fpm is not None:
        if fpm < 3:
            a["fillers_per_min"] = ("Clean delivery — very few filler words.", "good")
        elif fpm <= 6:
            a["fillers_per_min"] = ("A few fillers — noticeable but not distracting. A beat of silence beats an “um.”", "warn")
        else:
            a["fillers_per_min"] = ("Frequent fillers — they undercut confidence. Pause instead.", "bad")

    tr = m["talk_ratio"]
    if tr is not None:
        if tr < 45:
            a["talk_ratio"] = ("You spoke less than the interviewer — in a real interview YOU should carry most of the airtime.", "bad")
        elif tr <= 80:
            a["talk_ratio"] = ("A healthy balance — you led the conversation without steamrolling it.", "good")
        else:
            a["talk_ratio"] = ("You did nearly all the talking — good energy, but leave room to hear the question fully.", "warn")

    return a


def _delivery_summary(m: dict) -> str:
    """One or two sentences naming the 2-3 most salient delivery issues (or praise
    if there's nothing to fix). Built from the assessments, so it never contradicts
    the per-metric verdicts."""
    if not m.get("answers"):
        return "Not enough spoken answers this round to read your delivery."
    a = m.get("assessments", {})
    problems = [k for k, (_, tone) in a.items() if tone == "bad"]
    watch = [k for k, (_, tone) in a.items() if tone == "warn"]
    phrase = {
        "wpm": "your pace",
        "avg_answer_words": "how much you actually said",
        "fillers_per_min": "filler words",
        "talk_ratio": "how much airtime you took",
    }
    if not problems and not watch:
        return "Solid delivery all round — clear pace, good detail, and a confident, balanced conversation."
    lead = problems or watch[:2]
    names = [phrase.get(k, k) for k in lead[:3]]
    if len(names) == 1:
        focus = names[0]
    elif len(names) == 2:
        focus = f"{names[0]} and {names[1]}"
    else:
        focus = f"{names[0]}, {names[1]}, and {names[2]}"
    tail = ("The single biggest lever is giving fuller, more specific answers."
            if "avg_answer_words" in problems else
            "Small, deliberate changes here will make you noticeably clearer.")
    return f"Your delivery's biggest opportunities are {focus}. {tail}"


def _extract_json(raw: str) -> dict | None:
    """Pull the first JSON object out of the model's reply, tolerating stray
    prose or ```json fences around it — and, crucially, tolerating a truncated
    tail. Small models often emit valid JSON but forget the final `}`; we repair
    that by balancing any still-open braces/brackets before parsing."""
    start = raw.find("{")
    if start == -1:
        return None
    body = raw[start:]

    # Fast path: already-valid object somewhere in there.
    end = body.rfind("}")
    if end != -1:
        try:
            return json.loads(body[: end + 1])
        except json.JSONDecodeError:
            pass

    # Repair path: walk the text tracking bracket depth (ignoring anything inside
    # strings), then append the closers needed to balance it.
    stack: list[str] = []
    in_str = esc = False
    for ch in body:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]" and stack:
            stack.pop()

    candidate = body + ('"' if in_str else "") + "".join(
        "}" if c == "{" else "]" for c in reversed(stack)
    )
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


async def generate_debrief(
    messages: list[dict],
    turn_stats: list[dict],
    ai_speak_secs: float,
    director_notes: str | None = None,
    interview_type: str = "behavioral",
) -> dict:
    """Produce the full debrief payload: rubric scores (LLM) + delivery metrics
    (computed). Never raises — on any model/parse failure it degrades to metrics
    plus a plain note, so the user always gets *something*.

    `director_notes` is the agent's own evidence notebook (director.py), when the
    director ran: structured observations made LIVE during the interview, which
    ground the judge better than the raw transcript alone.

    `interview_type` selects the topic pack (packs.py) whose 4-dimension rubric
    the judge scores against — behavioral keeps today's four axes exactly;
    dsa/ml/system_design get their own domain-appropriate axes."""
    metrics = delivery_metrics(turn_stats, ai_speak_secs)

    if metrics["answers"] == 0:
        return {
            "type": "debrief",
            "ok": False,
            "headline": "Not enough to review — the interview ended before you answered anything.",
            "metrics": metrics,
        }

    pack = get_pack(interview_type)
    transcript = build_transcript(messages)
    rubric = "\n".join(f"- {name}: {desc}" for name, desc in pack.rubric)
    dims_example = ",\n".join(
        f'    {{"name": "{name}", "score": <1-5>, "comment": "<one sentence, specific>"}}'
        for name, _desc in pack.rubric
    )
    user_content = f"TRANSCRIPT:\n{transcript}"
    if director_notes:
        user_content += (
            "\n\nINTERVIEWER'S OWN NOTES (taken live during the interview — use them "
            f"as evidence, but judge from the transcript):\n{director_notes}"
        )
    scoring_messages = [
        {
            "role": "system",
            "content": SCORING_SYSTEM.format(
                label=pack.label, rubric=rubric, dims_example=dims_example
            ),
        },
        {"role": "user", "content": f"{user_content}\n\nReturn the JSON review now."},
    ]

    try:
        # This JSON report needs room for both model reasoning and the visible
        # dimensions/feedback when the hosted Responses engine is selected.
        raw = await engines.get_llm().reply(scoring_messages, max_tokens=1500)
        scored = _extract_json(raw)
    except Exception:  # noqa: BLE001 — the debrief must never crash the session
        log.exception("debrief scoring failed")
        scored = None

    if scored is None:
        return {
            "type": "debrief",
            "ok": True,
            "headline": "Here are your delivery metrics. (The scored review couldn't be generated this time.)",
            "metrics": metrics,
            "dimensions": [],
            "strengths": [],
            "improvements": [],
        }

    return {
        "type": "debrief",
        "ok": True,
        "overall": scored.get("overall"),
        "headline": scored.get("headline", "Interview complete."),
        "dimensions": scored.get("dimensions", []),
        "strengths": scored.get("strengths", []),
        "improvements": scored.get("improvements", []),
        "metrics": metrics,
    }
