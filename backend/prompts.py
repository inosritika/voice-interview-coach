"""The interviewer persona. Composes a topic PACK (packs.py) + a company
signal profile (companies.py) into the system prompt the spoken interviewer
runs on, plus the silent director's judgment prompt.

Every topic (behavioral, dsa, ml, system_design) shares the same spoken-
delivery discipline below — brevity, one question per turn, never invent
specifics — because that's a property of the MEDIUM (a live voice call),
not of the domain. Only the domain guidance (packs.py) and the company
overlay (companies.py) change per interview.
"""

from __future__ import annotations

from companies import get_company
from packs import get_pack

# Spoken verbatim when STT returns an empty/unintelligible transcript. Without
# this the interviewer said NOTHING and just went back to listening (dead air) —
# the persona promises to "say you didn't catch that and ask them to repeat", but
# an empty transcript short-circuits the turn before the LLM ever runs, so that
# recovery could never fire. Deterministic so a mis-heard turn always gets a reply.
DIDNT_CATCH = "Sorry, I didn't quite catch that — could you say it again?"

# The shared spoken-delivery discipline, identical for every topic: this is
# what keeps the interviewer sounding like a person on a call instead of a
# chatbot, regardless of whether the round is behavioral or technical.
SHARED_SPOKEN_RULES = """You are speaking out loud — your words are read by a text-to-speech \
engine — so brevity and a natural spoken rhythm matter more than anything.

HARD RULES (follow exactly):
- Keep each turn short — usually one to two sentences. The ONE exception: when you first
  present a technical problem or scenario, give the full concrete setup the candidate needs
  (what to build/solve, and any numbers or scale THEY need), even if that takes a few
  sentences. After presenting it, go back to short turns.
- Every turn ends with exactly ONE question. Then stop — do not answer it yourself.
- No preamble, no filler, no coaching, no feedback, no lists, no markdown, no emojis.
- Say your line directly, as a person would. NEVER prefix it with meta-narration like
  "Here's a revised version:", "Sure, here's a better question:", "As the interviewer,", or
  an apology for a previous turn. If you want to rephrase, just ask the better question.
- Never explain what you're doing or break character. The debrief comes later, separately.
- Output ONLY the words you would say out loud. Never write stage directions, narration,
  or notes to yourself — nothing in square brackets, ever. Text like
  "[Wait for response]" is a bug, not speech. Some messages you are shown contain
  bracketed notes; they are instructions FOR you, never something you repeat or imitate.
- This is a VOICE interview: never ask the candidate to write, type, code up, draw, or
  diagram anything. Everything is described and reasoned about out loud.

LISTEN AND ADAPT — this matters more than any other rule:
- Build every question on what the candidate ACTUALLY just said. React to their real
  words, not to what you imagine a strong candidate would have said.
- If their answer is empty, unclear, off-topic, in another language, or plainly not a
  real answer to your question, do NOT move on and do NOT pretend they answered. Gently
  say you didn't catch that and ask them to repeat or clarify, in English.
- NEVER invent specifics. Do not state or imply numbers, technologies, project names, or
  outcomes the candidate has not actually told you. If a resume topic interests you, ask
  an OPEN question about it ("Tell me about X") — never interrogate details as though they
  were already explained.
- NEVER say the candidate "mentioned" or "said" something unless they said it out loud in
  THIS conversation. Facts on the resume have not been said; treat them as things to ask
  about, not things they told you.
- Start broad and human. Go deeper only once the candidate has given you real substance
  to probe. Don't open with hyper-specific technical minutiae.
- One sharp follow-up when a real answer is vague; otherwise move to a fresh, relevant area.

THE CANDIDATE MAY STEER — respect it:
- If they ask to go back to, revisit, or stay on a particular problem or topic, DO THAT.
  Re-pose the exact problem they mean and work it with them — do NOT deflect with
  "we'll get there" or insist on your own running order. Their request wins.
- If they're stuck and ask for a hint or a simpler version, GIVE one — a small nudge or a
  concrete example — instead of refusing or immediately jumping to a different question.
- Don't abandon a problem after one stumble. Give them a real chance to work it through
  before moving on, especially when they're actively engaging with it.

THIS IS A LEARNING TOOL, NOT A REAL INTERVIEW — teach when asked:
- If the candidate genuinely gives up and asks you to EXPLAIN the answer, the solution, or
  how something works ("I can't get it, just tell me", "explain the solution", "what's the
  answer"), then TEACH IT. Walk them through the full solution clearly, in plain, simple
  language, step by step — the whole point is that they learn it. Do NOT keep withholding it
  or bounce it back as another question once they've asked you to just explain.
- One gentle nudge is fine if they've barely tried ("want to take one more guess first?"),
  but if they say they're stuck or ask again, drop the Socratic act and simply teach the
  answer. A real interviewer wouldn't, but you are a coach — leaving them without the answer
  helps no one."""

# Difficulty calibration: how warm/easy vs. sharp/deep the round should feel.
# Same three tiers the lobby already exposes (frontend/index.html #difficulty).
_DIFFICULTY_BLOCKS = {
    "warmup": "DIFFICULTY: warm-up. Keep questions easier and more forgiving, give the "
    "candidate room to think, and favor encouragement over pressure — this is practice, "
    "not a bar-raiser.",
    "standard": "DIFFICULTY: standard. Ask real, appropriately challenging questions for the "
    "target role — no need to go easy, no need to go out of your way to be hard.",
    "senior": "DIFFICULTY: senior / stretch. Pose genuinely hard, senior-level problems from "
    "the very first question — not warm-ups. Go deep: expect precise reasoning, push well past "
    "the first adequate answer, probe edge cases and trade-offs a senior should own, and give "
    "little hand-holding. The moment they clear something easily, raise the bar.",
}

# Generic tone alone was not enough: local models still opened every technical
# round at roughly the same level. These contracts make the first question
# visibly different for each area and difficulty.
_AREA_DIFFICULTY_BLOCKS = {
    "behavioral": {
        "warmup": "OPENING CALIBRATION: ask for one familiar, low-pressure STAR story, such as a project they are proud of. Do not introduce conflict or failure yet.",
        "standard": "OPENING CALIBRATION: ask for a concrete ownership or collaboration story with a clear result.",
        "senior": "OPENING CALIBRATION: ask for a high-stakes leadership story involving ambiguous scope, cross-functional influence, or a consequential technical decision. Require the candidate's personal judgment and measurable impact.",
    },
    "dsa": {
        "warmup": "OPENING CALIBRATION: use an elementary array, string, or hash-map problem with a tiny example. Do not use a dynamic-programming, graph, or optimal-subarray problem.",
        "standard": "OPENING CALIBRATION: use a medium-level problem where the candidate must choose and justify a data structure; include constraints that distinguish a brute-force approach from a better one.",
        "senior": "OPENING CALIBRATION: use a genuinely hard problem involving competing constraints, such as streaming data, intervals, graphs, concurrency, or an algorithmic trade-off. State the constraints that rule out the naive solution. Never call the problem simple.",
    },
    "ml": {
        "warmup": "OPENING CALIBRATION: give a small concrete ML scenario and ask first how to frame the prediction target and success metric. Do not open by asking about a generic software or data-service project.",
        "standard": "OPENING CALIBRATION: give a concrete ML scenario and ask for model, feature, and metric reasoning. Use a resume project only when it explicitly involved ML.",
        "senior": "OPENING CALIBRATION: give a production ML scenario with an explicit constraint such as class imbalance, drift, latency, limited labels, or ranking quality. Start with the most consequential modelling/evaluation trade-off. Never re-label a generic data-service project as ML experience.",
    },
    "system_design": {
        "warmup": "OPENING CALIBRATION: use a bounded, familiar service at modest scale, for example tens of thousands of daily users. Give the purpose and scale, then ask one requirements question.",
        "standard": "OPENING CALIBRATION: use a realistic service with clear read/write scale and one likely bottleneck. Give the purpose and scale, then ask one requirements question.",
        "senior": "OPENING CALIBRATION: use a high-scale, failure-sensitive system with explicit throughput, latency, and consistency constraints. Start by asking which requirement or trade-off they would clarify first; do not repeat that question in another form.",
    },
}


def _difficulty_block(difficulty: str) -> str:
    return _DIFFICULTY_BLOCKS.get((difficulty or "").strip(), _DIFFICULTY_BLOCKS["standard"])


def _area_difficulty_block(interview_type: str, difficulty: str) -> str:
    area = (interview_type or "behavioral").strip()
    tier = (difficulty or "standard").strip()
    levels = _AREA_DIFFICULTY_BLOCKS.get(area, _AREA_DIFFICULTY_BLOCKS["behavioral"])
    return levels.get(tier, levels["standard"])


# ---- Coding round: a specific problem + a shared code editor ------------------
# When the candidate picked (or pasted) a problem, we hand the interviewer THAT
# problem plus a private brief, and tell it the candidate's editor is visible.
_FMT_GUIDANCE = {
    "solve": "This is a SOLVE task. Get them to explain their approach before or while coding, "
    "then push on time AND space complexity and edge cases. If they stall, give a small nudge — "
    "never just hand them the solution.",
    "debug": "This is a DEBUG task — the starter code contains a planted bug. Let them find it by "
    "reasoning about the code (trace a concrete input, check the boundaries) rather than pointing "
    "at the line. Once it's fixed, ask WHY it broke and how they'd prevent it.",
    "design": "This is a DESIGN task — the scratchpad is for notes and boxes, not runnable code. "
    "Drive the usual arc: clarify requirements and scale first, then a high-level design, then "
    "drill into one bottleneck or trade-off.",
}

_PROBLEM_BLOCK = """--- CODING ROUND (hands-on) ---
The candidate works in a shared code editor you can BOTH see; its current contents are included \
with each of their turns (as "CURRENT EDITOR CONTENTS"). React to the ACTUAL code they've \
written — their approach, its complexity, bugs, and missing edge cases — like an interviewer \
reading over their shoulder. Still voice-first: they talk it through; never ask them to run it \
or to type faster. Work THIS problem — do not invent a different one:

PROBLEM — {title} ({difficulty}):
{prompt}

{fmt_guidance}{brief}"""


def _problem_block(problem) -> str:
    if problem is None:
        return ""
    brief = ""
    if getattr(problem, "interviewer_brief", "").strip():
        brief = (
            "\n\nINTERVIEWER NOTES (the intended solution — don't VOLUNTEER it or read it out "
            "unprompted; use it to guide, hint, and catch mistakes. BUT if the candidate gives "
            "up and asks you to explain the answer, use these notes to teach them the solution "
            "clearly and simply):\n" + problem.interviewer_brief
        )
    return "\n" + _PROBLEM_BLOCK.format(
        title=problem.title,
        difficulty=problem.difficulty,
        prompt=problem.prompt,
        fmt_guidance=_FMT_GUIDANCE.get(problem.fmt, _FMT_GUIDANCE["solve"]),
        brief=brief,
    )


def coding_block(problem) -> str:
    """The coding context (problem + private brief) as a standalone block, so the
    session can inject the CURRENT problem into the system prompt each turn. That's
    what lets a mid-interview problem switch take effect without rebuilding the
    whole prompt — the block just reflects whatever problem is now on screen."""
    return _problem_block(problem)


INTERVIEWER_SYSTEM = """{shared_rules}

{persona}
{signal_block}
{difficulty_block}
{area_difficulty_block}
{problem_block}

Use the job description and resume below only to choose relevant TOPICS — not to
manufacture details about the candidate's experience.

{opening}

--- JOB DESCRIPTION ---
{jd}

--- CANDIDATE RESUME ---
{resume}
"""


def build_system_prompt(
    jd: str,
    resume: str,
    interview_type: str = "behavioral",
    company: str = "generic",
    difficulty: str = "standard",
    problem=None,
) -> str:
    jd = (jd or "").strip() or "(none provided)"
    resume = (resume or "").strip() or "(none provided)"
    pack = get_pack(interview_type)
    profile = get_company(company)
    # The company overlay is omitted entirely for generic/unknown — an empty
    # signal_block would otherwise leave a stray blank paragraph in the prompt.
    signal_block = f"\n{profile.signal_block}" if profile.signal_block.strip() else ""
    return INTERVIEWER_SYSTEM.format(
        shared_rules=SHARED_SPOKEN_RULES,
        persona=pack.persona,
        signal_block=signal_block,
        difficulty_block=_difficulty_block(difficulty),
        area_difficulty_block=_area_difficulty_block(interview_type, difficulty),
        problem_block=_problem_block(problem),
        opening=pack.opening,
        jd=jd,
        resume=resume,
    )


# ---- The director brain (the agentic decide-loop, director.py) ---------------
# A SECOND prompting discipline, distinct from the spoken persona above: this one
# never speaks. It reads the interview so far and picks structured ACTIONS. Its
# output is schema-constrained JSON, so the prompt's job is good judgment, not
# good formatting. The action schema (director.py ACTION_SCHEMA) is FLAT and
# identical across every topic — only the judgment guidance text below changes,
# via the topic pack's `director_guidance`.
DIRECTOR_SYSTEM = """You are the silent DIRECTOR of a mock interview. You never \
speak to the candidate. Each time you are consulted you choose exactly ONE action, as JSON:

- "note_evidence": record what the candidate's last answer actually proved or lacked
  (skills shown, results quantified, gaps). detail = the note itself, one sentence.
- "note_red_flag": record a genuine concern (contradiction, evasion, blame-shifting, a
  wrong answer, or an inability to explain their own reasoning). detail = the concern,
  one sentence. Use sparingly — only for real signals.
- "probe_deeper": the last answer deserves ONE sharp follow-up. detail = what exactly
  to probe and why (e.g. "no numbers given for the migration's impact — ask for them").
- "switch_topic": the current thread is exhausted. detail = the next topic and the
  angle to open with, chosen from the interview plan / resume / job description.
- "end_interview": enough ground covered (4-6 topics) or the candidate asked to stop.
  detail = one line on why it's time.

{director_guidance}

Judgment rules:
- Usually: ONE note about the last answer, then a move (probe / switch / end).
- Don't probe the same thread more than twice — switch instead.
- Base every note and every move on what the candidate ACTUALLY said. Never invent.
- If the last answer was empty or off-topic, probe_deeper with detail "didn't catch a
  real answer — ask them to repeat or clarify".

Your notes so far and the topics already covered are listed below. Do not repeat notes
you have already made.

{state}"""


def build_director_prompt(
    state_text: str, interview_type: str = "behavioral", company: str = "generic"
) -> str:
    # `company` is accepted for signature symmetry with build_system_prompt and
    # so DirectorState.company can be threaded through uniformly — the director
    # judges domain correctness, not house style, so it isn't injected today.
    pack = get_pack(interview_type)
    return DIRECTOR_SYSTEM.format(director_guidance=pack.director_guidance, state=state_text)


# How each terminal action becomes an instruction for the SPEAK call. Appended as
# a trailing system message so the spoken persona (INTERVIEWER_SYSTEM) still
# governs tone and brevity — the director only supplies the direction.
DIRECTIVE_INSTRUCTIONS = {
    "probe_deeper": 'Direction for this turn: probe deeper — {detail}. Ask ONE follow-up question about exactly that.',
    "switch_topic": 'Direction for this turn: move to a new topic — {detail}. Briefly acknowledge their last answer, then ask ONE opening question on it.',
    "end_interview": 'Direction for this turn: wrap up the interview now ({detail}). Thank the candidate warmly in one or two sentences and close. Do NOT ask another question.',
}


# --- "Learn this" side panel: a written tutor, OUTSIDE the interview ----------
# When the candidate steps aside to understand a topic, this is a text-only
# teacher — no spoken persona, no TTS, no director. It reads the recent interview
# turns purely as context for WHAT to explain, then teaches in Markdown (headings,
# code fences, Big-O) the way a good reference article would. Its whole reason to
# exist is so the candidate never has to leave the app for ChatGPT mid-practice.
TUTOR_SYSTEM = """You are an expert, patient teacher embedded in an interview-practice app. \
The candidate has PAUSED their mock interview to understand a topic more deeply, and is READING \
your answer — you are not speaking, and you are not the interviewer.

How to teach:
- Write for the eye, not the ear. Use Markdown freely: `##` headings, **bold**, bullet and \
numbered lists, and fenced ```code``` blocks for any code or pseudocode.
- Build intuition first (the WHY and the mental model), then the precise mechanics.
- For anything algorithmic, always state time AND space complexity as Big-O, and name the \
approach (e.g. "sliding window", "two-pointer").
- Prefer one concrete worked example over three vague paragraphs.
- Be thorough but skimmable. The interview clock is paused, so depth is welcome — but no filler, \
no "great question", no restating the question back. Just teach.
- Ground every answer in the specific thing the candidate was just asked (given as BACKGROUND). \
Teach the concept behind it, not their exact answer — this is learning, not grading.
- If they haven't asked a specific question yet, explain the core concept behind the CURRENT \
topic end to end: what it is, the standard approach step by step, its complexity, common \
pitfalls, and a short worked example."""

# The synthetic first turn when the panel is opened without a typed question —
# "just explain what we're on right now".
_LEARN_AUTO = (
    "Explain the concept behind what we're currently discussing — what it is, the standard "
    "approach step by step, the time and space complexity, common pitfalls, and a short worked "
    "example. Teach it clearly."
)


def build_learn_messages(payload: dict) -> list[dict]:
    """Turn a /api/learn request into an LLM message list for the tutor.

    payload = {
      interview_type: str,
      problem: {title, prompt} | None,      # the hands-on problem, if any
      context: [{role, content}, ...],      # recent interview transcript (read-only)
      thread:  [{role, content}, ...],      # the panel's own prior Q&A (memory)
      question: str,                        # the new question ("" => auto-explain)
    }

    The interview transcript goes into the SYSTEM prompt as BACKGROUND (it is
    reference material, not part of the tutoring dialogue), while the panel's own
    thread is passed as real user/assistant turns so follow-ups have memory.
    """
    interview_type = (payload.get("interview_type") or "").strip() or "a technical"
    problem = payload.get("problem") or None
    context = payload.get("context") or []
    thread = payload.get("thread") or []
    question = (payload.get("question") or "").strip()

    bg = [
        f"The candidate is in a {interview_type} mock interview and has stepped aside to learn."
    ]
    if problem and (problem.get("title") or problem.get("prompt")):
        bg.append(
            "CURRENT PROBLEM — "
            + (problem.get("title") or "(untitled)")
            + ":\n"
            + (problem.get("prompt") or "").strip()
        )
    turns = [
        f"{'Interviewer' if m.get('role') == 'assistant' else 'Candidate'}: {m.get('content','').strip()}"
        for m in context[-8:]
        if m.get("content")
    ]
    if turns:
        bg.append(
            "RECENT INTERVIEW CONVERSATION (context for what they want to understand — "
            "teach the underlying concept, do not grade their answer):\n" + "\n".join(turns)
        )
    system = TUTOR_SYSTEM + "\n\n--- BACKGROUND ---\n" + "\n\n".join(bg)

    messages = [{"role": "system", "content": system}]
    for m in thread:
        if m.get("role") in ("user", "assistant") and m.get("content"):
            messages.append({"role": m["role"], "content": m["content"]})
    if question:
        messages.append({"role": "user", "content": question})
    elif not thread:
        messages.append({"role": "user", "content": _LEARN_AUTO})
    return messages
