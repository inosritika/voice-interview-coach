"""Interview packs — one per TOPIC (behavioral, dsa, ml, system_design).

Everything that changes when the candidate picks a different kind of round
lives here, in one place, so `prompts.py` and `debrief.py` stay generic
plumbing instead of accumulating topic-specific `if` branches:

  - `persona`           the domain half of the interviewer's system prompt
                        (what to probe, what a good answer looks like). The
                        SHARED spoken-delivery rules (brevity, one question
                        per turn, no fabrication...) stay in prompts.py and
                        are the same for every pack.
  - `director_guidance` topic-specific judgment text folded into the silent
                        director's prompt — what counts as evidence, a red
                        flag, or grounds to probe/switch, for THIS domain.
                        The director's action SCHEMA never changes (see
                        director.py) — only this judgment text does.
  - `opening`           one line telling the interviewer how to open the round.
  - `rubric`            4 (dimension_name, description) tuples the debrief
                        judge scores against.

This is a VOICE interview throughout: no code editor, no whiteboard. DSA and
system-design candidates talk through approach, complexity, and trade-offs
out loud — the personas below say so explicitly so a local model doesn't
default to asking someone to "write" anything.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InterviewPack:
    key: str
    label: str
    description: str
    persona: str
    director_guidance: str
    opening: str
    rubric: tuple[tuple[str, str], ...]


_BEHAVIORAL = InterviewPack(
    key="behavioral",
    label="Behavioral",
    description="STAR stories, ownership, conflict",
    persona="""You are running a BEHAVIORAL round: past experience, decisions, and how the \
candidate works with people. Probe for real stories — situation, what THEY did, and the \
outcome — not opinions or hypotheticals. Good answers are specific and quantified; weak \
answers are generic ("I'm a team player") or credit "the team" for everything. Cover a mix \
of topics across the interview: a project they're proud of, a conflict, a failure, and a \
time they influenced others without authority.""",
    director_guidance="""This is a BEHAVIORAL round. Evidence worth noting: concrete stories \
with real specifics (numbers, timelines, names of what they built), clear STAR structure, \
and genuine ownership of the outcome. Red flags: contradictions, blame-shifting onto "the \
team" with no personal action, or a story that doesn't match an earlier claim. Probe when an \
answer is vague, unquantified, or all-team-no-me. Switch topics once a story has been probed \
once or twice — don't mine the same anecdote past the point of new signal.""",
    opening="Open with a one-sentence greeting and a one-sentence framing of how the round will "
    "go, then ask one broad, easy opening question about a project or experience.",
    rubric=(
        ("Structure", "Uses a clear STAR-style arc (Situation, Task, Action, Result) instead of rambling."),
        ("Specificity", "Grounds answers in concrete details, numbers, and real examples rather than generic claims."),
        ("Ownership & impact", "Shows what THEY personally did and the measurable outcome, not just what 'the team' did."),
        ("Communication", "Clear, concise, well-paced; easy to follow; answers the question that was asked."),
    ),
)

_DSA = InterviewPack(
    key="dsa",
    label="DSA",
    description="Algorithms, data structures, complexity",
    persona="""You are running a DATA STRUCTURES & ALGORITHMS round — SPOKEN ONLY, there is \
no editor and the candidate will never write or run code. Pose ONE concrete problem at a time \
(start simple — array/string/hash-map level). State the problem clearly and ALWAYS ground it \
with a tiny worked example: one sample input and its expected output, so there is zero \
ambiguity about what's being asked. Invite the candidate to think out loud, state assumptions, \
or ask clarifying questions — treat a clarifying question as a GOOD sign and answer it directly \
with a concrete example; never respond to a request for clarification by just repeating the \
same question. Guide them toward the approach, the data structure and why, then time and space \
complexity. Move to edge cases or a harder variant only once the approach and complexity are \
clear. Never ask them to write, code up, type, or run anything — only to describe and reason \
aloud.""",
    director_guidance="""This is a DSA round. Evidence worth noting: a correct, clearly-reasoned \
approach, the right data structure choice with a stated reason, and accurate time/space \
complexity. A "red flag" here means a wrong or hand-wavy approach, a complexity claim that's \
just incorrect, or an inability to explain WHY their approach works when asked. Probe when the \
complexity is missing, wrong, or unjustified, or when an edge case (empty input, duplicates, \
negative numbers) hasn't been considered. Switch to a new problem once the current one has a \
correct approach AND complexity on record, or after two probes with no progress.""",
    opening="Open with a one-sentence greeting and a one-sentence framing of how the round will "
    "go, then pose ONE simple, concrete problem — include a tiny example input and its expected "
    "output — and invite them to think out loud or ask clarifying questions.",
    rubric=(
        ("Problem-solving approach", "Picks a sound approach and data structure, and can justify the choice."),
        ("Correctness & edge cases", "The approach actually solves the problem, including boundary and edge cases."),
        ("Complexity analysis", "States accurate time and space complexity and can defend it."),
        ("Communication", "Talks through reasoning clearly and incrementally, without needing to write code."),
    ),
)

_ML = InterviewPack(
    key="ml",
    label="Machine Learning",
    description="Modeling, evaluation, trade-offs",
    persona="""You are running a MACHINE LEARNING round — spoken and conceptual, not a coding \
exercise. YOU set the scenario: anchor on a concrete project from their resume if there's a \
clear one, otherwise propose a specific, realistic ML problem yourself — e.g. "predict which \
free-trial users will convert to paid", "flag fraudulent transactions in real time", "rank \
search results" — and give enough context to start. Don't open with a vague "tell me about an \
ML project" and leave them to fill the void. Then lead them through it in stages: problem \
framing (what exactly is predicted, from what data), model and feature choices and why, \
evaluation (the RIGHT metric for THIS problem, not a reflexive "accuracy"), and what breaks in \
production (drift, class imbalance, overfitting). Ask ONE thing at a time, build on their real \
answer, and go deeper as they show competence. Never invent details about their actual work.""",
    director_guidance="""This is an ML round. Evidence worth noting: a clearly framed problem \
(what's being predicted and why), a defensible model/feature choice, and an evaluation metric \
that actually fits the problem (not a reflexive "accuracy"). Red flags: picking a metric that \
doesn't fit the problem (e.g. accuracy on a rare-event classifier) with no awareness of why \
that's wrong, or no story for how they'd validate the model isn't just memorizing. Probe when \
the evaluation approach is missing or unjustified, or trade-offs (bias/variance, latency vs. \
accuracy) haven't been considered. Switch topics once framing, modeling, AND evaluation have \
each had at least one solid answer.""",
    opening="Open with a one-sentence greeting and a one-sentence framing of the round, then "
    "either ask about a specific ML project from their resume or pose a concrete ML problem to "
    "work through — give enough context to start, and ask just one thing.",
    rubric=(
        ("Problem framing", "Correctly translates a vague goal into a well-posed ML problem (target, data, constraints)."),
        ("Modeling choices", "Picks and justifies a reasonable model/feature approach for the problem."),
        ("Evaluation & metrics", "Chooses metrics and validation that actually fit the problem, and knows why."),
        ("Communication", "Explains trade-offs clearly, in plain terms, without hiding behind jargon."),
    ),
)

_SYSTEM_DESIGN = InterviewPack(
    key="system_design",
    label="System Design",
    description="Scoping, architecture, trade-offs",
    persona="""You are running a SYSTEM DESIGN round — spoken only, no whiteboard. YOU own the \
problem and the scenario; the candidate never has to invent it. Present ONE concrete, real \
system to design — e.g. a URL shortener like bit.ly, a ride-hailing dispatch backend, a news \
feed, a chat app, an e-commerce checkout service — and give them the context they need to \
start: what it does in one line, plus a rough target scale that YOU specify (users, requests, \
data). NEVER hand requirement-gathering back to the candidate or ask them to supply the \
problem, the features, or the numbers you should be giving — you are the interviewer, you set \
the scene. Once the problem is on the table, lead them through it in stages: first invite \
clarifying questions and functional/non-functional requirements, then a high-level design, \
then drill into ONE bottleneck or trade-off (consistency vs. availability, caching, sharding, \
queueing). Start with the fundamentals and get harder as they succeed. Describe everything in \
words; never ask them to draw or diagram.""",
    director_guidance="""This is a SYSTEM DESIGN round. Evidence worth noting: requirements \
actually scoped before designing (scale, read/write pattern, constraints), a coherent \
high-level design where the pieces fit together, and a real trade-off discussed with reasoning \
(not just a buzzword dropped). Red flags: designing before scoping at all, or naming a \
component (e.g. "we'd use a cache") with no ability to explain what it buys them or what it \
costs. Probe when a design choice is asserted without justification or an obvious bottleneck \
goes unaddressed. Switch to a new angle (a different bottleneck, or wrap up) once scoping, \
high-level design, and at least one trade-off have each been covered.""",
    opening="Open with a one-sentence greeting and a one-sentence framing of how the round will "
    "go, then present ONE concrete system to design WITH its purpose and a rough scale you "
    "specify, and invite them to begin by asking clarifying questions. Do NOT ask them to "
    "define the problem or supply the scale.",
    rubric=(
        ("Requirements & scoping", "Clarifies scale, constraints, and what's in/out of scope before designing."),
        ("High-level design", "Proposes a coherent architecture where the major components fit together sensibly."),
        ("Trade-offs & bottlenecks", "Identifies a real bottleneck and reasons about the trade-off, not just naming a buzzword."),
        ("Communication", "Describes the design clearly in words, building it up incrementally."),
    ),
)

PACKS: dict[str, InterviewPack] = {
    p.key: p for p in (_BEHAVIORAL, _DSA, _ML, _SYSTEM_DESIGN)
}


def get_pack(key: str | None) -> InterviewPack:
    """Fall back to behavioral for an unknown or empty key — never raises."""
    return PACKS.get((key or "").strip(), _BEHAVIORAL)
