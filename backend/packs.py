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

import random
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
    # A short list of this domain's jargon, fed to Whisper as an initial_prompt so
    # it transcribes technical terms correctly ("priority queue", not "priority
    # cube"; "hash map", not "hatch map"). Empty falls back to config's generic hint.
    stt_hint: str = ""


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
    stt_hint="A data structures and algorithms interview. Likely terms: array, string, "
    "hash map, hash set, priority queue, heap, stack, queue, linked list, binary tree, "
    "binary search, two pointers, sliding window, dynamic programming, recursion, "
    "backtracking, graph, BFS, DFS, time complexity, space complexity, big O.",
    persona="""You are running a DATA STRUCTURES & ALGORITHMS round — SPOKEN ONLY, there is \
no editor and the candidate will never write or run code. Pose ONE concrete problem at a time \
(start simple — array/string/hash-map level). State the problem clearly and ALWAYS ground it \
with a tiny worked example: one sample input and its expected output, so there is zero \
ambiguity about what's being asked. Write examples using ACTUAL DIGITS AND SIGNS, not spelled-out \
words — say "the array 2, 1, -3, 4 gives output 4, 4, 4, -1", NOT "two, one, negative three". \
The candidate READS these in the live transcript, so numeric form is what they need; a request \
for "the example in numbers" or "in mathematical form" is easily satisfied — just restate it with \
digits (e.g. -3, not "negative three"). Never refuse with "it's a voice interview so I can't \
write" — you are only stating numbers aloud, which is fine; what you avoid is asking THEM to write \
or draw. Invite the candidate to think out loud, state assumptions, \
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
    stt_hint="A machine learning interview. Likely terms: model, dataset, features, "
    "training, fine-tuning, LoRA, gradient descent, overfitting, regularization, "
    "precision, recall, F1 score, ROC AUC, cross-validation, embeddings, transformer, "
    "neural network, hyperparameters, class imbalance, data drift, inference.",
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
    stt_hint="A system design interview. Likely terms: load balancer, cache, Redis, "
    "database, SQL, NoSQL, sharding, replication, consistency, availability, latency, "
    "throughput, message queue, Kafka, API, microservices, CDN, rate limiting, "
    "horizontal scaling, partitioning, indexing.",
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

# Deterministic first questions establish the promised topic and level. The
# LLM takes over after the candidate answers, so later turns remain adaptive.
# Each (area, tier) is a POOL of calibrated openers, not one fixed line — a real
# interview never opens the same way twice. opening_question() picks one at random
# and avoids repeating the one it served last for that slot (see _last_opening).
# Every variant must stay on-tier (a warmup opener must not sneak in a hard graph
# problem) and end with "?" — test_packs pins both.
OPENING_QUESTIONS: dict[str, dict[str, list[str]]] = {
    "behavioral": {
        "warmup": [
            "Hi, thanks for joining. To start us off, could you tell me about a project or piece of work you are especially proud of?",
            "Hi, good to have you. To warm up, walk me through something you built or shipped recently that you genuinely enjoyed working on?",
            "Thanks for making the time. Let’s start easy — what’s a piece of work from the last year that you’d happily show someone?",
            "Hi there. To get us going, tell me about a project where you feel you did your best work, and what your part in it was?",
        ],
        "standard": [
            "Hi, thanks for joining. Tell me about a time you took ownership of an important problem and what changed because of your work?",
            "Thanks for joining. Describe a situation where you drove a project end to end — what outcome were you accountable for, and how did it land?",
            "Good to meet you. Tell me about a time you noticed something was broken or missing and took it on yourself to fix it — what happened?",
            "Hi. Walk me through a time you disagreed with a decision your team was making, and how you handled it — what was the result?",
        ],
        "senior": [
            "Hi, thanks for joining. Tell me about a high-stakes decision you led through ambiguity, including how you aligned people and measured the outcome?",
            "Thanks for joining. Tell me about a time you had to set direction with incomplete information — how did you get others behind it, and what was the impact?",
            "Good to have you. Describe the most consequential technical or organizational call you’ve made — how did you weigh the trade-offs, and how did it turn out?",
            "Hi. Tell me about a time you led through a serious setback or a conflict across teams — what did you do, and how did you know it worked?",
        ],
    },
    "dsa": {
        "warmup": [
            "Let’s start with a small problem. Given a list of integers, how would you tell whether any value appears more than once, and what time and space would your approach use?",
            "Let’s warm up. Given a string, how would you check whether it reads the same forwards and backwards, and what does your approach cost?",
            "Small one to start. Given an array of numbers, how would you return them with the duplicates removed, and what time and space does that take?",
            "Let’s ease in. Given two short strings, how would you decide whether one is a rearrangement of the other, and at what cost?",
        ],
        "standard": [
            "Let’s work through a medium problem. Given a stream of user IDs, return the first ID that has appeared exactly once so far; how would you avoid re-scanning the whole stream after every event?",
            "Here’s a medium one. Given an array and a target, return the indices of the two values that sum to it — can you beat the brute-force pair scan, and how?",
            "Let’s try this. You’re given a list of meetings with start and end times; how would you find the largest number that overlap at once, and what structure would you reach for?",
            "Medium problem. Given a string, how would you find the length of the longest substring with no repeating characters, and what makes your approach better than checking every substring?",
        ],
        "senior": [
            "Let’s work through a streaming problem. Events can arrive up to 30 seconds out of order, and you must count each user’s events in the previous five minutes; what data structures would you use, and which accuracy-versus-memory trade-off would you make?",
            "Here’s a harder one. You have a huge log of key events that doesn’t fit in memory; how would you return the top-k most frequent keys, and where would you accept approximation?",
            "Let’s go deep. Given a directed graph of build targets with dependencies, how would you produce a valid build order and detect a cycle, and what’s the cost at scale?",
            "Tough problem. You have many sorted streams of numbers arriving at once; how would you merge them into a single sorted output with bounded memory, and where’s your bottleneck?",
        ],
    },
    "ml": {
        "warmup": [
            "Let’s consider a subscription app that wants to predict which trial users will convert this week. What would you define as the prediction target, and how would you decide whether the model is useful?",
            "Let’s warm up with a scenario. A news app wants to flag articles that are likely spam. How would you frame the target, and how would you know the model is doing its job?",
            "Consider a photo app that wants to auto-tag pictures containing a pet. What exactly would you predict, and what would ‘good enough’ look like to you?",
            "Say a store wants to predict which customers will come back next month. How would you define the label, and what metric would tell you the model is worth shipping?",
        ],
        "standard": [
            "Consider a fraud classifier where only one transaction in a thousand is fraudulent. What features and evaluation metric would you start with, and why would accuracy be misleading here?",
            "Consider a model that flags support tickets as urgent, where most tickets aren’t. What features and metric would you start with, and why not accuracy?",
            "You’re building a churn model for a subscription product. Which features would you reach for first, how would you evaluate it, and what leakage would you watch out for?",
            "Consider recommending products a user might buy next. How would you frame the problem, what would you train on, and how would you measure success before shipping?",
        ],
        "senior": [
            "Consider a ranking model whose click-through rate is stable but whose conversion rate has fallen after a product change. How would you distinguish data drift from a product effect, and which offline and online metrics would drive your decision?",
            "A production classifier’s precision has quietly dropped over three months while volume grew. How would you separate data drift from a labelling or product change, and what would you monitor?",
            "You must ship a model under a strict latency budget with limited labelled data. Which trade-offs — model size, features, active learning — would you make first, and how would you validate them?",
            "A recommender looks strong offline but engagement fell after launch. How would you reconcile the offline and online metrics, and what experiment would you run to find the gap?",
        ],
    },
    "system_design": {
        "warmup": [
            "Let’s design a URL shortener for about ten thousand daily users. Before proposing components, which functional requirement would you clarify first?",
            "Let’s design a pastebin-style service for a small team. Before we sketch any components, which requirement would you pin down first?",
            "Let’s design a simple image upload-and-share service at modest scale. What functional requirement would you clarify before choosing an architecture?",
            "Let’s design a reminders service for a few thousand users. What’s the first requirement you’d want nailed down before designing anything?",
        ],
        "standard": [
            "Let’s design a notification service handling ten million sends per day across email and push. Which delivery guarantee or failure mode would you clarify before choosing the architecture?",
            "Let’s design a rate limiter for a public API serving millions of requests a day. Which behavior or failure mode would you clarify before picking an approach?",
            "Let’s design a news feed for a social app with heavy reads. Which read/write pattern or guarantee would you clarify before choosing the architecture?",
            "Let’s design a file-storage service with sharing and access control at moderate scale. Which requirement or bottleneck would you pin down first?",
        ],
        "senior": [
            "Let’s design a real-time trading-data cache serving millions of reads per second with sub-20-millisecond latency. Which consistency or failure-recovery requirement would you clarify first?",
            "Let’s design a global rate limiter that stays correct across regions at very high throughput. Which consistency or failure-recovery requirement would you settle first?",
            "Let’s design an ad-click ingestion pipeline handling hundreds of thousands of events per second with strong dedup guarantees. Which trade-off would you clarify before we start?",
            "Let’s design a multi-region key-value store with low-latency reads and tight durability needs. Which consistency-versus-availability requirement would you clarify first?",
        ],
    },
}

# Remembers the last opener served per (area, tier) so the next interview in the
# same slot doesn't repeat it. In-memory only — resets on restart, which is fine.
_last_opening: dict[tuple[str, str], str] = {}


def get_pack(key: str | None) -> InterviewPack:
    """Fall back to behavioral for an unknown or empty key — never raises."""
    return PACKS.get((key or "").strip(), _BEHAVIORAL)


def opening_question(interview_type: str | None, difficulty: str | None) -> str:
    """Return a calibrated first spoken question, chosen at random from the slot's
    pool so a new interview doesn't open with the same line every time. Avoids
    immediately repeating the previous opener for that same area+tier."""
    area = (interview_type or "behavioral").strip()
    tier = (difficulty or "standard").strip()
    by_tier = OPENING_QUESTIONS.get(area, OPENING_QUESTIONS["behavioral"])
    variants = by_tier.get(tier) or by_tier["standard"]
    if isinstance(variants, str):  # tolerate a legacy single-string slot
        return variants
    key = (area, tier)
    choices = [q for q in variants if q != _last_opening.get(key)] or variants
    pick = random.choice(choices)
    _last_opening[key] = pick
    return pick
