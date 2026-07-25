"""Tests for the problem bank + mid-interview problem switching (problems.py).

Pins the regression that swapped the problem mid-answer: the candidate said
"…and move on to the next element" (an algorithm step) and the switch detector
treated the bare "move on" as a request to change problems. Switch detection must
require an EXPLICIT change intent.

Run: python test_problems.py
"""

import ast

import problems

_CUR = problems.get_problem("merge-k-lists")  # a hard problem, like the live case

# Utterances that mention "move on"/"problem" but are NOT switch requests.
NOT_SWITCH = [
    "we are not going to put it inside our heap and move on to the next element",
    "we can iterate and move on to the next node in the list",
    "I would use a hash map here and move on",
    "can you repeat the problem",
    "give me a hint about the problem",
    "let me re-read the problem",
    "this is an easy problem honestly",
    "I don't fully understand the problem",
]

# Utterances that SHOULD switch.
SWITCH = [
    "can we do a medium graph problem instead",
    "give me an easier one",
    "something harder please",
    "next problem",
    "move to a medium question",
    "can we move on to the next question",
    "let us move on",
    "switch the problem",
    "skip this one",
    "give me another problem",
    "I want a harder problem",
]


def test_algorithm_move_on_does_not_switch():
    for t in NOT_SWITCH:
        assert problems.match_switch(t, _CUR) is None, f"false switch on: {t!r}"


def test_explicit_requests_switch():
    for t in SWITCH:
        assert problems.match_switch(t, _CUR) is not None, f"missed switch: {t!r}"


def test_switch_is_never_the_same_problem():
    for _ in range(20):
        nxt = problems.match_switch("next problem", _CUR)
        assert nxt is not None and nxt.id != _CUR.id


def test_difficulty_hint_is_respected():
    easy = problems.match_switch("give me an easy problem", _CUR)
    hard = problems.match_switch("give me a hard problem", _CUR)
    assert easy.difficulty == "easy"
    assert hard.difficulty == "hard"


def test_increase_the_difficulty_switches_and_escalates():
    """Regression: 'increase the difficulty' didn't match, so it fell through to
    the LLM, which improvised a THEORY question instead of a harder problem."""
    med = problems.get_problem("ml-gradient-step")           # medium
    for phrase in ["Can you increase the difficulty of the questions?",
                   "raise the difficulty please", "can we step up the level"]:
        nxt = problems.match_switch(phrase, med)
        assert nxt is not None, f"missed: {phrase!r}"
        assert nxt.difficulty == "hard", f"{phrase!r} -> {nxt.difficulty}"


def test_switching_does_not_repeat_a_shown_problem():
    """Regression: with only 3 ML problems it cycled between the same two."""
    cur = problems.get_problem("ml-gradient-step")
    shown = {cur.id}
    for _ in range(6):
        nxt = problems.match_switch("can we move to the next coding question", cur, exclude=shown)
        assert nxt is not None
        assert nxt.id not in shown, f"handed back an already-shown problem: {nxt.id}"
        shown.add(nxt.id)
        cur = nxt


def test_every_topic_has_real_depth():
    """The cycling happened because ML/design had 3 problems each."""
    for topic, least in (("dsa", 15), ("ml", 8), ("system_design", 6)):
        n = len([p for p in problems.PROBLEMS if p.topic == topic])
        assert n >= least, f"{topic} only has {n} problems"
    # and a hard tier exists where it matters
    for topic in ("dsa", "ml"):
        assert any(p.topic == topic and p.difficulty == "hard" for p in problems.PROBLEMS), topic


def test_all_starter_code_parses_and_has_a_budget():
    for p in problems.PROBLEMS:
        if p.language == "python":
            ast.parse(p.starter_code)
        assert p.budget() > 0
        assert "interviewer_brief" not in p.public()  # private brief never leaks


def test_debug_snippets_are_actually_buggy():
    ns = {}
    exec(problems.get_problem("debug-two-sum").starter_code, ns)
    assert ns["two_sum"]([3, 2, 4], 6) == [0, 0]  # the planted bug
    ns = {}
    exec(problems.get_problem("debug-off-by-one").starter_code, ns)
    assert ns["factorial"](5) == 24  # off by a factor of n


def test_company_favourites_all_exist():
    """A typo'd id in the company map would silently give a company fewer real
    favourites — catch it here."""
    ids = {p.id for p in problems.PROBLEMS}
    for company, favs in problems._COMPANY_FAVORITES.items():
        missing = [i for i in favs if i not in ids]
        assert not missing, f"{company} lists non-existent problems: {missing}"


def test_company_reorders_but_never_hides():
    """Picking a company floats its favourites to the top of the picker without
    dropping any problem — you can still choose the others."""
    plain = problems.list_problems(topic="dsa")
    google = problems.list_problems(topic="dsa", company="google")
    assert len(plain) == len(google)                       # nothing hidden
    assert "google" in google[0].companies                 # a favourite is first
    # generic is neutral — same order as no company at all
    assert [p.id for p in problems.list_problems(topic="dsa", company="generic")] == \
           [p.id for p in plain]


def test_switch_prefers_company_but_explicit_request_wins():
    cur = problems.get_problem("two-sum-sorted")
    picks = [problems.match_switch("give me another problem", cur, company="meta").id
             for _ in range(30)]
    assert all("meta" in problems.get_problem(i).companies for i in picks)
    # an explicit difficulty still overrides the soft company bias
    nxt = problems.match_switch("give me an easy problem", cur, company="google")
    assert nxt.difficulty == "easy"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            print(f"FAIL {t.__name__}: {e}")
        else:
            print(f"ok  {t.__name__}")
            passed += 1
    print(f"\n{passed}/{len(tests)} passed  ({len(problems.PROBLEMS)} problems in bank)")
