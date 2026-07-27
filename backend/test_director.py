"""Tests for the director's deterministic guardrails (director.decide).

Pins the two root-cause fixes for "it re-asks the same question forever and
ignores 'move on'":

  - (D) a candidate meta-request ("move on", "let's stop") forces a terminal
        move BEFORE the model is consulted, so the LLM can't overrule it.
  - (C) a hard loop-breaker rewrites a probe_deeper into a switch_topic once the
        current thread has been probed DIRECTOR_MAX_PROBES times.

Plus: the director now sees the compaction summary (a system message the old
transcript filter dropped).

Run: python test_director.py
"""

import asyncio
import json

import config
from director import DirectorState, decide


class FakeLLM:
    """Returns scripted raw JSON strings, records the messages it was called
    with, and counts calls — so a test can assert the model was never consulted."""

    def __init__(self, scripted: list[str]):
        self._scripted = list(scripted)
        self.calls = 0
        self.last_messages: list[dict] | None = None

    async def reply(self, messages, max_tokens=None, format_schema=None, reasoning_effort=None):
        self.calls += 1
        self.last_messages = messages
        return self._scripted.pop(0) if self._scripted else "{}"


def _act(action: str, detail: str = "x") -> str:
    return json.dumps({"action": action, "detail": detail})


def _collect(llm, history, state) -> list[dict]:
    async def run():
        return [a async for a in decide(llm, history, state)]
    return asyncio.run(run())


def _history(user_text: str) -> list[dict]:
    return [
        {"role": "system", "content": "PERSONA"},
        {"role": "assistant", "content": "Tell me about a project."},
        {"role": "user", "content": user_text},
    ]


# ---- (D) meta-requests bypass the model entirely -----------------------------

def test_move_on_forces_switch_without_calling_model():
    llm = FakeLLM([_act("probe_deeper")])  # would loop if consulted
    state = DirectorState()
    actions = _collect(llm, _history("Okay, can we move on to the next question?"), state)
    assert llm.calls == 0, "model must not be consulted on a meta-request"
    assert [a["action"] for a in actions] == ["switch_topic"], actions


def test_stop_request_forces_end_without_calling_model():
    llm = FakeLLM([_act("probe_deeper")])
    state = DirectorState()
    actions = _collect(llm, _history("Can we stop the interview?"), state)
    assert llm.calls == 0
    assert actions[-1]["action"] == "end_interview"
    assert state.ended is True


def test_pause_to_think_never_ends_the_interview():
    """Regression: "can we stop for a second and think" contains "can we stop" and
    ended the whole session. A thinking pause must yield a wait, not an end."""
    import director as director_mod

    for phrase in [
        "can we stop for a second and think about this",
        "wait a moment, let me think",
        "give me a second",
        "I need a minute here",
    ]:
        act = director_mod._meta_request_action(phrase)
        assert act is not None and act["action"] == "wait", (phrase, act)
    # while explicit endings still end
    for phrase in ["can we stop here for today", "let's stop the interview", "I'm done"]:
        act = director_mod._meta_request_action(phrase)
        assert act is not None and act["action"] == "end_interview", (phrase, act)


def test_give_up_forces_teaching_not_stalling():
    """Regression: "please just explain the solution" got one more "what have you
    tried?" stall. An explicit give-up now forces a teach directive."""
    import director as director_mod

    for phrase in [
        "I give up, what's the solution?",
        "please just explain the solution to me so I can learn it",
        "can you just tell me the answer",
    ]:
        act = director_mod._meta_request_action(phrase)
        assert act is not None and act["action"] == "teach", (phrase, act)
    # describing YOUR OWN solution must not trigger it
    for phrase in ["let me explain the solution I have in mind",
                   "I'll walk you through my solution now"]:
        act = director_mod._meta_request_action(phrase)
        assert act is None or act["action"] != "teach", (phrase, act)


def test_flow_inquiry_is_not_a_move_on_command():
    """Regression: "are we moving to the next question?" contains "next question"
    and force-switched the topic instead of being answered."""
    import director as director_mod

    for phrase in [
        "are we moving to the next question after this or staying on this one?",
        "is this the last question of the round?",
    ]:
        act = director_mod._meta_request_action(phrase)
        assert act is None, (phrase, act)
    # but an actual request with the same words still switches
    act = director_mod._meta_request_action("can we move on to the next question please")
    assert act is not None and act["action"] == "switch_topic"


def test_mid_answer_done_phrases_do_not_end():
    """Regression: "I'm done with this part" and "that's all for my approach" are
    answer phrases, not requests to leave — they ended the interview."""
    import director as director_mod

    for phrase in [
        "I'm done with this part, moving to the complexity",
        "that's all for my approach, should I code it now?",
        "once the loop is done we return the result",
    ]:
        assert director_mod._meta_request_action(phrase) is None, phrase


def test_ordinary_answer_is_not_a_meta_request():
    llm = FakeLLM([_act("probe_deeper")])
    state = DirectorState()
    actions = _collect(llm, _history("We moved the service to a new region."), state)
    # "moved"/"new" must NOT trip the move-on/stop matchers.
    assert llm.calls == 1
    assert actions[-1]["action"] == "probe_deeper"


def test_go_back_request_returns_to_earlier_problem():
    for phrase in [
        "Can you go back to the previous problem?",
        "I want that question only.",
        "Give me the earlier problem about consecutive numbers.",
    ]:
        llm = FakeLLM([_act("switch_topic", "some brand new topic")])
        state = DirectorState()
        actions = _collect(llm, _history(phrase), state)
        assert llm.calls == 0, phrase
        assert actions[-1]["action"] == "switch_topic"
        assert "earlier" in actions[-1]["detail"].lower() or "return" in actions[-1]["detail"].lower()


def test_stay_request_keeps_the_current_problem():
    for phrase in ["Listen to me and stay on this one.", "Don't move on, keep going."]:
        llm = FakeLLM([_act("switch_topic")])  # model wants to leave; the request overrides
        state = DirectorState()
        actions = _collect(llm, _history(phrase), state)
        assert llm.calls == 0, phrase
        assert actions[-1]["action"] == "probe_deeper", phrase


# ---- (C) hard loop-breaker ---------------------------------------------------

def test_probe_is_rewritten_to_switch_at_the_cap():
    llm = FakeLLM([_act("probe_deeper", "ask again")])
    state = DirectorState(times_probed=config.DIRECTOR_MAX_PROBES)
    actions = _collect(llm, _history("A vague non-answer."), state)
    assert actions[-1]["action"] == "switch_topic", actions
    assert state.times_probed == 0, "switching must reset the probe counter"


def test_probe_passes_through_under_the_cap():
    llm = FakeLLM([_act("probe_deeper")])
    state = DirectorState(times_probed=config.DIRECTOR_MAX_PROBES - 1)
    actions = _collect(llm, _history("A vague non-answer."), state)
    assert actions[-1]["action"] == "probe_deeper"
    assert state.times_probed == config.DIRECTOR_MAX_PROBES


def test_repeated_probes_hit_the_cap_and_switch():
    state = DirectorState()
    # Feed a fresh probe each turn; after MAX_PROBES the next one is forced over.
    for _ in range(config.DIRECTOR_MAX_PROBES):
        out = _collect(FakeLLM([_act("probe_deeper")]), _history("still vague"), state)
        assert out[-1]["action"] == "probe_deeper"
    out = _collect(FakeLLM([_act("probe_deeper")]), _history("still vague"), state)
    assert out[-1]["action"] == "switch_topic"


# ---- director sees the compaction summary ------------------------------------

def test_director_prompt_includes_compaction_summary():
    llm = FakeLLM([_act("probe_deeper")])
    history = [
        {"role": "system", "content": "PERSONA"},
        {"role": "system", "content": "Summary of the interview so far: candidate led a payments migration."},
        {"role": "assistant", "content": "And after that?"},
        {"role": "user", "content": "We shipped it."},
    ]
    _collect(llm, history, DirectorState())
    user_msg = llm.last_messages[-1]["content"]
    assert "EARLIER (summary)" in user_msg, user_msg
    assert "payments migration" in user_msg


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
    print(f"\n{passed}/{len(tests)} passed")
    print(f"(DIRECTOR_MAX_PROBES={config.DIRECTOR_MAX_PROBES})")
