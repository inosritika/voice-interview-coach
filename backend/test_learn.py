"""Tests for the 'Learn this' tutor context builder (prompts.build_learn_messages).

The tutor reads the interview transcript as BACKGROUND (system prompt) and keeps
its own follow-up thread as real turns, so a follow-up has memory. Run: python test_learn.py
"""

import prompts


def test_auto_explain_when_no_question():
    m = prompts.build_learn_messages({"interview_type": "DSA", "context": [], "thread": [], "question": ""})
    assert m[0]["role"] == "system"
    assert m[-1]["role"] == "user"                     # a synthetic "explain this" turn
    assert "step by step" in m[-1]["content"]


def test_transcript_goes_into_system_not_dialogue():
    ctx = [{"role": "assistant", "content": "what data structure?"},
           {"role": "user", "content": "a hashmap"}]
    m = prompts.build_learn_messages({"interview_type": "DSA", "context": ctx, "thread": [], "question": "why?"})
    assert "hashmap" in m[0]["content"]                # context lives in the system prompt
    assert m[-1]["content"] == "why?"                  # not polluted by transcript
    assert len(m) == 2                                 # system + the one question


def test_followup_thread_is_preserved_as_turns():
    thread = [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"}]
    m = prompts.build_learn_messages({"interview_type": "ml", "thread": thread, "question": "q2"})
    roles = [x["role"] for x in m]
    assert roles == ["system", "user", "assistant", "user"]
    assert m[-1]["content"] == "q2"


def test_problem_is_included_when_present():
    m = prompts.build_learn_messages({
        "interview_type": "DSA",
        "problem": {"title": "Top K Frequent", "prompt": "return the k most frequent elements"},
        "context": [], "thread": [], "question": "",
    })
    assert "Top K Frequent" in m[0]["content"]
    assert "k most frequent" in m[0]["content"]


def test_missing_fields_dont_crash():
    m = prompts.build_learn_messages({})               # empty payload
    assert m[0]["role"] == "system" and m[-1]["role"] == "user"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            print(f"FAIL {t.__name__}: {e}")
        else:
            print(f"ok  {t.__name__}"); passed += 1
    print(f"\n{passed}/{len(tests)} passed")
