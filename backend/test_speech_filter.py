"""Tests for the spoken-output guard (speech_filter.spoken_only).

Each case is a bug we actually saw in a live interview, pinned so it can't
come back. Run: python test_speech_filter.py
"""

import asyncio

from speech_filter import spoken_only


async def _drain(tokens: list[str], one_question: bool = True) -> str:
    async def gen():
        for t in tokens:
            yield t

    return "".join([c async for c in spoken_only(gen(), one_question)])


def _run(tokens, one_question=True):
    return asyncio.run(_drain(tokens, one_question))


def test_strips_the_leaked_stage_direction():
    # The exact leak seen live, streamed token-by-token.
    out = _run(["What ", "makes ", "it ", "stand ", "out", "?", " [Wait ", "for ", "response", "]"])
    assert out == "What makes it stand out?", out


def test_stops_after_first_question():
    out = _run(["Tell me about a project", "? ", "What did you do", "? ", "What did you learn", "?"])
    assert out == "Tell me about a project?", out


def test_closing_turn_with_no_question_streams_in_full():
    out = _run(["Thanks for your time, Ritika. ", "I appreciate it."])
    assert out == "Thanks for your time, Ritika. I appreciate it.", out


def test_bracket_spanning_multiple_tokens_is_removed():
    out = _run(["Got it. ", "[cut", " off by the ", "candidate", "]", " Why"], one_question=False)
    assert out == "Got it.  Why", out


def test_bracket_in_the_middle_keeps_surrounding_speech():
    out = _run(["Say ", "[note]", "this"], one_question=False)
    assert out == "Say this", out


def test_question_inside_a_bracket_does_not_end_the_turn():
    out = _run(["Right. ", "[Ask more?]", " Tell me why."], one_question=False)
    assert out == "Right.  Tell me why.", out


def test_unclosed_bracket_swallows_the_rest():
    # Safe failure: a short reply beats speaking a stage direction aloud.
    out = _run(["Okay. ", "[Direction: probe deeper and keep going"], one_question=False)
    assert out == "Okay. ", out


def test_one_question_off_keeps_every_question():
    out = _run(["A", "? ", "B", "?"], one_question=False)
    assert out == "A? B?", out


def test_nested_brackets():
    out = _run(["Hi ", "[a [b] c]", "there"], one_question=False)
    assert out == "Hi there", out


def test_data_bracket_with_numbers_is_kept():
    # DSA example arrays must survive — this was gutted before.
    out = _run(["For ", "[3, 5, 9, 1]", " the max is 9."], one_question=False)
    assert out == "For [3, 5, 9, 1] the max is 9.", out


def test_data_bracket_streamed_across_tokens_is_kept():
    out = _run(["an array like ", "[2,", " 7, ", "11]", "?"])
    assert out == "an array like [2, 7, 11]?", out


def test_bracket_with_comma_but_no_digit_is_kept():
    out = _run(["indices ", "[i, j]", " work"], one_question=False)
    assert out == "indices [i, j] work", out


def test_single_number_bracket_is_kept():
    out = _run(["returns ", "[9]", " here"], one_question=False)
    assert out == "returns [9] here", out


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
