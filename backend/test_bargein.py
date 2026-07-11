"""Tests for barge-in detection (main.Session._detect_bargein).

Pins the bug reported live: interrupting the interviewer "takes a lot of time /
a higher volume of voice". Cause was that a single sub-threshold frame reset the
speech run to zero, so only continuous shouting ever reached the required run.

Run: python test_bargein.py
"""

import asyncio

import config
import main
from main import Session

FRAME = b"\x00" * (config.VAD_FRAME_SAMPLES * 2)
LOUD = config.BARGEIN_THRESHOLD + 0.05
QUIET = config.BARGEIN_THRESHOLD - 0.2


def _session():
    """A Session with only the barge-in state wired up — no socket, no models."""
    s = Session.__new__(Session)
    s._bargein_run = 0
    s._bargein_gap = 0
    s._bargein_buf = []
    s.fired = False

    async def trigger():
        s.fired = True

    s._trigger_bargein = trigger
    return s


async def _feed(s, probs):
    for p in probs:
        if s.fired:
            break
        await Session._detect_bargein(s, p, FRAME)
    return s.fired


def _run(probs):
    return asyncio.run(_feed(_session(), probs))


def test_sustained_speech_interrupts():
    assert _run([LOUD] * main._BARGEIN_FRAMES) is True


def test_speech_with_syllable_dips_still_interrupts():
    """The regression: real speech dips between syllables. Before the gap
    tolerance this pattern never fired, no matter how long the user talked."""
    pattern = ([LOUD, LOUD, QUIET] * 10)[: 30]
    assert _run(pattern) is True


def test_a_short_cough_does_not_interrupt():
    assert _run([LOUD, LOUD] + [QUIET] * 40) is False


def test_run_is_abandoned_after_a_long_silence():
    """Speech, then a gap longer than BARGEIN_GAP_MS, then more speech that is
    itself too short — the two bursts must NOT add up into a false interrupt."""
    burst = [LOUD] * (main._BARGEIN_FRAMES - 1)
    long_gap = [QUIET] * (main._BARGEIN_GAP_FRAMES + 2)
    assert _run(burst + long_gap + [LOUD]) is False


def test_silence_alone_never_interrupts():
    assert _run([QUIET] * 100) is False


def test_replay_buffer_is_contiguous_across_a_dip():
    """Buffered frames feed the replayed utterance; dropping the quiet ones would
    clip syllables out of the interruption's first word."""
    s = _session()
    asyncio.run(_feed(s, [LOUD, QUIET, LOUD]))
    assert len(s._bargein_buf) == 3, len(s._bargein_buf)


def test_gap_counter_resets_on_resumed_speech():
    s = _session()
    asyncio.run(_feed(s, [LOUD] + [QUIET] * (main._BARGEIN_GAP_FRAMES - 1) + [LOUD]))
    assert s._bargein_gap == 0 and s._bargein_run == 2


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
    print(f"(frames needed={main._BARGEIN_FRAMES}, gap tolerated={main._BARGEIN_GAP_FRAMES})")
