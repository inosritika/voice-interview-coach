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


# ---- barge-in / re-capture is armed the whole time a turn is in flight --------
# SPEAKING  -> interrupt the audible reply.
# PROCESSING -> the user is continuing their answer; re-capture it instead of
#               dropping it (reported: a sentence spoken during the think phase
#               never made it into the chat).

class _ActiveTask:
    def done(self):
        return False


def _frame_session(floor):
    s = Session.__new__(Session)
    s._turn_task = _ActiveTask()  # a turn is in flight
    s.floor = floor
    s.detected = False

    class _VAD:
        def speech_prob(self, samples):
            return LOUD  # unmistakable speech every frame

    s.vad = _VAD()

    async def spy(prob, frame):
        s.detected = True

    s._detect_bargein = spy
    return s


def test_recapture_while_processing():
    s = _frame_session(main.Floor.PROCESSING)
    asyncio.run(Session._process_frame(s, FRAME))
    assert s.detected is True, "a sentence spoken while thinking must be re-captured, not dropped"


def test_bargein_active_while_speaking():
    s = _frame_session(main.Floor.SPEAKING)
    asyncio.run(Session._process_frame(s, FRAME))
    assert s.detected is True, "must interrupt while the reply is playing"


# ---- continuation merge: a paused answer resumed while thinking is glued back --

def test_start_turn_prepends_pending_audio():
    """The core of the first-sentence fix: when a continuation is pending, the new
    capture is glued onto the earlier audio so STT sees one combined utterance."""
    s = Session.__new__(Session)
    s._pending_prepend = b"SENTENCE_ONE"
    s._pending_replace = False
    s._bargein_run = 0
    captured = {}

    async def fake_wrapper(pcm, opening_text=None):
        captured["pcm"] = pcm

    s._turn_wrapper = fake_wrapper

    async def run():
        s._start_turn(b"SENTENCE_TWO")
        await asyncio.sleep(0)  # let the spawned task run

    asyncio.run(run())
    assert captured["pcm"] == b"SENTENCE_ONESENTENCE_TWO"
    assert s._pending_prepend is None
    assert s._inflight_pcm == b"SENTENCE_ONESENTENCE_TWO"
    assert s._turn_sent_transcript is False


def test_start_turn_without_pending_is_unchanged():
    s = Session.__new__(Session)
    s._pending_prepend = None
    s._pending_replace = False
    s._bargein_run = 0
    captured = {}

    async def fake_wrapper(pcm, opening_text=None):
        captured["pcm"] = pcm

    s._turn_wrapper = fake_wrapper

    async def run():
        s._start_turn(b"ONLY")
        await asyncio.sleep(0)

    asyncio.run(run())
    assert captured["pcm"] == b"ONLY"


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
