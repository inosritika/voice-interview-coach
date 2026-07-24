"""Endpointing / turn detection — deciding "they've finished" vs "they're just
thinking".

This is the concept step 2 exists to teach, and it's widely considered the
hardest UX problem in voice agents. VAD tells us speech-vs-silence per frame;
the Endpointer accumulates that stream and makes the *turn-level* decision.

Two strategies now live here, chosen by whether a `checker` is injected:

SILENCE (the classic baseline, and the default): once we've heard real speech,
a fixed-length pause (ENDPOINT_SILENCE_MS) means "done". Deliberately simple so
you can *feel* its failure modes:

  - too short  -> it guillotines you mid-thought ("My biggest win was… <pause>")
  - too long   -> every reply feels laggy

SMART (two-stage, ENDPOINT_MODE=semantic): the fixed threshold is replaced by
two, with a `checker` consulted between them —

    silence reaches MIN (450ms)
        -> ask the checker: is the utterance-so-far a FINISHED turn?
        complete   -> END now (snappier than the old 700ms)
        incomplete -> keep waiting…
    silence reaches MAX (1400ms) -> END regardless (nobody pauses forever)
    speech resumes at any point  -> pause state resets, checker re-arms

The `checker` itself can fuse up to THREE signals, each good at a different
input (see main.py `_utterance_completeness`):
  - silence  (Silero VAD)        — the timing this machine already tracks
  - prosody  (smart-turn, audio) — trailing intonation / phrase-final lengthening
  - semantic (LLM, transcript)   — does the sentence GRAMMATICALLY complete?
That's how production turn-taking is built: an acoustic and a linguistic signal
covering each other's blind spots. The checker is injected as an async callable
rather than imported here, so this state machine stays testable with a fake —
and knows nothing about ONNX, STT, or LLMs.
"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import Enum, auto
import logging

import config

log = logging.getLogger("interview-coach")


class Event(Enum):
    NONE = auto()    # nothing notable this frame
    START = auto()   # speech just began — start buffering the utterance
    END = auto()     # a real turn just finished — go process it
    CANCEL = auto()  # the "speech" was too short to be real — discard it


@dataclass
class Endpointer:
    """Feed it one VAD probability per frame via `update`; it returns an Event.

    A frame is VAD_FRAME_SAMPLES long (32 ms at 16 kHz), so we convert the
    millisecond thresholds from config into a number of frames once, up front.

    `checker`, when provided, switches on semantic mode: an async callable
    returning P(the utterance so far is a complete turn). `update` is async
    because that check happens inside it — but only ever once per pause.
    """

    threshold: float = config.VAD_THRESHOLD
    checker: Callable[[], Awaitable[float]] | None = None

    def __post_init__(self) -> None:
        frame_ms = config.VAD_FRAME_SAMPLES / config.SAMPLE_RATE * 1000
        self._silence_frames_needed = round(config.ENDPOINT_SILENCE_MS / frame_ms)
        self._min_silence_frames = round(config.ENDPOINT_MIN_SILENCE_MS / frame_ms)
        self._max_silence_frames = round(config.ENDPOINT_MAX_SILENCE_MS / frame_ms)
        self._min_speech_frames = round(config.MIN_SPEECH_MS / frame_ms)
        self.reset()

    def reset(self) -> None:
        self._in_speech = False    # are we currently inside an utterance?
        self._speech_frames = 0    # how many speech frames this utterance
        self._silence_run = 0      # consecutive silence frames right now
        self._checked_this_pause = False  # semantic: one verdict per pause

    async def update(self, speech_prob: float) -> Event:
        is_speech = speech_prob >= self.threshold

        if not self._in_speech:
            # Waiting for the turn to begin.
            if is_speech:
                self._in_speech = True
                self._speech_frames = 1
                self._silence_run = 0
                return Event.START
            return Event.NONE

        # Inside an utterance.
        if is_speech:
            self._speech_frames += 1
            self._silence_run = 0
            self._checked_this_pause = False  # new pause -> fresh verdict later
            return Event.NONE

        self._silence_run += 1

        if self.checker is None:
            # SILENCE mode: one fixed threshold.
            if self._silence_run < self._silence_frames_needed:
                return Event.NONE
            return self._finish()

        # SEMANTIC mode: two thresholds with a model consulted between them.
        if self._silence_run < self._min_silence_frames:
            return Event.NONE
        if self._silence_run >= self._max_silence_frames:
            return self._finish()  # patience exhausted — end regardless
        if not self._checked_this_pause:
            self._checked_this_pause = True
            prob = await self.checker()
            if prob >= config.SEMANTIC_TURN_THRESHOLD:
                log.info("endpoint: semantic END (complete %.2f)", prob)
                return self._finish()
            log.info("endpoint: sounds unfinished (%.2f) — extending patience", prob)
        return Event.NONE

    def _finish(self) -> Event:
        """The turn is over. Was there enough real speech to count?"""
        enough_speech = self._speech_frames >= self._min_speech_frames
        self.reset()
        return Event.END if enough_speech else Event.CANCEL
