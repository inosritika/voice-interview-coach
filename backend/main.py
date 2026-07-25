"""FastAPI app + WebSocket endpoint — the step 2 hands-free loop.

The button is gone. The browser now streams raw PCM (16 kHz, mono, 16-bit) the
whole time the mic is on. On the server:

  incoming PCM  ->  cut into 512-sample frames
                ->  VAD scores each frame (speech probability)
                ->  Endpointer turns that stream into turn boundaries
                ->  on END: assemble the utterance -> STT -> LLM -> TTS -> reply

A tiny floor-state machine tracks who has the floor. We only run VAD while
LISTENING; while we PROCESS a turn and while the reply is SPEAKING, we ignore
incoming audio (and the client mutes its mic too). True barge-in — interrupting
the reply — is step 4; this is the groundwork for it.

Protocol
  client -> server:
    - JSON text frame: {"type": "setup", "jd": "...", "resume": "..."}
    - JSON text frame: {"type": "debrief"}   # interview over -> score it (step 5)
    - binary frames: a continuous stream of raw PCM (16 kHz mono s16le)
  server -> client:
    - {"type": "state",       "text": "listening|processing|speaking"}  # floor state
    - {"type": "vad",         "text": "speech|silence"}                  # live mic hint
    - {"type": "status",      "text": "transcribing|thinking|speaking"}  # stage hint
    - {"type": "transcript",  "text": "..."}   # what STT heard you say
    - {"type": "reply_delta", "text": "..."}   # a chunk of the reply, as it streams
    - binary frame: WAV audio for that chunk (step 3: one per speakable chunk)
    - {"type": "metrics", "stt_ms":.., "text_ms":.., "audio_ms":..}  # latency HUD
    - {"type": "interrupt",   "text": ""}     # barge-in: stop playback NOW
    - {"type": "director",    "text": "note_evidence: ..."}  # agent action (dev panel)
    - {"type": "debrief", "overall":.., "dimensions":[..], "strengths":[..],
       "improvements":[..], "metrics":{..}}   # step 5: the scored review
    - {"type": "error",       "text": "..."}
"""

import asyncio
import collections
import io
import json
import logging
import time
import wave
from contextlib import asynccontextmanager
from enum import Enum, auto

import numpy as np
from fastapi import FastAPI, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, StreamingResponse
import pathlib

import config
import engines
import storage
from director import DirectorState
from endpointing import Endpointer, Event
from history import Compactor
from packs import get_pack, opening_question
import problems
from pipeline.base import DirectorAction, ReplyToken, Transcript
from prompts import build_learn_messages, build_system_prompt, coding_block
from speech_filter import spoken_only
from text_chunking import speakable_chunks


def _fmt(ms: float | None) -> str:
    return "-" if ms is None else f"{ms:.0f}ms"


def _wav_seconds(wav_bytes: bytes) -> float:
    """Playback length of a WAV clip, so the server can stay 'speaking' until the
    browser actually finishes playing it (not just until we finish sending it)."""
    try:
        with wave.open(io.BytesIO(wav_bytes)) as w:
            return w.getnframes() / float(w.getframerate())
    except Exception:  # noqa: BLE001
        return 0.0

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("interview-coach")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm the active pipeline at boot so the user's FIRST turn isn't slow. For
    # fused (Gemma) this loads weights + compiles GPU kernels up front; for
    # cascaded it's a cheap no-op. Failure here must not stop the server.
    try:
        await engines.get_pipeline().warmup()
    except Exception:  # noqa: BLE001
        log.exception("pipeline warmup failed (continuing without it)")
    yield


app = FastAPI(title="Voice Interview Coach", lifespan=lifespan)

FRONTEND_DIR = pathlib.Path(__file__).resolve().parent.parent / "frontend"

_FRAME_BYTES = config.VAD_FRAME_SAMPLES * 2  # 2 bytes per int16 sample
_PREROLL_FRAMES = max(1, round(config.SPEECH_PREROLL_MS / 32))  # 32 ms per frame
_BARGEIN_FRAMES = max(1, round(config.BARGEIN_MIN_SPEECH_MS / 32))  # sustained speech to interrupt
_BARGEIN_GAP_FRAMES = max(1, round(config.BARGEIN_GAP_MS / 32))     # quiet frames tolerated mid-run
_PARTIAL_INTERVAL_FRAMES = max(1, round(config.PARTIAL_INTERVAL_MS / 32))  # how often to re-transcribe live


class Floor(Enum):
    """Who currently holds the conversational floor. Explicit state now, so
    step 4's barge-in has something concrete to reason about."""

    LISTENING = auto()   # the user's turn — VAD is live
    PROCESSING = auto()  # STT/LLM/TTS running — ignore incoming audio
    SPEAKING = auto()    # the reply is playing on the client — ignore audio


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/api/progress")
async def api_progress(candidate: str = "") -> dict:
    """A candidate's coaching history: score trajectory plus the weak points that
    keep recurring. Reads the small cross-session profile (storage.py) — the last
    10 debriefs, a few KB — not the full transcripts."""
    profile = storage.load_profile(candidate)
    if not profile:
        return {"found": False, "candidate": candidate}
    sessions = profile.get("sessions", [])
    return {
        "found": True,
        "candidate": profile.get("candidate", candidate),
        "sessions": sessions,
        "recurring": storage.recurring_improvements(profile),
    }


@app.get("/api/problems")
async def api_problems(
    fmt: str | None = None, topic: str | None = None, company: str | None = None
) -> dict:
    """The lobby's problem picker. Returns PUBLIC fields only — the interviewer's
    private brief (intended solution / planted bug) never leaves the server. A
    `company` just re-orders the list (its favourites first); nothing is hidden."""
    return {"problems": [p.public() for p in problems.list_problems(fmt, topic, company)]}


@app.post("/api/learn")
async def api_learn(payload: dict) -> StreamingResponse:
    """The "Learn this" side panel: a text-only tutor, OUTSIDE the interview.

    The candidate pauses the mock interview to actually understand a topic. We
    take the recent transcript (+ the hands-on problem, if any) as CONTEXT, ask
    the LLM to teach it in Markdown, and STREAM the answer straight back — no
    TTS, no director, no touch to the live voice session (this is a plain HTTP
    call; the WebSocket keeps running underneath). Follow-up questions carry the
    panel's own thread so it's a real conversation, not one-shot.

    The whole point: learn while practicing, without leaving for another app."""
    messages = build_learn_messages(payload)

    async def gen():
        try:
            async for token in engines.get_llm().stream(messages):
                yield token
        except Exception as exc:  # noqa: BLE001 — a teaching miss must not 500
            log.exception("learn stream failed")
            yield f"\n\n_Sorry — the explanation failed to generate ({exc})._"

    # text/plain, not SSE: the client reads the raw byte stream and renders
    # Markdown as it grows. no-transform/no-buffering so proxies don't hold it.
    return StreamingResponse(
        gen(),
        media_type="text/plain; charset=utf-8",
        headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
    )


_EXTRACT_MAX_BYTES = 10 * 1024 * 1024  # a resume is never 10 MB of text


def _extract_pdf_text(data: bytes) -> tuple[str, int]:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages).strip(), len(reader.pages)


@app.post("/api/extract")
async def extract(file: UploadFile) -> dict:
    """Turn an uploaded resume/JD file into plain text for the lobby.

    Upload FILLS the textarea rather than replacing it: the extracted text
    lands back in the field, still visible and editable, and flows through the
    exact same `setup` message — nothing downstream changes. PDF extraction is
    text-layer only (pypdf); scanned/image PDFs would need OCR we don't do."""
    data = await file.read()
    if len(data) > _EXTRACT_MAX_BYTES:
        return {"error": "file too large (10 MB max)"}
    name = (file.filename or "").lower()
    try:
        if name.endswith(".pdf") or data[:5] == b"%PDF-":
            # pypdf is blocking CPU work — same rule as every model call: off
            # the event loop so a big PDF can't freeze live interviews.
            text, pages = await asyncio.to_thread(_extract_pdf_text, data)
            if not text:
                return {"error": "no extractable text — is this a scanned PDF?"}
            return {"text": text, "pages": pages}
        if name.endswith((".txt", ".md")):
            return {"text": data.decode("utf-8", errors="replace").strip(), "pages": 1}
    except Exception as exc:  # noqa: BLE001 — a bad file must not 500
        log.exception("extract failed for %r", file.filename)
        return {"error": f"couldn't read the file: {exc}"}
    return {"error": "unsupported file type — use PDF, .txt or .md"}


@app.websocket("/ws")
async def ws(sock: WebSocket) -> None:
    await sock.accept()
    log.info("client connected")
    session = Session(sock)
    try:
        await session.run()
    except WebSocketDisconnect:
        log.info("client disconnected")


class Session:
    """One connection = one interview. Holds the conversation history plus all
    the per-turn streaming state (VAD, endpointer, audio buffers)."""

    def __init__(self, sock: WebSocket) -> None:
        self.sock = sock
        self.messages: list[dict] = []
        self.floor = Floor.LISTENING

        # Step 5 (debrief): one entry per candidate answer (words + how long they
        # actually spoke + the text), plus total AI speaking time. Enough to
        # compute delivery metrics — WPM, filler rate, talk ratio — at the end.
        self.turn_stats: list[dict] = []
        self.ai_speak_secs = 0.0
        self.candidate = ""
        self.role = ""
        # Topic + company selected in the lobby (packs.py / companies.py).
        # Defaults match the pre-existing behavioral-only behavior.
        self.interview_type = "behavioral"
        self.company = "generic"

        # Coding round (problems.py): the chosen problem and the LIVE contents of
        # the shared editor. `code` is streamed from the client as they type and
        # injected into the interviewer's context each turn (code-awareness).
        self.problem = None
        self.code = ""

        # The director's memory: evidence notebook, red flags, topics covered.
        # Lives on the session (NOT in the cached strategy singleton), so every
        # interview starts with a clean notebook.
        self.director_state = DirectorState()

        # Context compaction: self.messages stays the FULL record (debrief and
        # saved transcripts need everything); the LLM gets a budgeted view. Runs
        # on the fast utility engine — it's an internal summary, not spoken.
        self.compactor = Compactor(engines.get_utility_llm())

        self.vad = engines.get_vad()
        # Smart endpointing (ENDPOINT_MODE=semantic): at a mid-pause checkpoint,
        # score the utterance-so-far with up to two signals — prosody (smart-turn
        # audio model) and, optionally, a true text-semantic check (LLM). Both
        # are fused in _utterance_completeness. Falls back to plain silence mode
        # if neither signal is available.
        self._prosody = engines.get_turn_checker()
        self._semantic = engines.get_semantic_checker()
        checker = self._utterance_completeness if (self._prosody or self._semantic) else None
        self.endpointer = Endpointer(checker=checker)

        # Leftover bytes that didn't fill a whole 512-sample frame yet.
        self._residual = bytearray()
        # A small ring buffer of recent frames, so we can prepend a little audio
        # from just before VAD triggered (avoids clipping the first phoneme).
        self._preroll: collections.deque[bytes] = collections.deque(maxlen=_PREROLL_FRAMES)
        # The frames of the utterance currently being captured.
        self._utterance: list[bytes] = []

        # Step 4: the current turn runs as its own cancellable task so the receive
        # loop stays free to hear the user interrupt. _bargein_run counts the speech
        # frames seen while the AI holds the floor, _bargein_gap counts the quiet
        # frames since the last speech one (a short gap is tolerated — see
        # _detect_bargein), and _bargein_buf keeps those frames so the
        # interruption's first word isn't thrown away when we hand the floor back.
        self._turn_task: asyncio.Task | None = None
        self._bargein_run = 0
        self._bargein_gap = 0
        self._bargein_buf: list[bytes] = []

        # Continuation-merge state. If the user pauses mid-answer, we start
        # processing, and they resume BEFORE the reply is audible, that resumed
        # speech isn't an interruption — it's the rest of their answer. We stash
        # the in-flight utterance's audio and glue it in front of the resumed
        # capture so the whole thing is transcribed as ONE turn (fixes: "my first
        # sentence didn't get saved, only the second showed up").
        self._inflight_pcm: bytes | None = None   # audio of the turn being processed
        self._pending_prepend: bytes | None = None  # audio to glue onto the next capture
        self._continuation = False                # set while a continuation-cancel is in flight
        self._turn_sent_transcript = False        # did this turn already show a transcript?
        self._pending_replace = False             # should the next transcript REPLACE the last one?
        self._replace_next_transcript = False     # armed for the merged turn's on_transcript

        # Live partial transcripts: while the user talks, re-transcribe the audio
        # so far every few hundred ms and stream it as an interim result, so their
        # words appear in real time instead of only after they pause.
        self._partial_task: asyncio.Task | None = None
        self._last_partial_frames = 0             # utterance length at the last partial
        self._utterance_gen = 0                   # bumped when an utterance ends — stales in-flight partials

    def _turn_active(self) -> bool:
        return self._turn_task is not None and not self._turn_task.done()

    async def run(self) -> None:
        while True:
            frame = await self.sock.receive()
            if frame.get("type") == "websocket.disconnect":
                break

            if (text := frame.get("text")) is not None:
                await self._on_text(text)
            elif (audio := frame.get("bytes")) is not None:
                await self._on_audio(audio)

    # ---- inbound handlers ----------------------------------------------------

    async def _on_text(self, text: str) -> None:
        try:
            msg = json.loads(text)
        except json.JSONDecodeError:
            # A malformed control frame must not take the whole session down.
            log.warning("ignoring malformed control frame: %.80s", text)
            return
        if msg.get("type") == "setup":
            # The lobby sends name/role explicitly (it also folds them into the
            # JD preface); we keep them for persistence + the candidate profile.
            self.candidate = (msg.get("name") or "").strip()
            self.role = (msg.get("role") or "").strip()
            self.interview_type = (msg.get("interviewType") or "behavioral").strip()
            self.company = (msg.get("company") or "generic").strip()
            difficulty = msg.get("difficulty", "standard")
            # Coding round: the lobby either picked a problem from the bank
            # (problemId) or pasted their own (customProblem). Resolve it here; the
            # editor pre-fills with its starter code (client-side, from /api/problems).
            self.problem = self._resolve_problem(msg)
            self.code = self.problem.starter_code if self.problem else ""
            # The problem is injected PER TURN (see _run_turn), not baked in here,
            # so a mid-interview switch changes what the interviewer sees.
            system = build_system_prompt(
                msg.get("jd", ""), msg.get("resume", ""),
                self.interview_type, self.company, difficulty,
            )
            self.director_state.problem = self.problem
            # The opening problem counts as shown, so "next question" moves past it.
            self.director_state.shown_problems = {self.problem.id} if self.problem else set()
            # Cross-session memory: if we've coached this candidate before, give
            # the interviewer their history (weak spots to revisit) — recalled at
            # setup, stored at debrief (storage.py).
            system += storage.profile_prompt_block(self.candidate)
            self.messages = [{"role": "system", "content": system}]
            # Thread the topic/company onto the director's state too, so the
            # silent decide-loop gets domain-appropriate judgment guidance
            # (prompts.build_director_prompt) without changing its call site.
            self.director_state.interview_type = self.interview_type
            self.director_state.company = self.company
            # Open by presenting the specific problem if there is one; otherwise
            # the calibrated area/difficulty opening.
            opening = (
                problems.coding_opening(self.problem) if self.problem
                else opening_question(self.interview_type, difficulty)
            )
            self._start_turn(None, opening_text=opening)
        elif msg.get("type") == "code":
            # Live editor contents — just stash them; the next spoken turn folds
            # them into the interviewer's context. Capped so a paste can't blow up
            # the prompt.
            self.code = (msg.get("text") or "")[:8000]
        elif msg.get("type") == "debrief":
            # The interview ended — score it and send the review back.
            await self._send_debrief()

    def _resolve_problem(self, msg: dict):
        """Turn the lobby's selection into a Problem, or None for a plain
        conversational round."""
        custom = msg.get("customProblem")
        if custom and (custom.get("prompt") or "").strip():
            return problems.custom_problem(
                title=custom.get("title", ""), prompt=custom.get("prompt", ""),
                starter_code=custom.get("starterCode", ""), fmt=custom.get("fmt", "solve"),
            )
        return problems.get_problem(msg.get("problemId"))

    async def _on_audio(self, chunk: bytes) -> None:
        # We ALWAYS process incoming audio now — even while the AI is speaking —
        # so we can detect a barge-in. _process_frame decides what to do based on
        # whether a turn is currently holding the floor.
        self._residual += chunk
        while len(self._residual) >= _FRAME_BYTES:
            frame_bytes = bytes(self._residual[:_FRAME_BYTES])
            del self._residual[:_FRAME_BYTES]
            await self._process_frame(frame_bytes)

    async def _process_frame(self, frame_bytes: bytes) -> None:
        samples = np.frombuffer(frame_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        prob = self.vad.speech_prob(samples)

        # While the AI holds the floor, incoming speech goes through the barge-in
        # path — and it's active in BOTH phases, because its meaning differs:
        #   SPEAKING   the reply is playing  -> a real interruption; cut it off.
        #   PROCESSING still thinking, nothing played yet -> the user is CONTINUING
        #              their answer (adding a sentence); cancel the premature turn
        #              and re-capture so those words aren't silently dropped
        #              (reported: "I said something 2s later and it wasn't written").
        # Same mechanism either way (cancel + replay the buffered speech). A
        # sustained run (BARGEIN_MIN_SPEECH_MS) is required so stray noise doesn't
        # trigger it, and the fast utility engine keeps PROCESSING short enough that
        # this can't live-lock the way a 40s local model once did.
        if self._turn_active():
            await self._detect_bargein(prob, frame_bytes)
            return

        # The user's turn: normal VAD -> endpointing. (update is async because
        # semantic mode may consult the turn model mid-pause — off-thread.)
        event = await self.endpointer.update(prob)
        if not self._utterance:
            # Not yet capturing — remember this frame in case speech starts soon.
            self._preroll.append(frame_bytes)

        if event is Event.START:
            await self._send("vad", "speech")
            # Seed the utterance with the pre-roll so we don't clip the onset.
            self._utterance = list(self._preroll)
            self._utterance.append(frame_bytes)
        elif event is Event.NONE:
            if self._utterance:  # mid-utterance: keep every frame (incl. pauses)
                self._utterance.append(frame_bytes)
        elif event is Event.CANCEL:
            # Too short to be a real turn — throw it away, keep listening.
            await self._send("vad", "silence")
            self._utterance = []
            self._end_capture()
        elif event is Event.END:
            await self._send("vad", "silence")
            utterance = b"".join(self._utterance)
            self._utterance = []
            self._end_capture()
            self._start_turn(utterance)

        # Stream a live partial transcript of the audio so far, so the user's words
        # show up as they speak instead of only once they pause.
        if self._utterance and config.PARTIAL_TRANSCRIPTS:
            self._maybe_partial()

    def _end_capture(self) -> None:
        """An utterance just ended or was cancelled: bump the generation so any
        in-flight partial transcription is recognized as stale and dropped, and
        reset the partial throttle for the next utterance."""
        self._utterance_gen += 1
        self._last_partial_frames = 0

    def _maybe_partial(self) -> None:
        """Throttle + spawn a live partial transcription of the audio so far. At
        most one runs at a time (transcription is slower than the frame rate), and
        only every _PARTIAL_INTERVAL_FRAMES so we don't re-transcribe every frame."""
        if self._partial_task is not None and not self._partial_task.done():
            return
        if len(self._utterance) - self._last_partial_frames < _PARTIAL_INTERVAL_FRAMES:
            return
        self._last_partial_frames = len(self._utterance)
        # Include any paused-answer prefix so the partial shows the whole answer.
        snapshot = (self._pending_prepend or b"") + b"".join(self._utterance)
        self._partial_task = asyncio.create_task(
            self._emit_partial(snapshot, self._utterance_gen)
        )

    async def _emit_partial(self, snapshot: bytes, gen: int) -> None:
        """Transcribe the snapshot (fast/greedy) and stream it as an interim result
        — unless the utterance ended while we were working (gen changed) or we've
        left the listening floor. A failed partial must never disrupt capture."""
        try:
            hint = get_pack(self.interview_type).stt_hint
            text = await engines.get_stt().transcribe(snapshot, initial_prompt=hint, fast=True)
        except Exception:  # noqa: BLE001
            return
        if text and gen == self._utterance_gen and self.floor is Floor.LISTENING:
            await self.sock.send_json({"type": "partial", "text": text})

    async def _utterance_completeness(self) -> float:
        """Endpointing hook: P(the utterance captured so far is a finished turn),
        fusing up to two signals with a CASCADE. Prosody (smart-turn, ~40ms) runs
        first; the expensive text-semantic check (STT + LLM, ~1s) only runs when
        prosody is genuinely uncertain — when prosody is confident either way we
        trust it and skip the slow call. Called once per pause by the Endpointer."""
        pcm = b"".join(self._utterance)

        # Prosody-only (no text signal configured): the original fast path.
        if self._semantic is None:
            return await self._prosody.completeness(pcm)

        # Text-only (prosody model missing but semantic enabled).
        if self._prosody is None:
            return await self._semantic.completeness(pcm)

        # Both available: cascade. Trust a confident prosody verdict; only pay for
        # the semantic check inside the uncertain band, then fuse the two.
        p = await self._prosody.completeness(pcm)
        if p <= config.SEMANTIC_CASCADE_LOW or p >= config.SEMANTIC_CASCADE_HIGH:
            return p
        s = await self._semantic.completeness(pcm)
        w = config.SEMANTIC_PROSODY_WEIGHT
        fused = w * p + (1 - w) * s
        log.info("endpoint fuse: prosody %.2f + semantic %.2f -> %.2f", p, s, fused)
        return fused

    async def _detect_bargein(self, prob: float, frame_bytes: bytes) -> None:
        """The AI is speaking; has the user started talking over it? Require a run
        of confident speech frames so residual echo or a cough doesn't cut the AI
        off spuriously. The run's frames are buffered so the ~240ms of speech that
        *triggered* the barge-in isn't lost — it's replayed after the cancel.

        The run tolerates a short GAP of quiet frames. Real speech dips below the
        VAD bar constantly — between syllables, on unvoiced consonants — so the
        original "any quiet frame resets the run to zero" rule demanded 8 loud
        frames with no dip, i.e. shouting continuously. That made interrupting
        feel broken (reported live: "it takes a lot of time / a higher volume").
        Counting through gaps up to BARGEIN_GAP_MS keeps the echo protection (a
        sustained run is still required) while letting normal speech interrupt."""
        if prob >= config.BARGEIN_THRESHOLD:
            self._bargein_run += 1
            self._bargein_gap = 0
            self._bargein_buf.append(frame_bytes)
            if self._bargein_run >= _BARGEIN_FRAMES:
                await self._trigger_bargein()
        elif self._bargein_run:
            # Mid-utterance dip, not the end of it — keep counting, and keep the
            # frame so the replayed audio stays contiguous (no clipped syllables).
            self._bargein_gap += 1
            self._bargein_buf.append(frame_bytes)
            if self._bargein_gap > _BARGEIN_GAP_FRAMES:
                self._bargein_run = 0
                self._bargein_gap = 0
                self._bargein_buf.clear()

    # ---- turn lifecycle ------------------------------------------------------

    def _start_turn(self, pcm: bytes | None, opening_text: str | None = None) -> None:
        """Spawn the turn as a background task and return immediately, so the
        receive loop keeps running and can interrupt it."""
        merged = pcm is not None and self._pending_prepend is not None
        if merged:
            # This capture is the continuation of a paused answer — prepend the
            # earlier audio so STT sees the whole answer as one utterance.
            pcm = self._pending_prepend + pcm
        self._pending_prepend = None
        # The merged turn's transcript REPLACES the partial one we already showed
        # (in place), so the user sees one growing bubble instead of a duplicate.
        # Only replace if a partial was actually on screen (_pending_replace).
        self._replace_next_transcript = merged and self._pending_replace
        self._pending_replace = False
        self._inflight_pcm = pcm
        self._turn_sent_transcript = False
        self._bargein_run = 0
        self._turn_task = asyncio.create_task(self._turn_wrapper(pcm, opening_text))

    async def _turn_wrapper(self, pcm: bytes | None, opening_text: str | None = None) -> None:
        try:
            await self._run_turn(pcm, opening_text)
        except asyncio.CancelledError:
            # Barge-in cancelled us mid-reply; the barge-in handler does cleanup.
            pass
        finally:
            self._turn_task = None

    async def _trigger_bargein(self) -> None:
        """The user started talking while a turn was in flight. What that MEANS
        depends on the floor:

          SPEAKING   the reply is audible  -> a real interruption; stop playback.
          otherwise  still thinking, nothing played -> the user is CONTINUING their
                     answer; merge the in-flight audio into this new capture so
                     nothing they said is lost.

        Either way we cancel the turn and hand the floor back, replaying the
        frames that triggered the detection so the first word survives."""
        continuation = self.floor is not Floor.SPEAKING
        inflight = self._inflight_pcm
        already_shown = self._turn_sent_transcript
        self._continuation = continuation  # read by _run_turn's cancel handler
        log.info(
            "barge-in: %s",
            "continuation — user resumed while thinking" if continuation
            else "user interrupted the reply",
        )
        task = self._turn_task
        self._turn_task = None
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._continuation = False

        if continuation and inflight:
            # Glue the earlier audio onto the resumed capture (see _start_turn).
            self._pending_prepend = inflight
            # If we already showed the earlier partial, the merged transcript will
            # UPDATE that same bubble in place (no duplicate). If we hadn't shown it
            # yet, the merged one is the first the user sees — a fresh bubble.
            self._pending_replace = already_shown
        else:
            await self._send("interrupt", "")  # client flushes queued/playing audio

        self._bargein_run = 0
        self._bargein_gap = 0
        replay = self._bargein_buf
        self._bargein_buf = []
        await self._resume_listening()
        # Replay the frames that fired the detection into the fresh listening
        # state: VAD sees them as speech, START fires, and the new utterance is
        # seeded with them — so the first word survives.
        for frame in replay:
            await self._process_frame(frame)

    # ---- turn processing -----------------------------------------------------

    async def _run_turn(self, pcm: bytes | None, opening_text: str | None = None) -> None:
        """Produce one interviewer turn — the opening greeting (pcm=None) or a
        reply to a captured user utterance.

        The active pipeline strategy decides HOW (cascaded STT+LLM, or fused
        Gemma); everything here is shared: cut the streamed reply into speakable
        chunks, synthesize each, stream its audio the moment it's ready, and time
        the turn for the HUD. One turn's failure is reported and never kills the
        session."""
        t_start = time.perf_counter() if pcm is not None else None
        await self._set_floor(Floor.PROCESSING)
        await self._send("status", "thinking")

        self._user_text: str | None = None
        m_transcript_ms: float | None = None
        m_first_text_ms: float | None = None
        m_first_audio_ms: float | None = None
        reply_parts: list[str] = []
        # Track how much audio we've queued to the client and when playback began,
        # so we can hold the "speaking" floor (and keep watching for a barge-in)
        # until the browser has actually finished playing it.
        audio_secs = 0.0
        first_audio_at: float | None = None

        def elapsed() -> float | None:
            return None if t_start is None else (time.perf_counter() - t_start) * 1000

        async def on_transcript(text: str) -> None:
            nonlocal m_transcript_ms
            self._user_text = text
            m_transcript_ms = elapsed()
            if text:
                self._turn_sent_transcript = True
                await self.sock.send_json({
                    "type": "transcript", "text": text,
                    "replace": self._replace_next_transcript,
                })
                self._replace_next_transcript = False

        recorded = False

        def record_turn(interrupted: bool = False) -> None:
            """Commit this turn to history + debrief stats — exactly once. Called
            on normal completion AND from the cancellation path below, so an early
            barge-in (mid-generation) can't erase the user's answer from history.
            (Appending the user message in on_transcript instead would double-add
            it: cascaded.py builds `history + [user]` AFTER yielding Transcript.)"""
            nonlocal recorded
            if recorded:
                return
            recorded = True
            reply = " ".join(reply_parts).strip()
            if interrupted and reply:
                # Let the LLM (and the debrief evaluator) know this line was
                # never fully delivered.
                reply += " [cut off by the candidate]"
            if self._user_text:
                self.messages.append({"role": "user", "content": self._user_text})
            if reply:
                self.messages.append({"role": "assistant", "content": reply})
            if pcm is not None:
                text = self._user_text or ""
                # The capture includes ~160ms preroll + ~700ms endpoint silence
                # that isn't talking — trim it so WPM isn't deflated.
                raw_secs = len(pcm) / 2 / config.SAMPLE_RATE  # int16 = 2 bytes/sample
                overhead = (config.SPEECH_PREROLL_MS + config.ENDPOINT_SILENCE_MS) / 1000
                self.turn_stats.append({
                    "words": len(text.split()),
                    "speech_secs": max(0.0, raw_secs - overhead),
                    "text": text,
                })

        async def on_director(text: str) -> None:
            # Surface the agent's thinking to the UI (dev panel) — never spoken.
            await self.sock.send_json({"type": "director", "text": text})

        try:
            tts = engines.get_tts()
            # The pipeline reads a budget-compacted view; self.messages (which
            # record_turn appends to) remains the full, lossless record.
            llm_view = await self.compactor.view(self.messages)
            # Code-awareness: append the CURRENT editor contents to the (pinned)
            # system prompt for this turn, so the interviewer reacts to what the
            # candidate has actually written. Appending to index 0 keeps it out of
            # the compaction-summary slot and the director's transcript view.
            if self.problem is not None:
                head = dict(llm_view[0])
                head["content"] += (
                    coding_block(self.problem)
                    + "\n\n--- CURRENT EDITOR CONTENTS (the candidate's live scratchpad) ---\n"
                    + (self.code.strip() or "(empty)")
                )
                llm_view = [head] + llm_view[1:]
            events = engines.get_pipeline().run(
                pcm, llm_view, director_state=self.director_state, opening_text=opening_text
            )
            # spoken_only sits between the model and the chunker: it strips
            # bracketed stage directions the model copies from our own annotations
            # ("[cut off by the candidate]", "[Direction for this turn: …]") and
            # cuts the turn off after its first question. Everything downstream —
            # TTS, the UI transcript, self.messages — sees only speakable words.
            async for chunk in speakable_chunks(
                spoken_only(
                    self._reply_tokens(events, on_transcript, on_director),
                    one_question=config.ONE_QUESTION_PER_TURN,
                )
            ):
                if m_first_text_ms is None:
                    m_first_text_ms = elapsed()
                reply_parts.append(chunk)
                await self._send("reply_delta", chunk + " ")

                audio = await tts.synthesize(chunk)
                if m_first_audio_ms is None:
                    # First speech of this turn — take the floor, start the clock.
                    await self._set_floor(Floor.SPEAKING)
                    await self._send("status", "speaking")
                    m_first_audio_ms = elapsed()
                    first_audio_at = time.perf_counter()
                await self.sock.send_bytes(audio)
                audio_secs += _wav_seconds(audio)

            record_turn()
            await self._maybe_switch_problem()
            log.info(
                "turn: transcript %s · first-text %s · first-audio %s",
                _fmt(m_transcript_ms), _fmt(m_first_text_ms), _fmt(m_first_audio_ms),
            )

            if t_start is not None and m_first_audio_ms is not None:
                # stt_ms is only meaningful when the transcript arrived before the
                # first audio (cascaded); fused emits it last, so report None then.
                stt_ms = (
                    round(m_transcript_ms)
                    if m_transcript_ms is not None and m_transcript_ms <= m_first_audio_ms
                    else None
                )
                await self.sock.send_json({
                    "type": "metrics",
                    "stt_ms": stt_ms,
                    "text_ms": round(m_first_text_ms) if m_first_text_ms is not None else None,
                    "audio_ms": round(m_first_audio_ms),
                })

            # We finished GENERATING the reply, but the browser is still PLAYING
            # it. Stay on the SPEAKING floor for the remaining playback time so a
            # barge-in during playback still interrupts (this sleep is cancellable,
            # so _trigger_bargein cuts it short). Without this, mid-playback speech
            # would fall through and be captured as the next turn instead.
            if first_audio_at is not None:
                remaining = audio_secs - (time.perf_counter() - first_audio_at)
                if remaining > 0:
                    await asyncio.sleep(remaining)
            # The user heard the whole reply — count all of it toward AI airtime.
            self.ai_speak_secs += audio_secs
        except asyncio.CancelledError:
            # Barge-in cancelled us. For a real INTERRUPTION, commit what we have —
            # without this, an interrupt during generation erased the user's answer
            # (and partial reply) from history, so the next question ignored it.
            # For a CONTINUATION, do NOT commit: this same audio is being merged
            # into the next capture and re-processed, so committing it here would
            # duplicate the user's words in history.
            if not self._continuation:
                record_turn(interrupted=True)
                if first_audio_at is not None:
                    # Count only the audio that actually played before the cut-off.
                    self.ai_speak_secs += min(audio_secs, time.perf_counter() - first_audio_at)
            raise
        except Exception as exc:  # noqa: BLE001 — surface, don't crash the session
            log.exception("turn failed")
            await self._send("error", str(exc))

        await self._resume_listening()

    @staticmethod
    async def _reply_tokens(events, on_transcript, on_director):
        """Flatten a strategy's event stream into just the reply tokens (for the
        chunker), routing Transcript and DirectorAction events to their callbacks
        as they appear."""
        async for ev in events:
            if isinstance(ev, Transcript):
                await on_transcript(ev.text)
            elif isinstance(ev, DirectorAction):
                await on_director(ev.text)
            elif isinstance(ev, ReplyToken):
                yield ev.text

    async def _resume_listening(self) -> None:
        # Fresh turn: clear VAD memory and all capture buffers.
        self.vad.reset()
        self.endpointer.reset()
        self._residual = bytearray()
        self._preroll.clear()
        self._utterance = []
        self._bargein_run = 0
        self._bargein_gap = 0
        self._bargein_buf = []
        # Stale any partial still in flight from the turn we just finished, and
        # reset the throttle for the next utterance.
        self._utterance_gen += 1
        self._last_partial_frames = 0
        await self._set_floor(Floor.LISTENING)

    # ---- debrief -------------------------------------------------------------

    async def _maybe_switch_problem(self) -> None:
        """If the reply just presented a NEW problem (the candidate asked to switch,
        detected deterministically in cascaded.reply_events), load it into the
        editor and reset the per-question clock so the screen matches the words."""
        nxt = getattr(self.director_state, "pending_problem", None)
        if nxt is None:
            return
        self.director_state.pending_problem = None
        self.problem = nxt
        self.code = nxt.starter_code
        try:
            await self.sock.send_json({"type": "problem", **nxt.public()})
        except Exception:  # noqa: BLE001 — a UI push must not break the turn
            log.exception("failed to push switched problem")

    async def _send_debrief(self) -> None:
        """Score the finished interview and send the review. Cancel any in-flight
        turn first so the transcript we score isn't mutating under us."""
        if self._turn_active():
            task = self._turn_task
            self._turn_task = None
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        from debrief import generate_debrief

        await self._send("status", "scoring")
        try:
            payload = await generate_debrief(
                self.messages, self.turn_stats, self.ai_speak_secs,
                director_notes=self.director_state.notes_for_debrief(),
                interview_type=self.interview_type,
            )
            await self.sock.send_json(payload)
            # Persist the finished interview (MCP server reads these) and fold
            # the outcome into the candidate's cross-session profile.
            if payload.get("ok"):
                storage.save_session(
                    candidate=self.candidate,
                    role=self.role,
                    messages=self.messages,
                    turn_stats=self.turn_stats,
                    debrief=payload,
                    director_notes=self.director_state.notes_for_debrief(),
                    interview_type=self.interview_type,
                    company=self.company,
                )
                storage.update_profile(self.candidate, payload)
        except Exception as exc:  # noqa: BLE001 — never crash the session on debrief
            log.exception("debrief failed")
            await self._send("error", f"debrief failed: {exc}")

    # ---- outbound helpers ----------------------------------------------------

    async def _set_floor(self, floor: Floor) -> None:
        self.floor = floor
        await self._send("state", floor.name.lower())

    async def _send(self, type_: str, text: str) -> None:
        await self.sock.send_json({"type": type_, "text": text})
