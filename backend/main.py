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
from fastapi.responses import FileResponse
import pathlib

import config
import engines
import storage
from director import DirectorState
from endpointing import Endpointer, Event
from history import Compactor
from pipeline.base import DirectorAction, ReplyToken, Transcript
from prompts import build_system_prompt
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


class Floor(Enum):
    """Who currently holds the conversational floor. Explicit state now, so
    step 4's barge-in has something concrete to reason about."""

    LISTENING = auto()   # the user's turn — VAD is live
    PROCESSING = auto()  # STT/LLM/TTS running — ignore incoming audio
    SPEAKING = auto()    # the reply is playing on the client — ignore audio


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


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

        # The director's memory: evidence notebook, red flags, topics covered.
        # Lives on the session (NOT in the cached strategy singleton), so every
        # interview starts with a clean notebook.
        self.director_state = DirectorState()

        # Context compaction: self.messages stays the FULL record (debrief and
        # saved transcripts need everything); the LLM gets a budgeted view.
        self.compactor = Compactor(engines.get_llm())

        self.vad = engines.get_vad()
        # Semantic endpointing (ENDPOINT_MODE=semantic): inject a checker that
        # scores the utterance captured SO FAR; falls back to plain silence
        # mode automatically if the model isn't available.
        checker = None
        if engines.get_turn_checker() is not None:
            checker = self._utterance_completeness
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
            system = build_system_prompt(
                msg.get("jd", ""), msg.get("resume", ""),
                self.interview_type, self.company, msg.get("difficulty", "standard"),
            )
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
            # Let the interviewer open with a greeting + first question.
            self._start_turn(None)
        elif msg.get("type") == "debrief":
            # The interview ended — score it and send the review back.
            await self._send_debrief()

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

        # While the AI holds the floor, we're not endpointing a user turn — we're
        # watching for an interruption.
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
        elif event is Event.END:
            await self._send("vad", "silence")
            utterance = b"".join(self._utterance)
            self._utterance = []
            self._start_turn(utterance)

    async def _utterance_completeness(self) -> float:
        """Semantic endpointing hook: P(the utterance captured so far is a
        finished turn), from the smart-turn model. Called by the Endpointer at
        the mid-pause checkpoint."""
        return await engines.get_turn_checker().completeness(b"".join(self._utterance))

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

    def _start_turn(self, pcm: bytes | None) -> None:
        """Spawn the turn as a background task and return immediately, so the
        receive loop keeps running and can interrupt it."""
        self._bargein_run = 0
        self._turn_task = asyncio.create_task(self._turn_wrapper(pcm))

    async def _turn_wrapper(self, pcm: bytes | None) -> None:
        try:
            await self._run_turn(pcm)
        except asyncio.CancelledError:
            # Barge-in cancelled us mid-reply; the barge-in handler does cleanup.
            pass
        finally:
            self._turn_task = None

    async def _trigger_bargein(self) -> None:
        """The user cut in. Cancel the reply, tell the client to stop playing
        instantly, and hand the floor back so we capture their new utterance."""
        log.info("barge-in: user interrupted the reply")
        task = self._turn_task
        self._turn_task = None
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        await self._send("interrupt", "")  # client flushes queued/playing audio
        self._bargein_run = 0
        self._bargein_gap = 0
        replay = self._bargein_buf
        self._bargein_buf = []
        await self._resume_listening()
        # Replay the frames that fired the barge-in into the fresh listening
        # state: VAD sees them as speech, START fires, and the new utterance is
        # seeded with them — so the first word of the interruption survives.
        for frame in replay:
            await self._process_frame(frame)

    # ---- turn processing -----------------------------------------------------

    async def _run_turn(self, pcm: bytes | None) -> None:
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
                await self._send("transcript", text)

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
            events = engines.get_pipeline().run(
                pcm, llm_view, director_state=self.director_state
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
            # Barge-in cancelled us. Commit what we have first — without this, an
            # interrupt during generation erased the user's answer (and partial
            # reply) from history, so the next question ignored what they said.
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
        await self._set_floor(Floor.LISTENING)

    # ---- debrief -------------------------------------------------------------

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
