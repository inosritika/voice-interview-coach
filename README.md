# Voice Interview Coach

A hands-free mock-interview partner, built to **learn the voice-AI stack by hand**.
We build the cascaded pipeline (VAD → Endpoint → STT → LLM → TTS) piece by piece,
then swap in an end-to-end realtime model at the end.

Everything runs **locally by default** — no API keys needed. Every stage is behind
an adapter interface, so you can flip a flag to use a hosted engine instead.

**New to the concepts?** Open [docs/learning-guide.html](docs/learning-guide.html) in a browser —
a full glossary with examples, architecture diagrams, and a what/why/how walkthrough of every component.

## Where we are

**The agent-harness layer is built.** On top of the director (below), five features
landed in one pass — each hand-rolled, each teaching one concept:

- **Context compaction** (`backend/history.py`, `COMPACTION=on`): long interviews no
  longer overflow llama3's 8K window. The LLM reads a budgeted view — system prompt
  pinned, oldest exchanges rolled into an *incremental* summary, recent turns verbatim —
  while the session keeps the full lossless record for the debrief. Closes the project's
  oldest known issue.
- **Session persistence + candidate memory** (`backend/storage.py`): every finished
  interview is saved to `backend/data/sessions/` (transcript, metrics, debrief, the
  director's notes), and a per-candidate profile accumulates across sessions — enter the
  same name in the lobby and the interviewer quietly knows what needed work last time.
- **MCP server** (`backend/mcp_server.py`): the coach as a tool for *other* agents.
  `list_interviews` / `get_transcript` / `get_debrief` / `get_progress` over the saved
  sessions, speaking the Model Context Protocol over stdio. Register with
  `claude mcp add interview-coach -- ./.venv/bin/python mcp_server.py` and ask an agent
  "what keeps going wrong in my mock interviews?".
- **Eval harness** (`backend/evals/run_eval.py`): regression tests for interviewer
  *behavior*. Simulated candidates (strong / rambler / evasive personas) interview the
  real stack — same prompts, same director — and a judge model scores the interviewer
  against a checklist (one question per turn, probes vague answers, no fabrication…) as
  structured JSON. `python -m evals.run_eval --personas evasive --turns 3` → scores table
  saved to `data/evals/`. Prompt changes are now measured, not vibed.
- **Semantic turn detection** (`ENDPOINT_MODE=semantic`): two-stage endpointing. At
  450 ms of silence, the open ~8 MB smart-turn model (BSD-2-Clause, runs in ~40 ms on
  CPU) judges from *prosody* whether you sound finished: yes → the turn ends early
  (snappier than the fixed 700 ms), no → patience extends to 1.4 s, so thinking pauses
  stop getting guillotined. Off by default; one-line model download in
  `backend/turndetect/smart_turn.py`.

**The interview director — the interviewer is now an agent.** With `DIRECTOR=on`
(default), each turn runs a hand-rolled tool-use loop before speaking: the LLM picks
structured actions as **schema-constrained JSON** (Ollama structured outputs — invalid
JSON is impossible, not just unlikely): `note_evidence`, `note_red_flag`, `probe_deeper`,
`switch_topic`, `end_interview`. Notes accumulate in an evidence notebook that lives
*outside* the model (per session), feeds the debrief judge, and streams to the UI's
developer panel so you can watch the agent think. The terminal move then guides the
normal streamed spoken reply, so step 3's token streaming is preserved. The harness does
validation, one bounded retry on bad output, an iteration cap, and a safe fallback move —
a broken director can never break the interview. **Honest cost:** ~2 extra small LLM
calls per turn (~3–4s on llama3:8b) before speech starts; `DIRECTOR=off` restores the
single-call speed. (Found live: a conversation ending in a `system` message breaks
llama3's chat template — the directive is folded into the user message instead.)

**Step 5 — scored debrief.** When the interview ends, the whole transcript is scored
against a behavioral rubric (structure / specificity / ownership / communication) by the
LLM acting as an *evaluator*, and combined with **delivery metrics we compute ourselves**
(words-per-minute, filler rate, average answer length, talk-time ratio — no model, just
arithmetic on the captured speech). The frontend shows a debrief screen with an overall
score, per-dimension bars, strengths/improvements, and a transcript export.

**Step 4 — barge-in.** You can interrupt the interviewer mid-sentence. Each turn runs as
a cancellable `asyncio` task so the receive loop stays free to hear you; the mic stays
live during playback (browser echo cancellation stops the AI hearing itself), and sustained
user speech (stricter VAD bar + ~240ms, tunable via `BARGEIN_*`) cancels the reply, tells
the browser to flush its audio queue, and hands the floor back.

The UI is now a proper interview platform: a pre-interview lobby (candidate details,
interview-type picker, live mic check), a live stage (timer, question counter, interviewer
presence, mic meter, mute/end), and the debrief screen — with honest "Coming Soon" badges
on features not built yet (recording/export, video, technical/coding/system-design rounds).

Build progression: 1) push-to-talk · 2) VAD + endpointing · 3) streaming + latency HUD
· 4) barge-in · 5) debrief/scoring ← _here_ · 6) realtime S2S swap.

**Tip:** use headphones for the most reliable barge-in — on speakers, the browser's echo
cancellation does the heavy lifting but residual leak can occasionally mis-trigger.

Step 1 (push-to-talk) is documented in [docs/learning-guide.html](docs/learning-guide.html);
the code has since evolved to the streaming input + streaming output paths above.

## Layout

```
interview-coach/
├── backend/
│   ├── main.py            FastAPI app + /ws WebSocket loop (floor, barge-in, turns)
│   ├── config.py          all flags (read from .env)
│   ├── engines.py         factory: picks each engine per flag, caches it
│   ├── prompts.py         interviewer persona + director brain + directives
│   ├── director.py        the agentic tool-use loop (actions, state, harness)
│   ├── history.py         context compaction (incremental rolling summary)
│   ├── storage.py         saved sessions + cross-session candidate profiles
│   ├── mcp_server.py      MCP server over saved interviews (stdio)
│   ├── debrief.py         rubric scoring + delivery metrics
│   ├── endpointing.py     silence + two-stage semantic endpointing
│   ├── text_chunking.py   token stream -> speakable clauses
│   ├── evals/run_eval.py  agent-vs-agent behavior evals
│   ├── pipeline/  base.py · cascaded.py · fused.py   (turn strategies)
│   ├── stt/  base.py · local_whisper.py · deepgram.py
│   ├── tts/  base.py · local_piper.py   · cartesia.py
│   ├── llm/  base.py · local_ollama.py  · openai_api.py
│   ├── vad/  base.py · silero.py
│   ├── turndetect/  base.py · smart_turn.py   (semantic endpointing model)
│   └── data/  sessions/ · profiles/ · evals/  (created at runtime)
├── frontend/index.html    lobby + live stage + debrief (AudioWorklet, Web Audio)
├── docs/learning-guide.html   the full what/why/how guide
└── .env.example           copy to .env
```

The pipeline in `main.py` only ever calls `engines.get_stt()/get_tts()/get_llm()`.
It never knows which concrete engine is active — that's the whole adapter point.

## Setup (all-local)

Prereqs: **ffmpeg** (for whisper audio decode) and **Ollama** (local LLM).

```bash
brew install ffmpeg ollama       # macOS
ollama serve &                   # start the local LLM server
ollama pull llama3.2             # pull a model

cd interview-coach/backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Download a piper voice into backend/voices/ and point PIPER_VOICE at it:
#   https://github.com/rhasspy/piper/blob/master/VOICES.md
# e.g. en_US-lessac-medium.onnx  (+ its .onnx.json alongside it)

cp ../.env.example ../.env       # defaults are all-local; edit if needed
uvicorn main:app --reload
```

Open http://localhost:8000 → paste a JD + resume → **Start interview** → hold **Hold
to talk** (or the spacebar) to answer.

## Pipeline: cascaded vs. fused (Gemma 4)

`PIPELINE` in `.env` swaps the whole turn strategy, not just one stage:

- `cascaded` (default) — whisper STT → Ollama LLM, two models. **Faster on this
  project's hardware** (M3 Pro): ~2.7s to first reply, measured.
- `fused` — Gemma 4 (`google/gemma-4-E4B-it`) does speech-understanding and the
  reply in one call. Genuinely works (verified: accurate transcript, relevant
  follow-up questions) but **measured ~1.8x slower** (~4.8s) than cascaded here —
  Transformers-on-MPS isn't as optimized as Ollama's llama.cpp backend for raw
  token generation, so removing the STT→LLM handoff doesn't overcome that gap.
  Needs `pip install "transformers>=4.55" accelerate pillow torchvision` (not
  gated, Apache-2.0, but a multi-GB download on first run).

One fixed bug worth knowing if you use `fused` on Apple Silicon: transformers'
`device_map="auto"` silently offloaded some layers to *disk* even with plenty of
RAM free (logged as "Some parameters are on the meta device…"), which alone
accounted for over half the latency. `pipeline/fused.py` now detects `mps`
explicitly instead of trusting `"auto"`.

## Swapping to a hosted engine

Flip a flag in `.env` and add the matching key — no code changes:

| Flag | local (default) | hosted |
|---|---|---|
| `STT_ENGINE` | faster-whisper | `deepgram` (`DEEPGRAM_API_KEY`) |
| `TTS_ENGINE` | piper | `cartesia` (`CARTESIA_API_KEY`, `CARTESIA_VOICE_ID`) |
| `LLM_ENGINE` | ollama | `openai` (`OPENAI_API_KEY`) |

## WebSocket protocol

```
client → server:  {"type":"setup","jd":"…","resume":"…"}   (text)
                  <binary audio clip>                       (one turn)
                  {"type":"debrief"}                         (end of interview → score it)
server → client:  {"type":"status","text":"thinking"}       (UI hints)
                  {"type":"transcript","text":"…"}           (what STT heard)
                  {"type":"reply","text":"…"}                (interviewer words)
                  <binary WAV>                               (reply audio)
                  {"type":"debrief", "overall":.., "dimensions":[…],
                   "strengths":[…], "improvements":[…], "metrics":{…}}
                  {"type":"error","text":"…"}
```
