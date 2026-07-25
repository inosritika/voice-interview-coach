# Interview Studio — Voice Interview Coach

A hands-free, real-time **AI mock-interview partner** you run on your own machine.
You talk; it listens, asks follow-ups, and talks back — like a real interviewer.
Practice **behavioral, DSA, machine-learning, and system-design** rounds, work
real coding problems in a live editor, step aside to **read up on any topic**
mid-interview, and get a **scored debrief** at the end.

It runs **locally by default** — speech-to-text and text-to-speech are on-device,
and you choose the "brain": your **Claude Code subscription** (no API key), a
**local Ollama model** (fully offline), or a **hosted API**. Every stage sits
behind an adapter, so switching is a one-line change in `.env`.

> Built as a hands-on way to learn the voice-AI + agent stack from scratch. If you
> want the *why* behind each component, open
> [docs/learning-guide.html](docs/learning-guide.html) in a browser — a full
> what/why/how walkthrough with diagrams and a glossary.

---

## ✨ What you can do

- **🎙️ Just talk.** Hands-free voice in, voice out. No push-to-talk — the app
  detects when you start and stop speaking. **Interrupt any time** (barge-in),
  even mid-sentence, just like a real conversation.
- **🧠 An interviewer that adapts.** An agentic "director" decides each turn —
  probe deeper, move on, or wrap up — based on what you actually said. It asks
  **one question at a time** and doesn't read from a script.
- **📚 Four kinds of round:** Behavioral · DSA · Machine Learning · System Design.
  Each opens with a different question every time.
- **Two formats per round:**
  - **💬 Discussion** — talk through concepts, reasoning, and trade-offs (voice only).
  - **⌨️ Hands-on** — a **live code editor** the interviewer can see as you type,
    paired with a real problem from a built-in bank of **88 problems**
    (easy / medium / hard across DSA, ML, and system design) and a per-question timer.
- **🗣️ Steer it like a real interview.** Say *"give me a harder one"*, *"next
  problem"*, *"can I get a hint"*, or *"I give up — walk me through the answer"*,
  and it responds. Ask for a **debug-this-code** problem and you get buggy code to fix.
- **📖 "Learn this" — learn without leaving the app.** During any interview, hit
  **📖 Learn this** to open a side panel with a **written** explanation of the
  current topic (approach, complexity, worked example) — no voice, just a doc you
  can study. Ask follow-up questions in the panel, then jump back into the interview.
- **📝 Live transcript** of the whole conversation, written as you speak.
- **📊 Scored debrief.** At the end you get an overall score, per-dimension bars
  (structure, specificity, ownership, communication), strengths/improvements, and
  delivery metrics computed from your own speech (words-per-minute, filler rate,
  talk-time ratio). Export the transcript.
- **📈 Progress across sessions.** A lightweight local profile tracks your recurring
  weak points so you can see what keeps coming up.

---

## 🚀 Quick start

### 1. Prerequisites

| Need | Why | Install |
|---|---|---|
| **Python 3.10+** | runs the backend | python.org / your package manager |
| **ffmpeg** | whisper decodes your mic audio | macOS `brew install ffmpeg` · Debian/Ubuntu `sudo apt install ffmpeg` |
| **A piper voice** | the interviewer's voice (local TTS) | see step 3 |
| **An LLM engine** | the interviewer's brain | **Claude CLI** (default) *or* **Ollama** (offline) — see step 4 |

> **Windows:** run it under **WSL2** (Ubuntu). The audio pipeline and shell steps
> assume a Unix-like environment.

### 2. Install

```bash
git clone https://github.com/inosritika/voice-interview-coach.git
cd voice-interview-coach/backend

python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Download a voice for the interviewer

Piper voices are two files (`.onnx` + `.onnx.json`). Grab one from the
[piper voices list](https://github.com/rhasspy/piper/blob/master/VOICES.md) and
drop both into `backend/voices/`. A good default is `en_US-lessac-medium`:

```bash
mkdir -p voices && cd voices
curl -LO https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx
curl -LO https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json
cd ..
```

`.env` already points `PIPER_VOICE` at this file. (The first whisper run also
auto-downloads its small STT model — no manual step.)

### 4. Pick the interviewer's brain

Copy the config, then choose **one** of the two easy paths:

```bash
cp ../.env.example ../.env
```

**Path A — Claude via your Claude Code subscription (default, best quality, no API key):**

```bash
npm install -g @anthropic-ai/claude-code   # the `claude` CLI
claude                                      # run once, then /login to sign in
```

`.env` already has `LLM_ENGINE=claude`. That's it.

**Path B — Fully offline with Ollama (no subscription, no internet):**

```bash
# install Ollama from https://ollama.com, then:
ollama pull qwen2.5:7b
```

Then set `LLM_ENGINE=local` in `.env`.

### 5. Run

```bash
# from backend/, with the venv active
uvicorn main:app --reload
```

Open **http://localhost:8000** and click **Start interview**. Allow microphone
access when the browser asks. 🎧 **Headphones are recommended** — they make
interrupting the interviewer more reliable.

---

## 🧭 Using the interviewer

1. **In the lobby**, pick a **round type** (Behavioral / DSA / ML / System Design)
   and a difficulty. Optionally paste or upload a **job description** and **resume**
   so questions are tailored — nothing is uploaded anywhere; it stays on your machine.
2. **Choose a format:**
   - **💬 Discussion** for a talk-it-through round, or
   - **⌨️ Hands-on** to get a live editor and a real problem. Pick one from the
     bank or paste your own.
3. **Start**, then **just talk** to answer. The interviewer speaks; start speaking
   any time to **interrupt** it.
4. **Steer the session by voice** — for coding rounds especially:
   - *"Can we move on to the next problem?"* / *"next question"*
   - *"Give me something harder"* / *"an easier one"* / *"increase the difficulty"*
   - *"Give me a **medium graph** problem"* (topic + difficulty)
   - *"Can I get a hint?"*
   - *"I give up — can you explain the solution?"* → it teaches you the answer
   - *"Give me a **debug** problem"* → you get buggy code to fix
5. **📖 Learn this** (top of the transcript) opens the written-explanation panel for
   the current topic. Type follow-up questions there, then **Back to interview**.
6. **End interview** → you get the **scored debrief**. Export the transcript if you like.
7. **Progress** (from the lobby) shows your recurring weak points across sessions.

---

## ⚙️ Configuration

Everything is a flag in `.env` (copied from
[`.env.example`](.env.example), which documents each one). The most useful:

| Flag | Default | Options |
|---|---|---|
| `LLM_ENGINE` | `claude` | `claude` (subscription, no key) · `local` (Ollama) · `openai` (`OPENAI_API_KEY`) |
| `CLAUDE_MODEL` | `claude-sonnet-4-6` | any alias (`sonnet`/`opus`/`haiku`) or full model id |
| `OLLAMA_MODEL` | `qwen2.5:7b` | any model you've `ollama pull`ed |
| `STT_ENGINE` | `local` (faster-whisper) | `deepgram` (`DEEPGRAM_API_KEY`) |
| `TTS_ENGINE` | `local` (piper) | `cartesia` (`CARTESIA_API_KEY`, `CARTESIA_VOICE_ID`) |
| `WHISPER_MODEL` | `small` | `tiny`→`large-v3` (accuracy vs. speed) |
| `DIRECTOR` | `on` | `off` for a simpler, faster single-call interviewer |

**Which LLM should I use?** `claude` gives the smartest interviewer and needs no
API key (just the signed-in CLI), at ~8s per turn since it spawns the CLI each
turn. `local` (Ollama) is fully offline and snappier but a smaller model is a
weaker interviewer. `openai` uses the hosted Responses API if you have a key.

---

## 🏗️ How it works (in brief)

Each turn flows through a **cascaded pipeline**:

```
your voice ─▶ VAD (Silero) ─▶ endpointing ─▶ STT (whisper)
                                                  │
                                                  ▼
                              director (agentic decide-loop) ─▶ LLM reply
                                                  │
                                                  ▼
your speakers ◀── TTS (piper) ◀── sentence chunking ◀── streamed tokens
```

The core design idea: `main.py` only ever calls `engines.get_stt()`,
`get_llm()`, `get_tts()` — it never knows which concrete engine is behind them.
That adapter boundary is what lets you swap Claude for Ollama, or piper for
Cartesia, without touching the pipeline. The **"Learn this"** feature is separate
from the voice loop entirely: a plain streaming HTTP endpoint (`/api/learn`) that
returns text only — no TTS, and it never touches the live session.

For the full component-by-component explanation, open
[docs/learning-guide.html](docs/learning-guide.html).

---

## 📁 Project layout

```
interview-coach/
├── backend/
│   ├── main.py            FastAPI app + /ws WebSocket loop + /api/learn tutor
│   ├── config.py          all flags (read from .env)
│   ├── engines.py         factory: picks each engine per flag, caches it
│   ├── prompts.py         interviewer persona · director brain · "Learn this" tutor
│   ├── packs.py           per-round persona, rubric, and random opening questions
│   ├── problems.py        the 88-problem coding bank + spoken-request matching
│   ├── director.py        the agentic tool-use loop (actions, state, guards)
│   ├── debrief.py         rubric scoring + delivery metrics
│   ├── storage.py         saved sessions + cross-session progress profiles
│   ├── history.py         context compaction for long interviews
│   ├── endpointing.py     silence + semantic turn detection
│   ├── mcp_server.py      expose saved interviews to other agents (MCP, stdio)
│   ├── evals/run_eval.py  agent-vs-agent behavior evals
│   ├── pipeline/          turn strategies (cascaded · fused)
│   ├── stt/ · tts/ · llm/ · vad/ · turndetect/   swappable engine adapters
│   └── data/              sessions · profiles · evals   (created at runtime, git-ignored)
├── frontend/index.html    lobby · live stage · code editor · Learn panel · debrief
├── docs/learning-guide.html   the full what/why/how guide
├── .env.example           copy to .env
└── LICENSE                MIT
```

---

## 🧪 Development

Run the test suite (from `backend/`, venv active):

```bash
for t in test_*.py; do python "$t"; done
```

Behavior evals (simulated candidates interview the real stack, a judge model scores it):

```bash
python -m evals.run_eval --personas evasive --turns 3
```

---

## 📄 License

[MIT](LICENSE) © 2026 Ritika Soni. Free to use, fork, and modify.

The built-in coding problems are original write-ups of canonical, widely-known
interview questions, written for this project.
