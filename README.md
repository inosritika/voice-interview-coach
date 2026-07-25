# Interview Studio

A voice-based mock interview app that runs on your laptop. You talk, it listens
and talks back, and it behaves like an actual interviewer — asks a question,
hears you out, follows up on what you actually said, and moves on when it's time.
It covers behavioral, DSA, machine learning, and system design rounds, and for
coding rounds there's a live editor it can watch while you type.

I built it to learn the voice-AI and agent stack by hand, so it isn't a thin
wrapper around one "realtime" API. The pipeline — mic → voice detection →
speech-to-text → LLM → text-to-speech — is wired up stage by stage, and every
stage is swappable behind an adapter. If you want the long version of how it all
fits together, there's a walkthrough in
[docs/learning-guide.html](docs/learning-guide.html).

Speech-to-text and text-to-speech run locally. The one real choice you make is
the brain: your Claude Code subscription (no API key), a local model through
Ollama (fully offline), or a hosted API.

## What it does

You pick a round — behavioral, DSA, ML, or system design — and one of two formats.
Discussion is a talk-it-through round: concepts, reasoning, trade-offs, voice only.
Hands-on gives you a code editor and a real problem to work; the interviewer sees
your code as you write it and reacts to it. There's a bank of 88 problems (easy
through hard, across DSA, ML, and system design) with a per-question timer, and
you can paste your own instead.

Once you're in, you just talk. There's no push-to-talk button — it figures out
when you've started and stopped. You can cut in while it's still speaking, the
same way you'd interrupt a person. And you can steer it out loud: ask for a harder
problem, the next question, a hint, or "just show me the answer" and it goes along
with it.

Two things worth calling out. **Learn this** is a button in the transcript that
opens a side panel and writes you a proper explanation of whatever you're stuck on
— the approach, the complexity, a worked example — as text you can read, with
follow-up questions, without the interviewer reading it aloud and without leaving
the app for ChatGPT. And at the end you get a **debrief**: an overall score, a
breakdown by dimension, what you did well and what to work on, plus delivery
numbers pulled from your own speech (pace, filler words, how much you talked).
A small local profile remembers your recurring weak points between sessions.

## Running it

You'll need Python 3.10 or newer and ffmpeg (whisper uses it to decode your mic).
On macOS that's `brew install ffmpeg`; on Debian/Ubuntu, `sudo apt install ffmpeg`.
On Windows, run the whole thing under WSL2 — the audio path assumes a Unix-ish
environment.

Clone it and install the Python side:

```bash
git clone https://github.com/inosritika/voice-interview-coach.git
cd voice-interview-coach/backend

python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The interviewer needs a voice. Piper voices come as two files (`.onnx` and
`.onnx.json`); grab one from the [piper voices list](https://github.com/rhasspy/piper/blob/master/VOICES.md)
and drop both into `backend/voices/`. `en_US-lessac-medium` is a fine default and
it's what `.env` already points at:

```bash
mkdir -p voices && cd voices
curl -LO https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx
curl -LO https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json
cd ..
```

(The speech-to-text model downloads itself the first time you run — nothing to do
there.)

Now copy the config and pick the brain:

```bash
cp ../.env.example ../.env
```

The default is Claude through your Claude Code subscription, which is the best
interviewer and needs no API key — just the CLI, signed in:

```bash
npm install -g @anthropic-ai/claude-code
claude          # run it once and use /login to sign in
```

If you'd rather stay fully offline, install [Ollama](https://ollama.com), pull a
model, and switch one line:

```bash
ollama pull qwen2.5:7b
# then set LLM_ENGINE=local in .env
```

Then start it, from `backend/` with the venv active:

```bash
uvicorn main:app --reload
```

Open http://localhost:8000, hit Start interview, and let the browser use your mic.
Use headphones if you can — they make interrupting the interviewer a lot more
reliable, since the app won't hear itself through your speakers.

## Using it

The lobby is where you set up the round: the type, the difficulty, and optionally
a job description and resume so the questions lean toward what you're actually
interviewing for. That text never leaves your machine. Then choose Discussion or
Hands-on, and for Hands-on either pick a problem or paste your own.

During the interview you answer by talking. To steer it, just say what you want —
these all work, and they're most useful in coding rounds:

- "move on to the next problem" / "next question"
- "give me something harder" / "an easier one" / "increase the difficulty"
- "give me a medium graph problem" (topic and difficulty together)
- "can I get a hint?"
- "I give up — walk me through the answer" (it'll actually teach you)
- "give me a debug problem" (you get buggy code to fix)

The Learn this button sits at the top of the transcript. Open it whenever you want
a written explanation of the current topic, ask follow-ups in the panel, and hit
Back to interview when you're done. End interview takes you to the debrief, and
Progress (from the lobby) shows what keeps tripping you up across sessions.

## Configuration

Everything is a flag in `.env`, and [`.env.example`](.env.example) explains each
one. The ones you're most likely to touch:

| Flag | Default | Other options |
|---|---|---|
| `LLM_ENGINE` | `claude` | `local` (Ollama) · `openai` (needs `OPENAI_API_KEY`) |
| `CLAUDE_MODEL` | `claude-sonnet-4-6` | an alias like `opus`/`haiku`, or a full model id |
| `OLLAMA_MODEL` | `qwen2.5:7b` | anything you've pulled |
| `STT_ENGINE` | `local` (faster-whisper) | `deepgram` |
| `TTS_ENGINE` | `local` (piper) | `cartesia` |
| `WHISPER_MODEL` | `small` | `tiny`…`large-v3`, trading speed for accuracy |
| `DIRECTOR` | `on` | `off` for a simpler, faster, single-call interviewer |

On the LLM choice: Claude is the smartest interviewer and needs no key, but it
spawns the CLI each turn, so expect roughly eight seconds before it speaks. Ollama
is snappier and works with no internet, at the cost of a smaller model that makes a
weaker interviewer. OpenAI is there if you have a key and want the hosted route.

## How it works

Every turn runs through the cascaded pipeline:

```
your voice → VAD (Silero) → endpointing → speech-to-text (whisper)
                                              → director → LLM reply
                       → sentence chunking → text-to-speech (piper) → your speakers
```

The trick that makes it swappable is that `main.py` only ever calls
`engines.get_stt()`, `get_llm()`, and `get_tts()` — it never knows which engine is
behind them. That's what lets you trade Claude for Ollama, or piper for Cartesia,
without touching the pipeline. Learn this deliberately sits outside all of it: it's
a plain streaming HTTP endpoint (`/api/learn`) that returns text, so it never goes
near the voice loop or the text-to-speech. The rest — the agent loop that decides
each turn, context compaction for long interviews, the scoring — is covered in
[docs/learning-guide.html](docs/learning-guide.html).

## Layout

```
interview-coach/
├── backend/
│   ├── main.py            FastAPI app, the /ws WebSocket loop, /api/learn
│   ├── config.py          all the flags (read from .env)
│   ├── engines.py         factory that picks and caches each engine per flag
│   ├── prompts.py         interviewer persona, director brain, Learn this tutor
│   ├── packs.py           per-round persona, rubric, and opening questions
│   ├── problems.py        the 88-problem coding bank + spoken-request matching
│   ├── director.py        the agent loop that decides each turn
│   ├── debrief.py         rubric scoring + delivery metrics
│   ├── storage.py         saved sessions + cross-session progress
│   ├── history.py         context compaction for long interviews
│   ├── endpointing.py     silence + semantic turn detection
│   ├── mcp_server.py      saved interviews as a tool for other agents (MCP)
│   ├── evals/             behavior evals (simulated candidates vs. the real stack)
│   ├── pipeline/          the turn strategies (cascaded, fused)
│   ├── stt/ tts/ llm/ vad/ turndetect/   the swappable engine adapters
│   └── data/              sessions, profiles, evals (made at runtime, git-ignored)
├── frontend/index.html    lobby, live stage, editor, Learn panel, debrief
├── docs/learning-guide.html
├── .env.example
└── LICENSE
```

## Tests

From `backend/`, with the venv active:

```bash
for t in test_*.py; do python "$t"; done
```

The behavior evals are separate — simulated candidates interview the real stack and
a judge model scores it:

```bash
python -m evals.run_eval --personas evasive --turns 3
```

## License

MIT — see [LICENSE](LICENSE). Use it, fork it, change it. The built-in coding
problems are original write-ups of well-known interview questions, written for this
project.
