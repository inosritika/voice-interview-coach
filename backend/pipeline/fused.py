"""The fused turn strategy: Gemma 4 is used for BOTH speech-understanding and the
interviewer reply — but as two clean, single-purpose calls, not one.

We originally tried a single call that emitted "transcript <newline> reply" and
parsed the stream. Diagnostics showed the model obeys that format only *some* of
the time (sometimes it skips the transcript and replies directly), so the parser
mis-split, leaked instruction text into the transcript, and poisoned the history —
which snowballed into nonsense over a few turns. Lesson learned: don't make one
small-model generation do two jobs in a self-delimited format.

So each turn is two calls on the same Gemma model:
  1. ASR call:   audio -> transcript only ("output only the transcript").
  2. Reply call: history + transcript (as text) -> the interviewer's next turn.

This is still "one model does STT and LLM" — just reliably. The cost is two
inferences per turn (slower); the payoff is that each call has a single job it
can't get wrong. Runs on Transformers + torch (heavier than our ONNX/CTranslate2
cascaded stack).

Model class (AutoModelForMultimodalLM -> Gemma4ForConditionalGeneration) and repo
id (google/gemma-4-E4B-it, Apache-2.0, not gated) are verified against a live
transformers==5.13.0 install and a real end-to-end run.
"""

import asyncio
import logging
from collections.abc import AsyncIterator
from threading import Thread

import numpy as np

import config
from pipeline.base import ReplyToken, Transcript, TurnEvent, TurnStrategy
from prompts import DIDNT_CATCH

log = logging.getLogger("interview-coach")

# Call 1's job: transcription, nothing else. Kept blunt on purpose so a small
# model doesn't wander off into commentary.
_ASR_INSTRUCTION = (
    "Transcribe this audio exactly, word for word. Output ONLY the transcript "
    "text — no preamble, no quotes, no explanation, nothing else."
)


class FusedStrategy(TurnStrategy):
    def __init__(self) -> None:
        self._model = None
        self._processor = None

    def _resolve_device(self) -> str:
        """Pick a concrete device ourselves instead of trusting transformers'
        device_map="auto". Measured on this project's own M3 Pro: "auto" decided
        to offload some layers to *disk* even with 36 GB of RAM free (logged as
        "Some parameters are on the meta device…"), and every forward pass paid
        the cost of paging them back in. Pinning everything to one real device
        (mps here) removed that entirely."""
        if config.GEMMA_DEVICE != "auto":
            return config.GEMMA_DEVICE
        import torch

        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch  # noqa: F401  (also selects the backend)
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        device = self._resolve_device()
        log.info(
            "loading Gemma 4 (%s) onto %s — first load downloads weights…",
            config.GEMMA_MODEL, device,
        )
        self._processor = AutoProcessor.from_pretrained(config.GEMMA_MODEL)
        # AutoModelForMultimodalLM resolves gemma4 -> Gemma4ForConditionalGeneration
        # via its internal mapping (verified against transformers 5.13.0).
        self._model = AutoModelForMultimodalLM.from_pretrained(
            config.GEMMA_MODEL,
            dtype=torch.bfloat16,
            device_map=device,
        )

    async def warmup(self) -> None:
        """Pay the load + MPS-kernel-compilation cost at server boot, not on the
        user's first turn. We exercise BOTH code paths — a tiny text generation
        and a tiny audio (ASR) generation — because MPS compiles kernels lazily
        per path, and the audio encoder's first run is the slowest part."""
        import time

        await asyncio.to_thread(self._load)
        log.info("warming up Gemma (compiling kernels, first turn will be fast)…")
        t0 = time.perf_counter()
        async for _ in self._generate(
            [{"role": "user", "content": "hi"}], audio=None, max_new_tokens=2
        ):
            pass
        silence = np.zeros(config.SAMPLE_RATE, dtype=np.float32)  # 1s of silence
        async for _ in self._generate(
            [], audio=silence, instruction=_ASR_INSTRUCTION, greedy=True, max_new_tokens=2
        ):
            pass
        log.info("Gemma warm (%.1fs)", time.perf_counter() - t0)

    async def run(
        self,
        utterance_pcm: bytes | None,
        history: list[dict],
        director_state=None,  # the agentic director is cascaded-only for now
        opening_text: str | None = None,
    ) -> AsyncIterator[TurnEvent]:
        if utterance_pcm is None:
            if opening_text is not None:
                yield ReplyToken(opening_text)
                return
            # Opening greeting: text-only reply, no transcript.
            async for tok in self._generate(history, audio=None):
                yield ReplyToken(tok)
            return

        audio = np.frombuffer(utterance_pcm, dtype=np.int16).astype(np.float32) / 32768.0

        # Call 1 — Gemma as STT. Greedy (no sampling) for faithful transcription.
        transcript = await self._transcribe(audio)
        yield Transcript(transcript)
        if not transcript.strip():
            # Nothing intelligible — ask them to repeat rather than go silent.
            yield ReplyToken(DIDNT_CATCH)
            return

        # Call 2 — Gemma as the interviewer. Clean text history + this turn's
        # transcript. No audio here, so it's the faster of the two calls.
        reply_history = history + [{"role": "user", "content": transcript}]
        async for tok in self._generate(reply_history, audio=None):
            yield ReplyToken(tok)

    async def _transcribe(self, audio: np.ndarray) -> str:
        out = ""
        async for tok in self._generate(
            [], audio=audio, instruction=_ASR_INSTRUCTION, greedy=True, max_new_tokens=128
        ):
            out += tok
        return out.strip()

    async def _generate(
        self,
        history: list[dict],
        audio: np.ndarray | None,
        instruction: str | None = None,
        greedy: bool = False,
        max_new_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """Run Gemma's generate with a streamer on a worker thread, bridged to
        async so it never blocks the event loop (same reason whisper/piper use
        to_thread)."""
        await asyncio.to_thread(self._load)
        from transformers import TextIteratorStreamer

        messages = list(history)
        if audio is not None:
            content: list[dict] = [{"type": "audio", "audio": audio}]
            if instruction:
                content.append({"type": "text", "text": instruction})
            messages.append({"role": "user", "content": content})

        inputs = self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self._model.device)

        streamer = TextIteratorStreamer(
            self._processor.tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        gen_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=max_new_tokens or config.GEMMA_MAX_NEW_TOKENS,
        )
        if greedy:
            gen_kwargs["do_sample"] = False
        else:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = config.OLLAMA_TEMPERATURE
        Thread(target=self._model.generate, kwargs=gen_kwargs, daemon=True).start()

        it = iter(streamer)
        while (piece := await asyncio.to_thread(next, it, None)) is not None:
            yield piece
