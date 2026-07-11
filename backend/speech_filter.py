"""Guard the spoken channel: strip stage directions, enforce one question.

The interviewer's system prompt *asks* for two things a small local model keeps
forgetting: never narrate, and end every turn with exactly ONE question. This
module makes both true regardless of what the model does — the same lesson as
the director's schema validation (director.py) and the debrief's computed
metrics (§17): **move what can be enforced deterministically out of the model's
head.**

Why it's needed, concretely. Two places legitimately annotate the conversation
with bracketed meta-text:

  - `[cut off by the candidate]`   appended to an interrupted reply (main.py)
  - `[Direction for this turn: …]` folded into the user's message (cascaded.py)

Both are useful context for the model. But llama3 *reads* those brackets as part
of the dialogue's style and starts emitting its own — observed live, spoken aloud
by the TTS and printed in the transcript:

    "…what makes this project stand out? [Wait for response before proceeding]"

Rather than beg the prompt to stop, we filter the token stream: bracketed spans
never reach TTS, the UI, or the saved history. The same pass stops the stream at
the first question mark, which turns "AT MOST two sentences, exactly ONE
question" from a hopeful instruction into a guarantee.

This sits between the strategy's token stream and the chunker (text_chunking.py),
so it costs nothing in latency — it's a pass-through generator, not a buffer.
"""

from collections.abc import AsyncIterator


async def spoken_only(
    tokens: AsyncIterator[str], one_question: bool = True
) -> AsyncIterator[str]:
    """Yield only what the interviewer should actually SAY.

    - Drops every `[...]` span (stage directions, narration, leaked directives).
      Nesting is tracked, and an unclosed `[` swallows the rest of the turn —
      which is the safe failure: better a short reply than a spoken stage note.
    - With `one_question`, ends the turn at the first `?`. Closing turns (which
      end the interview and ask nothing) contain no `?`, so they stream in full.

    Square brackets are only DROPPED when they hold a prose stage direction; a
    bracket that holds DATA is kept and spoken. The distinguisher: a data bracket
    contains a digit or a comma (`[2, 7, 11, 15]`, `[i, j]`, `[9]`) — which a DSA
    interviewer naturally uses to show an example array — while a stage direction
    is words only (`[Wait for response]`, `[cut off by the candidate]`,
    `[Direction for this turn: probe deeper]`). Without this, DSA example arrays
    got gutted to "an array like " (observed live). Parentheses are always left
    alone — models use them for real speech ("we'd cache it (in Redis, say)").

    Ending early (on the "?") means we stop pulling tokens the model is still
    generating. We must CLOSE the source stream deterministically rather than
    leave it suspended for the garbage collector: GC-finalizing httpx's live
    streaming generator races with itself ("aclose(): asynchronous generator is
    already running") and can leak the connection. The `finally` closes the
    chain in-task, between yields — which unwinds cleanly down to Ollama's
    `async with`. It also runs harmlessly on normal exhaustion (aclose on a
    finished generator is a no-op).
    """
    depth = 0
    bracket = ""  # the whole "[...]" span buffered while inside brackets
    try:
        async for tok in tokens:
            out: list[str] = []
            stop = False
            for ch in tok:
                if ch == "[":
                    depth += 1
                    bracket += "["
                    continue
                if depth:
                    bracket += ch
                    if ch == "]":
                        depth -= 1
                        if depth == 0:
                            inner = bracket[1:-1]
                            # keep data ([2, 7, 11]); drop prose stage directions
                            if any(c.isdigit() for c in inner) or "," in inner:
                                out.append(bracket)
                            bracket = ""
                    continue
                out.append(ch)
                if one_question and ch == "?":
                    stop = True
                    break  # the turn's one question just landed
            text = "".join(out)
            if text:
                yield text
            if stop:
                return
    finally:
        aclose = getattr(tokens, "aclose", None)
        if aclose is not None:
            await aclose()  # propagates down to close the LLM stream in-task
