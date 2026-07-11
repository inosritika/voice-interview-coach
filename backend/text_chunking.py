"""Turn a stream of LLM tokens into a stream of *speakable chunks*.

This is the hinge of step 3. The LLM emits tokens one at a time; if we wait for
the whole reply before synthesizing, we're back to sequential latency. Instead we
watch the tokens accumulate and, as soon as we have a complete clause or sentence,
we hand it to TTS and start playing — while the LLM keeps generating the rest.

The trick for low latency: get the *first* chunk out as early as possible. So for
the very first chunk we'll break at an early clause boundary (a comma), then switch
to whole sentences after that (which sound more natural). First-word-fast, then
smooth.
"""

from collections.abc import AsyncIterator

_SENTENCE_ENDERS = ".!?"
_CLAUSE_ENDERS = ",;:"


def _is_boundary(buf: str, i: int) -> bool:
    """Punctuation only ends a chunk if whitespace follows it. This keeps
    decimals ("3.5"), abbreviations ("Node.js") and grouped numbers ("1,000")
    from being split mid-token and spoken in pieces. Punctuation at the very
    end of the buffer is NOT a boundary yet — we can't know what follows until
    the next token arrives (the final tail gets flushed after the stream ends)."""
    return i + 1 < len(buf) and buf[i + 1].isspace()


def _find_split(buf: str, first: bool, first_clause_min: int) -> int | None:
    """Index of the character to split *after*, or None if no good split yet."""
    # Earliest sentence ender always wins.
    best: int | None = None
    for i, ch in enumerate(buf):
        if ch in _SENTENCE_ENDERS and _is_boundary(buf, i):
            best = i
            break
    # For the first chunk only, a comma past a minimum length is good enough —
    # it gets audio playing sooner. Take it if it comes before any sentence ender.
    if first:
        for i, ch in enumerate(buf):
            if ch in _CLAUSE_ENDERS and i >= first_clause_min and _is_boundary(buf, i):
                if best is None or i < best:
                    best = i
                break
    return best


async def speakable_chunks(
    tokens: AsyncIterator[str], first_clause_min: int = 18
) -> AsyncIterator[str]:
    """Consume an async token stream, yield trimmed speakable chunks."""
    buf = ""
    first = True
    async for tok in tokens:
        buf += tok
        while (idx := _find_split(buf, first, first_clause_min)) is not None:
            chunk = buf[: idx + 1].strip()
            buf = buf[idx + 1 :]
            if chunk:
                yield chunk
                first = False
    tail = buf.strip()
    if tail:
        yield tail
