"""Tests for text_chunking.speakable_chunks.

Pins the regression where a DSA example array (`[2, 7, 11, 15]`) was split on a
comma INSIDE the brackets — TTS then spoke "…array two" <pause> "seven, eleven…".
Punctuation inside square brackets must never be a chunk boundary.

Run: python test_chunking.py
"""

import asyncio

from text_chunking import speakable_chunks


async def _gen(s: str):
    # Emit token-by-token to exercise the streaming split, not a single blob.
    for word in s.split(" "):
        yield word + " "


def _chunks(s: str) -> list[str]:
    async def run():
        return [c async for c in speakable_chunks(_gen(s))]
    return asyncio.run(run())


def test_example_array_is_not_split_on_inner_comma():
    chunks = _chunks("Given the array [2, 7, 11, 15], find two numbers that add to nine.")
    assert not any(c.rstrip().endswith("[2,") for c in chunks), chunks
    # The whole array survives intact inside a single chunk.
    assert any("[2, 7, 11, 15]" in c for c in chunks), chunks


def test_normal_sentences_still_split():
    chunks = _chunks("First we scope it. Then we design it. Finally we scale it.")
    assert len(chunks) == 3, chunks


def test_first_chunk_still_breaks_early_on_a_comma():
    # No brackets: the first-clause comma rule should still fire for low latency
    # (the comma must be past first_clause_min=18 chars to count).
    chunks = _chunks("In my experience so far, we always scope the problem first.")
    assert chunks[0] == "In my experience so far,", chunks


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
