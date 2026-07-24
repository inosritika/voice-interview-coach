"""Regression checks for calibrated first-question delivery."""

from __future__ import annotations

import asyncio

from pipeline.base import ReplyToken
from pipeline.cascaded import reply_events


def test_calibrated_opening_skips_the_model_and_is_spoken_verbatim():
    class NeverCalled:
        def stream(self, _history):
            raise AssertionError("the model must not generate a calibrated first question")

    # reply_events imports the engine factory only for normal generated replies;
    # a calibrated opening should complete before it reaches that code path.
    async def run():
        events = [
            event async for event in reply_events(
                [{"role": "system", "content": "prompt"}],
                opening_text="What consistency requirement would you clarify first?",
            )
        ]
        assert events == [ReplyToken("What consistency requirement would you clarify first?")]

    asyncio.run(run())


if __name__ == "__main__":
    test_calibrated_opening_skips_the_model_and_is_spoken_verbatim()
    print("1/1 passed")
