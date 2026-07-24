"""Pure regression checks for the OpenAI Responses adapter — no API key or network."""

from __future__ import annotations

import asyncio

import config
from director import ACTION_SCHEMA, DirectorState, decide
from llm.openai_api import OpenAILLM


def test_responses_payload_is_private_and_allows_per_call_reasoning_effort():
    llm = OpenAILLM.__new__(OpenAILLM)  # _payload does not require an API key
    payload = llm._payload(
        [{"role": "system", "content": "direct the interview"}],
        max_tokens=512,
        format_schema=ACTION_SCHEMA,
        reasoning_effort="none",
    )

    assert payload["store"] is False
    assert payload["max_output_tokens"] == 512
    assert payload["reasoning"] == {"effort": "none"}
    assert payload["input"][0]["role"] == "developer"
    assert payload["text"]["format"]["strict"] is True


def test_director_uses_its_own_reasoning_effort():
    class FakeLLM:
        seen_effort: str | None = None

        async def reply(self, _messages, **kwargs):
            self.seen_effort = kwargs["reasoning_effort"]
            return '{"action":"probe_deeper","detail":"ask for a concrete example"}'

    async def run():
        llm = FakeLLM()
        actions = [
            action
            async for action in decide(
                llm,
                [{"role": "user", "content": "I improved a pipeline."}],
                DirectorState(),
            )
        ]
        assert actions[-1]["action"] == "probe_deeper"
        assert llm.seen_effort == config.OPENAI_DIRECTOR_REASONING_EFFORT

    asyncio.run(run())


def _run_all():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"\n{len(tests)}/{len(tests)} passed")


if __name__ == "__main__":
    _run_all()
