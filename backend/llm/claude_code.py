"""Claude via your local Claude Code subscription — no API key.

This is a THIRD LLM engine next to local Ollama (llm/local_ollama.py) and hosted
OpenAI (llm/openai_api.py). It drives the installed `claude` CLI through the
Claude Agent SDK (`claude_agent_sdk`), and — this is the whole point — auth is the
CLI's OWN logged-in session (your Pro/Max subscription). So unlike the OpenAI
adapter it needs NO ANTHROPIC_API_KEY; it needs the `claude` CLI installed and
signed in (run `claude`, and `/login` if the session ever expires).

Select it with LLM_ENGINE=claude (config.py). Everything downstream — the
director loop, compaction, debrief, the cascaded speak call — is unchanged,
because this still implements the same LLMEngine interface (llm/base.py).

Mapping our messages -> the SDK. The app hands us an OpenAI-style array (a pinned
system prompt, optionally a compaction summary, then alternating user/assistant
turns ending on the candidate). The SDK instead takes ONE prompt string plus a
`system_prompt`, so we:
  - lift every system message into `system_prompt`, and
  - render the conversation as a labeled INTERVIEWER/CANDIDATE transcript — the
    same rendering the director and eval harness already use.
Each call passes the FULL history (the Session owns memory + compaction), so we
deliberately do NOT use the SDK's own session resumption — every call is stateless.

Two interface knobs the CLI doesn't expose, and how we honor them anyway:
  - `format_schema` (constrained JSON for the director): the CLI has no grammar
    decoding, so we inject a strict "reply with ONLY this JSON" instruction. The
    director's _parse_action still validates + retries, so a stray token is caught.
  - `max_tokens` / `reasoning_effort`: no CLI equivalent, so they're ignored —
    the persona + speech_filter already bound spoken length.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    StreamEvent,
    TextBlock,
    query,
)

from config import CLAUDE_MODEL, CLAUDE_TIMEOUT_S
from llm.base import LLMEngine

log = logging.getLogger("interview-coach")


async def _iter_messages(prompt: str, options: ClaudeAgentOptions):
    """Drive one SDK query under a hard timeout, and ALWAYS close the underlying
    CLI subprocess when we're done — on success, on early break, or on timeout.
    Without the timeout a stalled request (flaky network / VPN) hangs the turn on
    the PROCESSING floor indefinitely; without the aclose a killed query leaks the
    `claude` subprocess."""
    agen = query(prompt=prompt, options=options)
    try:
        async with asyncio.timeout(CLAUDE_TIMEOUT_S):
            async for msg in agen:
                yield msg
    except TimeoutError as exc:  # asyncio.timeout fires this
        log.warning("claude: call exceeded %.0fs — aborting (network/VPN?)", CLAUDE_TIMEOUT_S)
        raise RuntimeError(
            f"Claude call timed out after {CLAUDE_TIMEOUT_S:.0f}s "
            "(check your network — a VPN can block the connection to Anthropic)"
        ) from exc
    finally:
        aclose = getattr(agen, "aclose", None)
        if aclose is not None:
            await aclose()


def _split(messages: list[dict]) -> tuple[str, str]:
    """Lift system messages into one system prompt; render user/assistant turns
    as a transcript. The trailing candidate line is the cue for the model's next
    interviewer turn (its persona/rules live in the system prompt)."""
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    lines = [
        f"{'INTERVIEWER' if m['role'] == 'assistant' else 'CANDIDATE'}: {m['content']}"
        for m in messages
        if m["role"] in ("user", "assistant")
    ]
    prompt = "\n".join(lines) if lines else "(the interview is just starting)"
    return system, prompt


def _options(system: str, format_schema: dict | None, stream: bool) -> ClaudeAgentOptions:
    if format_schema is not None:
        system += (
            "\n\nRespond with ONLY a single JSON object matching this schema, and "
            "NOTHING else — no prose, no explanation, no code fences:\n"
            + json.dumps(format_schema)
        )
    return ClaudeAgentOptions(
        system_prompt=system or None,
        model=CLAUDE_MODEL,
        allowed_tools=[],       # pure text generation — no tools, no agentic looping
        max_turns=1,            # one model turn per call; we drive the loop ourselves
        setting_sources=[],     # ignore CLAUDE.md / project settings for isolation + speed
        include_partial_messages=stream,
    )


def _delta_text(event: object) -> str:
    """Pull the text out of one streamed Anthropic event (content_block_delta)."""
    if not isinstance(event, dict):
        return ""
    if event.get("type") == "content_block_delta":
        delta = event.get("delta") or {}
        if delta.get("type") == "text_delta":
            return delta.get("text", "")
    return ""


class ClaudeCodeLLM(LLMEngine):
    async def reply(
        self,
        messages: list[dict],
        max_tokens: int | None = None,
        format_schema: dict | None = None,
        reasoning_effort: str | None = None,
    ) -> str:
        del max_tokens, reasoning_effort  # no CLI equivalent — see module docstring
        system, prompt = _split(messages)
        out: list[str] = []
        async for msg in _iter_messages(prompt, _options(system, format_schema, stream=False)):
            if isinstance(msg, AssistantMessage):
                out += [b.text for b in msg.content if isinstance(b, TextBlock)]
            elif isinstance(msg, ResultMessage) and msg.is_error:
                raise RuntimeError(msg.result or "Claude Code returned an error result")
        return "".join(out).strip()

    async def stream(self, messages: list[dict]) -> AsyncIterator[str]:
        system, prompt = _split(messages)
        saw_delta = False
        async for msg in _iter_messages(prompt, _options(system, None, stream=True)):
            if isinstance(msg, StreamEvent):
                text = _delta_text(msg.event)
                if text:
                    saw_delta = True
                    yield text
            elif isinstance(msg, AssistantMessage) and not saw_delta:
                # Fallback: if partial deltas weren't delivered, still emit the
                # full turn once so the interviewer is never silent.
                for b in msg.content:
                    if isinstance(b, TextBlock) and b.text:
                        yield b.text
            elif isinstance(msg, ResultMessage) and msg.is_error:
                raise RuntimeError(msg.result or "Claude Code returned an error result")
