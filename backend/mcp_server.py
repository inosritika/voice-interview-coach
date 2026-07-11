"""MCP server — the interview coach as a tool other agents can use.

The Model Context Protocol is the adapter pattern (learning-guide §6) one level
up: instead of a *code-level* interface between our pipeline and an engine,
this is a *protocol-level* interface between our whole app and any MCP-capable
agent (Claude Code, Claude Desktop, other harnesses). The server describes its
tools — name, typed schema, docstring — and clients discover and call them
without knowing anything about our internals.

What it exposes (read-only, over the sessions that storage.py persists):
    list_interviews()        -> summaries of every saved mock interview
    get_transcript(id)       -> the full conversation of one interview
    get_debrief(id)          -> scores, feedback, delivery metrics, notes
    get_progress(candidate)  -> score/improvement trajectory across sessions

Run it (stdio transport — the client spawns this process and speaks JSON-RPC
over stdin/stdout):
    ./.venv/bin/python mcp_server.py

Claude Code registration (from the backend directory):
    claude mcp add interview-coach -- ./.venv/bin/python mcp_server.py

Then an agent can be asked things like "look at my last three mock interviews
and tell me what keeps going wrong" — and it will call these tools.
"""

from __future__ import annotations

import json

from mcp.server.fastmcp import FastMCP

import storage

mcp = FastMCP(
    "interview-coach",
    instructions="Read-only access to a user's saved mock-interview sessions: "
    "transcripts, scored debriefs, delivery metrics, and cross-session progress. "
    "Use list_interviews first to discover session ids.",
)


@mcp.tool()
def list_interviews() -> str:
    """List all saved mock interviews, newest first: id, date, candidate, role,
    overall score (0-100), and the debrief's one-line verdict."""
    sessions = storage.list_sessions()
    if not sessions:
        return "No saved interviews yet. Sessions are saved when an interview ends with a debrief."
    return json.dumps(sessions, indent=2)


@mcp.tool()
def get_transcript(session_id: str) -> str:
    """The full conversation of one interview as INTERVIEWER/CANDIDATE lines.
    Get session ids from list_interviews."""
    record = storage.load_session(session_id)
    if record is None:
        return f"No session with id {session_id!r}. Call list_interviews for valid ids."
    lines = [
        f"{'INTERVIEWER' if m['role'] == 'assistant' else 'CANDIDATE'}: {m['content']}"
        for m in record.get("transcript", [])
    ]
    return "\n".join(lines) or "(empty transcript)"


@mcp.tool()
def get_debrief(session_id: str) -> str:
    """One interview's full review: rubric scores per dimension, strengths,
    improvements, delivery metrics (WPM, fillers, talk ratio), and the
    interviewer-agent's own evidence notes taken live during the interview."""
    record = storage.load_session(session_id)
    if record is None:
        return f"No session with id {session_id!r}. Call list_interviews for valid ids."
    out = {"debrief": record.get("debrief"), "director_notes": record.get("director_notes")}
    return json.dumps(out, indent=2)


@mcp.tool()
def get_progress(candidate: str) -> str:
    """A candidate's trajectory across all their saved interviews: date, overall
    score, and what needed work each time. Useful for 'what keeps going wrong'
    and 'am I improving' questions."""
    block = storage.profile_prompt_block(candidate)
    if not block:
        return f"No saved history for candidate {candidate!r}."
    # strip the interviewer-facing framing; the caller just wants the data
    lines = [l for l in block.splitlines() if l.startswith("- ")]
    return "\n".join(lines)


if __name__ == "__main__":
    mcp.run()  # stdio transport
