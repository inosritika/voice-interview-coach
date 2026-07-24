"""Session persistence + cross-session candidate memory.

Two small, connected features with one storage layer:

1. Every finished interview is saved to `data/sessions/<id>.json` — the full
   transcript, delivery metrics, debrief, and the director's evidence notebook.
   This is what the MCP server (mcp_server.py) exposes to outside agents.

2. A per-candidate profile at `data/profiles/<slug>.json` accumulates across
   interviews (dates, overall scores, the improvements that keep coming up).
   At setup, the profile is injected into the system prompt — so session 5
   knows you've struggled with quantifying impact since session 1. This is the
   memory-architecture lesson: decide what to store (small, structured,
   judgment-bearing), when to recall (setup), and how to present it (as topics
   to revisit, never as facts to assert about the candidate).

Plain JSON files, no database: single-user local app, and files keep every
artifact inspectable — you can open your own interview history in an editor.
"""

from __future__ import annotations

import json
import logging
import pathlib
import re
import time
import uuid

log = logging.getLogger("interview-coach")

DATA_DIR = pathlib.Path(__file__).resolve().parent / "data"
SESSIONS_DIR = DATA_DIR / "sessions"
PROFILES_DIR = DATA_DIR / "profiles"


def _slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return s or "anonymous"


# ---- sessions ----------------------------------------------------------------

def save_session(
    *,
    candidate: str,
    role: str,
    messages: list[dict],
    turn_stats: list[dict],
    debrief: dict,
    director_notes: str | None,
    interview_type: str = "behavioral",
    company: str = "generic",
) -> str:
    """Persist one finished interview. Returns the session id. Never raises —
    losing a save must not break the debrief the user is looking at."""
    session_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    record = {
        "id": session_id,
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "candidate": candidate or "anonymous",
        "role": role,
        "interview_type": interview_type,
        "company": company,
        # transcript only — the system prompt is instructions, not conversation
        "transcript": [m for m in messages if m["role"] in ("user", "assistant")],
        "turn_stats": turn_stats,
        "debrief": debrief,
        "director_notes": director_notes,
    }
    try:
        SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        path = SESSIONS_DIR / f"{session_id}.json"
        path.write_text(json.dumps(record, indent=2))
        log.info("saved session %s (%d transcript messages)", session_id, len(record["transcript"]))
    except Exception:  # noqa: BLE001
        log.exception("failed to save session")
    return session_id


def list_sessions() -> list[dict]:
    """Newest-first summaries of all saved interviews."""
    if not SESSIONS_DIR.exists():
        return []
    out = []
    for path in sorted(SESSIONS_DIR.glob("*.json"), reverse=True):
        try:
            r = json.loads(path.read_text())
            out.append({
                "id": r["id"],
                "saved_at": r["saved_at"],
                "candidate": r.get("candidate", "anonymous"),
                "role": r.get("role", ""),
                "answers": r.get("debrief", {}).get("metrics", {}).get("answers"),
                "overall": r.get("debrief", {}).get("overall"),
                "headline": r.get("debrief", {}).get("headline", ""),
            })
        except Exception:  # noqa: BLE001 — one corrupt file must not hide the rest
            log.warning("skipping unreadable session file %s", path.name)
    return out


def load_session(session_id: str) -> dict | None:
    # The id is used as a filename — refuse anything path-like.
    if not re.fullmatch(r"[A-Za-z0-9-]+", session_id or ""):
        return None
    path = SESSIONS_DIR / f"{session_id}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:  # noqa: BLE001
        return None


# ---- candidate profiles (cross-session memory) --------------------------------

def update_profile(candidate: str, debrief: dict) -> None:
    """Fold one interview's outcome into the candidate's profile. Stores only
    small, structured judgment: date, score, headline, improvements."""
    if not candidate:
        return
    entry = {
        "date": time.strftime("%Y-%m-%d"),
        "overall": debrief.get("overall"),
        "headline": debrief.get("headline", ""),
        "improvements": debrief.get("improvements", [])[:3],
    }
    try:
        PROFILES_DIR.mkdir(parents=True, exist_ok=True)
        path = PROFILES_DIR / f"{_slug(candidate)}.json"
        profile = json.loads(path.read_text()) if path.exists() else {
            "candidate": candidate, "sessions": [],
        }
        profile["sessions"].append(entry)
        profile["sessions"] = profile["sessions"][-10:]  # keep the last 10
        path.write_text(json.dumps(profile, indent=2))
        log.info("updated profile for %r (%d sessions)", candidate, len(profile["sessions"]))
    except Exception:  # noqa: BLE001
        log.exception("failed to update profile")


def load_profile(candidate: str) -> dict | None:
    """The raw cross-session profile for a candidate, or None."""
    if not candidate:
        return None
    path = PROFILES_DIR / f"{_slug(candidate)}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:  # noqa: BLE001
        return None


# Coaching notes are free text and worded differently every time, so matching on
# the words themselves clusters nothing ("quantify the metrics" vs "provide
# specific numbers" are the same weak point). Instead we bucket each note into a
# THEME by keyword — deterministic, no NLP dependency, and it's the grouping a
# candidate actually wants ("you keep getting told to quantify your impact").
_THEMES: tuple[tuple[str, str], ...] = (
    ("Quantify your impact", r"quantif|number|metric|figure|measurab|percentage|how much|magnitude|scale of"),
    ("Use a clear STAR structure", r"\bstar\b|structur|situation|rambl|organiz|hard to follow"),
    ("Be concrete, not generic", r"specific|concrete|example|vague|generic|detail"),
    ("Show personal ownership", r"ownership|your role|personally|took the lead|what you did|credit"),
    ("Explain trade-offs and reasoning", r"trade[- ]?off|justif|reasoning|why you|complexity|defend"),
    ("Engage with the question", r"deflect|skip|avoid|refus|non-?answer|did ?n.t answer|engage|abandon"),
    ("Cover edge cases", r"edge case|boundary|corner case|empty|null|duplicat"),
    ("Prepare stories in advance", r"prepare|rehears|ahead of time|before any interview|practice"),
)


def _theme_of(text: str) -> str:
    low = (text or "").lower()
    for name, pattern in _THEMES:
        if re.search(pattern, low):
            return name
    return "Other feedback"


def recurring_improvements(profile: dict, top: int = 8) -> list[dict]:
    """The weak points that keep coming back, most-repeated first. Counted once
    per session, so a single wordy debrief can't inflate a theme."""
    counts: dict[str, dict] = {}
    for s in profile.get("sessions", []):
        seen: set[str] = set()
        for imp in s.get("improvements", []):
            theme = _theme_of(imp)
            if theme in seen:
                continue
            seen.add(theme)
            entry = counts.setdefault(
                theme, {"theme": theme, "count": 0, "last": "", "examples": []}
            )
            entry["count"] += 1
            entry["last"] = max(entry["last"], s.get("date", ""))
            if len(entry["examples"]) < 2 and imp:
                entry["examples"].append(imp)
    return sorted(counts.values(), key=lambda d: (-d["count"], d["last"]), reverse=False)[:top]


def profile_prompt_block(candidate: str) -> str:
    """The recall side: past sessions rendered for the system prompt, or ""
    if this candidate has no history. Presented as coaching CONTEXT — topics
    and weaknesses to revisit — never as facts to assert back at them."""
    if not candidate:
        return ""
    path = PROFILES_DIR / f"{_slug(candidate)}.json"
    if not path.exists():
        return ""
    try:
        profile = json.loads(path.read_text())
    except Exception:  # noqa: BLE001
        return ""
    sessions = profile.get("sessions", [])
    if not sessions:
        return ""

    lines = []
    for s in sessions[-5:]:
        score = f"scored {s['overall']}/100" if s.get("overall") is not None else "unscored"
        fixes = "; ".join(s.get("improvements", [])) or "no notes"
        lines.append(f"- {s['date']}: {score} — needed work on: {fixes}")

    return (
        "\n\nPAST PRACTICE SESSIONS WITH THIS CANDIDATE (their private coaching "
        "history — use it to pick topics that exercise their known weak spots, "
        "and to notice improvement; NEVER recite it back at them or reference "
        "these sessions out loud):\n" + "\n".join(lines)
    )
