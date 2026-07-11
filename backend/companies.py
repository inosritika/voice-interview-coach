"""Company signal profiles — what a given bar actually screens for and its
interview house-style, layered on TOP of the topic pack (packs.py).

Kept intentionally short and non-cartoonish: a paragraph of real signal, not a
caricature. `generic` contributes an empty block so nothing is injected when
the candidate hasn't named a target company.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CompanyProfile:
    key: str
    label: str
    signal_block: str


COMPANY_PROFILES: dict[str, CompanyProfile] = {
    "openai": CompanyProfile(
        key="openai",
        label="OpenAI",
        signal_block="This round is calibrated to OpenAI's bar: research-quality reasoning "
        "under ambiguity, genuine first-principles thinking over memorized answers, and "
        "pragmatic judgment about what actually matters for shipping. Reward candidates who "
        "reason out loud and revise when they spot a flaw in their own approach.",
    ),
    "anthropic": CompanyProfile(
        key="anthropic",
        label="Anthropic",
        signal_block="This round is calibrated to Anthropic's bar: careful, well-reasoned "
        "thinking, intellectual honesty about trade-offs and uncertainty, and a preference for "
        "substance over polish. Reward candidates who reason carefully and are upfront about "
        "what they don't know, over ones who sound confident but hand-wave.",
    ),
    "google": CompanyProfile(
        key="google",
        label="Google",
        signal_block="This round is calibrated to Google's bar: rigor at scale, clean "
        "structured problem-solving, and general cognitive ability ('Googleyness') alongside "
        "the domain skill. Reward candidates who methodically break a problem down and reason "
        "about how a solution holds up under real scale and constraints.",
    ),
    "meta": CompanyProfile(
        key="meta",
        label="Meta",
        signal_block="This round is calibrated to Meta's bar: a strong bias for impact and "
        "shipping, product sense alongside technical depth, and moving fast without losing "
        "rigor. Reward candidates who connect their approach back to concrete outcomes and "
        "user or business impact, not just technical correctness in the abstract.",
    ),
    "generic": CompanyProfile(
        key="generic",
        label="Generic",
        signal_block="",
    ),
}


def get_company(key: str | None) -> CompanyProfile:
    """Fall back to the neutral generic profile for an unknown or empty key —
    never raises."""
    return COMPANY_PROFILES.get((key or "").strip(), COMPANY_PROFILES["generic"])
