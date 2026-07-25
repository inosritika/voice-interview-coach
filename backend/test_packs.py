"""Focused unit tests for the topic/company pack system (packs.py,
companies.py) and how prompts.py composes them. Pure — no Ollama, no network.

Runnable either as `python -m pytest test_packs.py` or directly as
`python test_packs.py` (plain asserts, prints OK on success) — matching the
plain-asserts style the rest of this project's ad-hoc checks use.
"""

from __future__ import annotations

from companies import COMPANY_PROFILES, get_company
from debrief import SCORING_SYSTEM
from packs import PACKS, get_pack, opening_question
from prompts import SHARED_SPOKEN_RULES, build_director_prompt, build_system_prompt


def test_get_pack_returns_right_pack_by_key():
    assert get_pack("behavioral").key == "behavioral"
    assert get_pack("dsa").key == "dsa"
    assert get_pack("ml").key == "ml"
    assert get_pack("system_design").key == "system_design"


def test_get_pack_unknown_falls_back_to_behavioral():
    assert get_pack("underwater-basket-weaving").key == "behavioral"
    assert get_pack("").key == "behavioral"
    assert get_pack(None).key == "behavioral"


def test_every_pack_has_required_nonempty_fields():
    assert set(PACKS) == {"behavioral", "dsa", "ml", "system_design"}
    for pack in PACKS.values():
        assert pack.persona.strip(), f"{pack.key} persona is empty"
        assert pack.director_guidance.strip(), f"{pack.key} director_guidance is empty"
        assert pack.opening.strip(), f"{pack.key} opening is empty"
        assert isinstance(pack.rubric, tuple)
        assert len(pack.rubric) == 4, f"{pack.key} rubric must have exactly 4 dimensions"
        for dim in pack.rubric:
            assert isinstance(dim, tuple) and len(dim) == 2
            name, desc = dim
            assert name.strip() and desc.strip()


def test_get_company_fallback_to_generic():
    assert get_company("not-a-real-company").key == "generic"
    assert get_company("").key == "generic"
    assert get_company(None).key == "generic"


def test_generic_company_signal_block_is_empty():
    assert COMPANY_PROFILES["generic"].signal_block.strip() == ""


def test_all_five_companies_present():
    assert set(COMPANY_PROFILES) == {"openai", "anthropic", "google", "meta", "generic"}
    for key, profile in COMPANY_PROFILES.items():
        if key == "generic":
            continue
        assert profile.signal_block.strip(), f"{key} signal_block is empty"


def test_build_system_prompt_composes_persona_company_and_shared_rules():
    prompt = build_system_prompt(
        "some JD", "some resume", "system_design", "anthropic", "senior"
    )
    sd_pack = get_pack("system_design")
    anthropic = get_company("anthropic")

    # The system_design persona domain guidance made it in.
    assert "SYSTEM DESIGN" in prompt
    assert sd_pack.opening in prompt
    # The anthropic company signal made it in.
    assert anthropic.signal_block in prompt
    # The shared spoken-delivery hard rules are still present regardless of topic.
    assert SHARED_SPOKEN_RULES in prompt
    # Senior difficulty calibration made it in.
    assert "senior / stretch" in prompt.lower() or "senior" in prompt.lower()


def test_build_system_prompt_defaults_still_behavioral():
    prompt = build_system_prompt("some JD", "some resume")
    behavioral = get_pack("behavioral")
    assert "BEHAVIORAL" in prompt
    assert behavioral.opening in prompt
    # Defaulting to generic company: no signal block text should leak in.
    for key, profile in COMPANY_PROFILES.items():
        if key == "generic":
            continue
        assert profile.signal_block not in prompt


def test_build_system_prompt_generic_company_injects_nothing_odd():
    prompt = build_system_prompt("jd", "resume", "behavioral", "generic")
    # No stray blank-paragraph artifact from an empty signal block: the two
    # newlines from persona + signal_block("") shouldn't produce a run of 3+.
    assert "\n\n\n\n" not in prompt


def test_build_director_prompt_injects_dsa_guidance():
    prompt = build_director_prompt("EVIDENCE NOTES SO FAR:\n- (none yet)", "dsa")
    dsa = get_pack("dsa")
    assert dsa.director_guidance in prompt
    # The flat action schema wording must still be intact regardless of topic.
    assert "note_evidence" in prompt
    assert "note_red_flag" in prompt
    assert "probe_deeper" in prompt
    assert "switch_topic" in prompt
    assert "end_interview" in prompt


def test_build_director_prompt_default_is_behavioral():
    prompt = build_director_prompt("EVIDENCE NOTES SO FAR:\n- (none yet)")
    behavioral = get_pack("behavioral")
    assert behavioral.director_guidance in prompt


def test_each_area_and_difficulty_gets_a_distinct_opening_contract():
    expected = {
        ("behavioral", "warmup"): "low-pressure STAR story",
        ("behavioral", "senior"): "high-stakes leadership story",
        ("dsa", "warmup"): "elementary array, string, or hash-map",
        ("dsa", "senior"): "genuinely hard problem",
        ("ml", "warmup"): "small concrete ML scenario",
        ("ml", "senior"): "production ML scenario",
        ("system_design", "warmup"): "modest scale",
        ("system_design", "senior"): "high-scale, failure-sensitive system",
    }
    for (area, difficulty), phrase in expected.items():
        prompt = build_system_prompt("JD", "Resume", area, "generic", difficulty)
        assert phrase in prompt, f"{area}/{difficulty} has no distinct opening contract"


def test_opening_questions_match_all_areas_and_difficulties():
    for area in PACKS:
        questions = [opening_question(area, difficulty) for difficulty in ("warmup", "standard", "senior")]
        assert len(set(questions)) == 3
        assert all(question.endswith("?") for question in questions)


def test_opening_question_varies_across_interviews():
    """Regression: the opener was one fixed string per slot, so every DSA interview
    started with the SAME problem. Each slot is now a pool picked at random."""
    for area in PACKS:
        for tier in ("warmup", "standard", "senior"):
            seen = {opening_question(area, tier) for _ in range(60)}
            assert len(seen) >= 2, f"{area}/{tier} never varies its opening question"


def test_opening_question_avoids_immediate_repeat():
    """A fresh interview must not open with the exact line the last one used."""
    prev = None
    for _ in range(200):
        q = opening_question("dsa", "standard")
        assert q != prev, "opening question repeated back-to-back"
        prev = q


def test_ml_rubric_drives_debrief_scoring_prompt():
    ml_rubric = get_pack("ml").rubric
    rubric_text = "\n".join(f"- {name}: {desc}" for name, desc in ml_rubric)
    dims_example = ",\n".join(
        f'    {{"name": "{name}", "score": <1-5>, "comment": "<one sentence, specific>"}}'
        for name, _desc in ml_rubric
    )
    rendered = SCORING_SYSTEM.format(label="Machine Learning", rubric=rubric_text, dims_example=dims_example)
    assert "Machine Learning" in rendered
    for name, _desc in ml_rubric:
        assert name in rendered
    # The literal JSON scaffold (escaped braces in the template) survived .format().
    assert '"overall": <integer 0-100>' in rendered


def _run_all():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    passed = 0
    for t in tests:
        t()
        passed += 1
        print(f"ok  {t.__name__}")
    print(f"\n{passed}/{len(tests)} passed")


if __name__ == "__main__":
    _run_all()
