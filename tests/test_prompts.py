import pytest

from src.config import (
    AnalysisFocus,
    ExplanationStyle,
)
from src.prompts import (
    DEFAULT_ASSISTANT_NAME,
    DEFAULT_MAX_RECOMMENDATIONS,
    PROMPT_VERSION,
    PromptConfigurationError,
    PromptProfile,
    build_default_system_instruction,
    build_prompt_profile,
    build_system_instruction,
)


def _normalized(text: str) -> str:
    return " ".join(text.split())


@pytest.mark.parametrize(
    "style",
    list(ExplanationStyle),
)
@pytest.mark.parametrize(
    "focus",
    list(AnalysisFocus),
)
def test_builds_every_style_and_focus_combination(
    style: ExplanationStyle,
    focus: AnalysisFocus,
) -> None:
    prompt = build_system_instruction(
        explanation_style=style,
        analysis_focus=focus,
    )

    normalized = _normalized(prompt)

    assert prompt
    assert f"Prompt version: {PROMPT_VERSION}" in normalized
    assert f"Explanation style: {style.value}" in normalized
    assert f"Analysis focus: {focus.value}" in normalized


def test_grounding_rules_are_present() -> None:
    prompt = _normalized(
        build_default_system_instruction()
    )

    required_phrases = [
        "only source of truth",
        "Never invent row counts",
        "use the relevant tool before answering",
        "If the requested evidence is unavailable",
        "latest valid tool result",
        "Never claim that a tool was called",
    ]

    for phrase in required_phrases:
        assert phrase in prompt


def test_fact_indication_recommendation_are_separated() -> None:
    prompt = _normalized(
        build_default_system_instruction()
    )

    assert "FACT:" in prompt
    assert "INDICATION:" in prompt
    assert "RECOMMENDATION:" in prompt
    assert (
        "Do not present an indication or recommendation "
        "as a confirmed fact."
    ) in prompt


def test_read_only_rules_are_present() -> None:
    prompt = _normalized(
        build_default_system_instruction()
    )

    assert "The application audits and explains data" in prompt
    assert "it does not modify data" in prompt
    assert "Never claim to delete duplicates" in prompt
    assert "destructive data operations" in prompt


def test_quality_score_is_explicitly_heuristic() -> None:
    prompt = _normalized(
        build_default_system_instruction()
    )

    assert (
        "quality score is a heuristic prioritization aid"
        in prompt
    )
    assert "It is not a certification" in prompt
    assert (
        "A high score does not cancel high-severity issues."
        in prompt
    )


def test_beginner_style_has_beginner_specific_guidance() -> None:
    prompt = _normalized(
        build_system_instruction(
            explanation_style="beginner",
            analysis_focus="general",
        )
    )

    assert "Use plain language, short paragraphs" in prompt
    assert "define technical terms" in prompt
    assert "simple practical example" in prompt
    assert "Do not expose internal tool names" in prompt
    assert "raw tool fields" in prompt
    assert "Translate internal audit fields into user-facing labels" in prompt
    assert "Keep detailed tool evidence for technical-style answers" in prompt
    assert "Avoid internal uppercase status values" in prompt
    assert "looks mostly good but still needs review" in prompt
    assert "needs cleaning" in prompt
    assert "Prefer beginner-facing labels" in prompt
    assert "2-4 short paragraphs" in prompt
    assert "do not use audit terminology" in prompt
    assert "Say \"serious issues\" instead of \"high-severity issues\"" in prompt
    assert "14 serious issues" in prompt
    assert "Include one simple next step" in prompt


def test_business_style_has_business_specific_guidance() -> None:
    prompt = _normalized(
        build_system_instruction(
            explanation_style="business",
            analysis_focus="general",
        )
    )

    assert "business impact" in prompt
    assert "operational risk" in prompt
    assert "Do not invent financial impact" in prompt


def test_technical_style_has_technical_specific_guidance() -> None:
    prompt = _normalized(
        build_system_instruction(
            explanation_style="technical",
            analysis_focus="general",
        )
    )

    assert "include exact metrics" in prompt
    assert "thresholds, detection methods" in prompt
    assert "methodological nuance" in prompt


def test_general_focus_has_correct_boundaries() -> None:
    prompt = _normalized(
        build_system_instruction(
            explanation_style="technical",
            analysis_focus="general",
        )
    )

    assert "Prioritize completeness" in prompt
    assert "general analysis and reporting" in prompt
    assert (
        "Do not extend the assessment into model performance"
        in prompt
    )


def test_ml_focus_has_correct_boundaries() -> None:
    prompt = _normalized(
        build_system_instruction(
            explanation_style="technical",
            analysis_focus="ml",
        )
    )

    assert "feature usability for machine learning" in prompt
    assert "potential identifiers" in prompt
    assert "target-column concerns only when target evidence" in prompt
    assert "Do not claim model performance" in prompt


def test_response_contract_uses_requested_limit() -> None:
    prompt = _normalized(
        build_system_instruction(
            explanation_style="business",
            analysis_focus="general",
            max_recommendations=3,
        )
    )

    assert "no more than 3 recommended actions" in prompt


def test_missing_audit_behavior_is_present() -> None:
    prompt = _normalized(
        build_default_system_instruction()
    )

    assert "If no active audit report is available" in prompt
    assert "no dataset has been audited" in prompt
    assert "Do not answer with invented example metrics" in prompt


def test_prompt_accepts_aliases() -> None:
    profile = build_prompt_profile(
        explanation_style="stakeholder",
        analysis_focus="ml-readiness",
    )

    assert profile.explanation_style is (
        ExplanationStyle.BUSINESS_FRIENDLY
    )
    assert profile.analysis_focus is (
        AnalysisFocus.MACHINE_LEARNING_READINESS
    )


def test_profile_public_summary_is_stable() -> None:
    profile = PromptProfile()

    assert profile.public_summary() == {
        "prompt_version": PROMPT_VERSION,
        "assistant_name": DEFAULT_ASSISTANT_NAME,
        "explanation_style": "business-friendly",
        "analysis_focus": "general-data-quality",
        "max_recommendations": (
            DEFAULT_MAX_RECOMMENDATIONS
        ),
    }


def test_rejects_empty_assistant_name() -> None:
    with pytest.raises(
        PromptConfigurationError,
        match="assistant_name cannot be empty",
    ):
        PromptProfile(assistant_name="   ")


@pytest.mark.parametrize(
    "limit",
    [0, 11],
)
def test_rejects_invalid_recommendation_limit(
    limit: int,
) -> None:
    with pytest.raises(
        PromptConfigurationError,
        match="between 1 and 10",
    ):
        PromptProfile(max_recommendations=limit)


def test_prompt_is_deterministic() -> None:
    first = build_system_instruction(
        explanation_style="technical",
        analysis_focus="ml",
        assistant_name="DataSentry AI",
        max_recommendations=4,
    )
    second = build_system_instruction(
        explanation_style="technical",
        analysis_focus="ml",
        assistant_name="DataSentry AI",
        max_recommendations=4,
    )

    assert first == second


def test_prompt_contains_no_dataset_specific_metrics() -> None:
    prompt = build_default_system_instruction()

    assert "92.02" not in prompt
    assert "sample_dirty_customers.csv" not in prompt
    assert "annual_income" not in prompt
    assert "customer_id" not in prompt


def test_default_builder_matches_explicit_defaults() -> None:
    default_prompt = build_default_system_instruction()
    explicit_prompt = build_system_instruction(
        explanation_style=(
            ExplanationStyle.BUSINESS_FRIENDLY
        ),
        analysis_focus=(
            AnalysisFocus.GENERAL_DATA_QUALITY
        ),
    )

    assert default_prompt == explicit_prompt


def test_prompt_has_expected_response_sections() -> None:
    prompt = _normalized(
        build_default_system_instruction()
    )

    for heading in [
        "1. Assessment",
        "2. Evidence",
        "3. Recommended actions",
        "4. Limitations",
    ]:
        assert heading in prompt
