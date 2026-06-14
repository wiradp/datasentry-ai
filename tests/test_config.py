from pathlib import Path

import pytest

from src.config import (
    AnalysisFocus,
    ConfigurationError,
    DEFAULT_GEMINI_MODEL,
    ExplanationStyle,
    GeminiConfig,
    available_analysis_focuses,
    available_explanation_styles,
    load_gemini_config,
    parse_analysis_focus,
    parse_explanation_style,
)


def test_defaults_can_load_without_api_key() -> None:
    config = load_gemini_config(
        env_file=None,
        environ={},
    )

    assert config.api_key is None
    assert config.has_api_key is False
    assert config.model == DEFAULT_GEMINI_MODEL
    assert config.temperature == 0.2
    assert config.max_output_tokens == 2048
    assert config.max_conversation_messages == 12
    assert config.max_tool_rounds == 5
    assert config.request_timeout_seconds == 60.0
    assert config.default_explanation_style is (
        ExplanationStyle.BUSINESS_FRIENDLY
    )
    assert config.default_analysis_focus is (
        AnalysisFocus.GENERAL_DATA_QUALITY
    )


def test_require_api_key_rejects_missing_key() -> None:
    with pytest.raises(
        ConfigurationError,
        match="API key is not configured",
    ) as error:
        load_gemini_config(
            env_file=None,
            environ={},
            require_api_key=True,
        )

    assert error.value.code == "missing_api_key"


def test_reads_gemini_api_key() -> None:
    config = load_gemini_config(
        env_file=None,
        environ={
            "GEMINI_API_KEY": " gemini-secret ",
        },
        require_api_key=True,
    )

    assert config.require_api_key() == "gemini-secret"
    assert config.api_key_source == "GEMINI_API_KEY"


def test_google_api_key_takes_precedence() -> None:
    config = load_gemini_config(
        env_file=None,
        environ={
            "GEMINI_API_KEY": "gemini-secret",
            "GOOGLE_API_KEY": "google-secret",
        },
    )

    assert config.require_api_key() == "google-secret"
    assert config.api_key_source == "GOOGLE_API_KEY"
    assert len(config.configuration_warnings) == 1
    assert "takes precedence" in (
        config.configuration_warnings[0]
    )


def test_secret_is_not_exposed_in_repr_or_public_summary() -> None:
    config = GeminiConfig(
        api_key="very-secret-value",
        api_key_source="GEMINI_API_KEY",
    )

    assert "very-secret-value" not in repr(config)

    summary = config.public_summary()

    assert "api_key" not in summary
    assert summary["api_key_configured"] is True
    assert "very-secret-value" not in str(summary)


def test_parses_runtime_settings() -> None:
    config = load_gemini_config(
        env_file=None,
        environ={
            "GEMINI_MODEL": "gemini-2.5-flash",
            "GEMINI_TEMPERATURE": "0.1",
            "GEMINI_MAX_OUTPUT_TOKENS": "4096",
            "GEMINI_MAX_CONVERSATION_MESSAGES": "20",
            "GEMINI_MAX_TOOL_ROUNDS": "4",
            "GEMINI_REQUEST_TIMEOUT_SECONDS": "90",
            "DATASENTRY_DEFAULT_EXPLANATION_STYLE": (
                "technical"
            ),
            "DATASENTRY_DEFAULT_ANALYSIS_FOCUS": (
                "ml-readiness"
            ),
        },
    )

    assert config.temperature == 0.1
    assert config.max_output_tokens == 4096
    assert config.max_conversation_messages == 20
    assert config.max_tool_rounds == 4
    assert config.request_timeout_seconds == 90.0
    assert config.default_explanation_style is (
        ExplanationStyle.TECHNICAL
    )
    assert config.default_analysis_focus is (
        AnalysisFocus.MACHINE_LEARNING_READINESS
    )


def test_runtime_environment_overrides_dotenv(
    tmp_path: Path,
) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            [
                "GEMINI_MODEL=from-dotenv",
                "GEMINI_TEMPERATURE=0.8",
            ]
        ),
        encoding="utf-8",
    )

    config = load_gemini_config(
        env_file=env_path,
        environ={
            "GEMINI_MODEL": "from-runtime",
        },
    )

    assert config.model == "from-runtime"
    assert config.temperature == 0.8


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (
            "beginner",
            ExplanationStyle.BEGINNER_FRIENDLY,
        ),
        (
            "business friendly",
            ExplanationStyle.BUSINESS_FRIENDLY,
        ),
        (
            "TECH",
            ExplanationStyle.TECHNICAL,
        ),
    ],
)
def test_explanation_style_aliases(
    value: str,
    expected: ExplanationStyle,
) -> None:
    assert parse_explanation_style(value) is expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (
            "general",
            AnalysisFocus.GENERAL_DATA_QUALITY,
        ),
        (
            "data quality",
            AnalysisFocus.GENERAL_DATA_QUALITY,
        ),
        (
            "ml",
            AnalysisFocus.MACHINE_LEARNING_READINESS,
        ),
    ],
)
def test_analysis_focus_aliases(
    value: str,
    expected: AnalysisFocus,
) -> None:
    assert parse_analysis_focus(value) is expected


def test_rejects_invalid_explanation_style() -> None:
    with pytest.raises(
        ConfigurationError,
        match="Unsupported explanation style",
    ):
        parse_explanation_style("academic")


def test_rejects_invalid_analysis_focus() -> None:
    with pytest.raises(
        ConfigurationError,
        match="Unsupported analysis focus",
    ):
        parse_analysis_focus("financial-audit")


def test_rejects_invalid_numeric_environment_values() -> None:
    with pytest.raises(
        ConfigurationError,
        match="must be a valid number",
    ):
        load_gemini_config(
            env_file=None,
            environ={
                "GEMINI_TEMPERATURE": "not-a-number",
            },
        )

    with pytest.raises(
        ConfigurationError,
        match="must be a valid integer",
    ):
        load_gemini_config(
            env_file=None,
            environ={
                "GEMINI_MAX_TOOL_ROUNDS": "2.5",
            },
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"temperature": -0.1},
        {"temperature": 2.1},
        {"max_output_tokens": 0},
        {"max_output_tokens": 65_537},
        {"max_conversation_messages": 1},
        {"max_conversation_messages": 101},
        {"max_tool_rounds": 0},
        {"max_tool_rounds": 11},
        {"request_timeout_seconds": 4.9},
        {"request_timeout_seconds": 300.1},
    ],
)
def test_rejects_out_of_range_settings(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(ConfigurationError):
        GeminiConfig(**kwargs)


def test_generation_kwargs_are_sdk_ready() -> None:
    config = GeminiConfig(
        temperature=0.15,
        max_output_tokens=3072,
    )

    assert config.generation_kwargs() == {
        "temperature": 0.15,
        "max_output_tokens": 3072,
    }


def test_available_ui_options_are_stable() -> None:
    assert available_explanation_styles() == (
        "beginner-friendly",
        "business-friendly",
        "technical",
    )

    assert available_analysis_focuses() == (
        "general-data-quality",
        "machine-learning-readiness",
    )
