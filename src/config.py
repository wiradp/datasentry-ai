from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Mapping

from dotenv import dotenv_values


class ConfigurationError(ValueError):
    """Raised when application configuration is missing or invalid."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "configuration_error",
    ) -> None:
        super().__init__(message)
        self.code = code


class ExplanationStyle(str, Enum):
    """Supported response styles for the Gemini explanation layer."""

    BEGINNER_FRIENDLY = "beginner-friendly"
    BUSINESS_FRIENDLY = "business-friendly"
    TECHNICAL = "technical"


class AnalysisFocus(str, Enum):
    """Supported audit perspectives exposed in the UI."""

    GENERAL_DATA_QUALITY = "general-data-quality"
    MACHINE_LEARNING_READINESS = "machine-learning-readiness"


DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_OUTPUT_TOKENS = 2048
DEFAULT_MAX_CONVERSATION_MESSAGES = 12
DEFAULT_MAX_TOOL_ROUNDS = 5
DEFAULT_REQUEST_TIMEOUT_SECONDS = 60.0


_EXPLANATION_STYLE_ALIASES = {
    "beginner": ExplanationStyle.BEGINNER_FRIENDLY,
    "beginner-friendly": ExplanationStyle.BEGINNER_FRIENDLY,
    "beginnerfriendly": ExplanationStyle.BEGINNER_FRIENDLY,
    "simple": ExplanationStyle.BEGINNER_FRIENDLY,
    "business": ExplanationStyle.BUSINESS_FRIENDLY,
    "business-friendly": ExplanationStyle.BUSINESS_FRIENDLY,
    "businessfriendly": ExplanationStyle.BUSINESS_FRIENDLY,
    "stakeholder": ExplanationStyle.BUSINESS_FRIENDLY,
    "technical": ExplanationStyle.TECHNICAL,
    "tech": ExplanationStyle.TECHNICAL,
}

_ANALYSIS_FOCUS_ALIASES = {
    "general": AnalysisFocus.GENERAL_DATA_QUALITY,
    "general-data-quality": AnalysisFocus.GENERAL_DATA_QUALITY,
    "generaldataquality": AnalysisFocus.GENERAL_DATA_QUALITY,
    "data-quality": AnalysisFocus.GENERAL_DATA_QUALITY,
    "machine-learning": AnalysisFocus.MACHINE_LEARNING_READINESS,
    "machine-learning-readiness": AnalysisFocus.MACHINE_LEARNING_READINESS,
    "machinelearningreadiness": AnalysisFocus.MACHINE_LEARNING_READINESS,
    "ml": AnalysisFocus.MACHINE_LEARNING_READINESS,
    "ml-readiness": AnalysisFocus.MACHINE_LEARNING_READINESS,
}


def _normalize_choice(value: str) -> str:
    return (
        value.strip()
        .casefold()
        .replace("_", "-")
        .replace(" ", "-")
    )


def parse_explanation_style(
    value: str | ExplanationStyle,
) -> ExplanationStyle:
    """Parse a user or environment value into ExplanationStyle."""

    if isinstance(value, ExplanationStyle):
        return value

    normalized = _normalize_choice(str(value))
    compact = normalized.replace("-", "")

    style = (
        _EXPLANATION_STYLE_ALIASES.get(normalized)
        or _EXPLANATION_STYLE_ALIASES.get(compact)
    )

    if style is None:
        allowed = ", ".join(
            style.value for style in ExplanationStyle
        )
        raise ConfigurationError(
            (
                f"Unsupported explanation style: {value!r}. "
                f"Allowed values: {allowed}."
            ),
            code="invalid_explanation_style",
        )

    return style


def parse_analysis_focus(
    value: str | AnalysisFocus,
) -> AnalysisFocus:
    """Parse a user or environment value into AnalysisFocus."""

    if isinstance(value, AnalysisFocus):
        return value

    normalized = _normalize_choice(str(value))
    compact = normalized.replace("-", "")

    focus = (
        _ANALYSIS_FOCUS_ALIASES.get(normalized)
        or _ANALYSIS_FOCUS_ALIASES.get(compact)
    )

    if focus is None:
        allowed = ", ".join(
            focus.value for focus in AnalysisFocus
        )
        raise ConfigurationError(
            (
                f"Unsupported analysis focus: {value!r}. "
                f"Allowed values: {allowed}."
            ),
            code="invalid_analysis_focus",
        )

    return focus


def available_explanation_styles() -> tuple[str, ...]:
    """Return stable values suitable for a Streamlit selectbox."""

    return tuple(style.value for style in ExplanationStyle)


def available_analysis_focuses() -> tuple[str, ...]:
    """Return stable values suitable for a Streamlit selectbox."""

    return tuple(focus.value for focus in AnalysisFocus)


@dataclass(frozen=True)
class GeminiConfig:
    """Validated Gemini and conversation settings."""

    api_key: str | None = field(
        default=None,
        repr=False,
    )
    api_key_source: str | None = None
    model: str = DEFAULT_GEMINI_MODEL
    temperature: float = DEFAULT_TEMPERATURE
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS
    max_conversation_messages: int = (
        DEFAULT_MAX_CONVERSATION_MESSAGES
    )
    max_tool_rounds: int = DEFAULT_MAX_TOOL_ROUNDS
    request_timeout_seconds: float = (
        DEFAULT_REQUEST_TIMEOUT_SECONDS
    )
    default_explanation_style: ExplanationStyle = (
        ExplanationStyle.BUSINESS_FRIENDLY
    )
    default_analysis_focus: AnalysisFocus = (
        AnalysisFocus.GENERAL_DATA_QUALITY
    )
    configuration_warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        cleaned_model = self.model.strip()

        if not cleaned_model:
            raise ConfigurationError(
                "Gemini model cannot be empty.",
                code="empty_model",
            )

        if any(character.isspace() for character in cleaned_model):
            raise ConfigurationError(
                "Gemini model cannot contain whitespace.",
                code="invalid_model",
            )

        object.__setattr__(self, "model", cleaned_model)

        if not 0.0 <= self.temperature <= 2.0:
            raise ConfigurationError(
                "GEMINI_TEMPERATURE must be between 0.0 and 2.0.",
                code="temperature_out_of_range",
            )

        if not 1 <= self.max_output_tokens <= 65_536:
            raise ConfigurationError(
                (
                    "GEMINI_MAX_OUTPUT_TOKENS must be between "
                    "1 and 65536."
                ),
                code="max_output_tokens_out_of_range",
            )

        if not 2 <= self.max_conversation_messages <= 100:
            raise ConfigurationError(
                (
                    "GEMINI_MAX_CONVERSATION_MESSAGES must be between "
                    "2 and 100."
                ),
                code="conversation_limit_out_of_range",
            )

        if not 1 <= self.max_tool_rounds <= 10:
            raise ConfigurationError(
                (
                    "GEMINI_MAX_TOOL_ROUNDS must be between "
                    "1 and 10."
                ),
                code="tool_round_limit_out_of_range",
            )

        if not 5.0 <= self.request_timeout_seconds <= 300.0:
            raise ConfigurationError(
                (
                    "GEMINI_REQUEST_TIMEOUT_SECONDS must be between "
                    "5 and 300."
                ),
                code="timeout_out_of_range",
            )

        object.__setattr__(
            self,
            "default_explanation_style",
            parse_explanation_style(
                self.default_explanation_style
            ),
        )
        object.__setattr__(
            self,
            "default_analysis_focus",
            parse_analysis_focus(
                self.default_analysis_focus
            ),
        )

        cleaned_api_key = (
            self.api_key.strip()
            if isinstance(self.api_key, str)
            else None
        )

        object.__setattr__(
            self,
            "api_key",
            cleaned_api_key or None,
        )

    @property
    def has_api_key(self) -> bool:
        return bool(self.api_key)

    def require_api_key(self) -> str:
        """Return the key or raise a user-safe configuration error."""

        if not self.api_key:
            raise ConfigurationError(
                (
                    "Gemini API key is not configured. Set "
                    "GEMINI_API_KEY in .env or the runtime environment."
                ),
                code="missing_api_key",
            )

        return self.api_key

    def generation_kwargs(self) -> dict[str, int | float]:
        """Return settings later passed to GenerateContentConfig."""

        return {
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
        }

    def public_summary(self) -> dict[str, object]:
        """Return configuration details without exposing the secret key."""

        return {
            "api_key_configured": self.has_api_key,
            "api_key_source": self.api_key_source,
            "model": self.model,
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
            "max_conversation_messages": (
                self.max_conversation_messages
            ),
            "max_tool_rounds": self.max_tool_rounds,
            "request_timeout_seconds": (
                self.request_timeout_seconds
            ),
            "default_explanation_style": (
                self.default_explanation_style.value
            ),
            "default_analysis_focus": (
                self.default_analysis_focus.value
            ),
            "configuration_warnings": list(
                self.configuration_warnings
            ),
        }


def _load_environment(
    *,
    env_file: str | Path | None,
    environ: Mapping[str, str] | None,
) -> dict[str, str]:
    """
    Load .env values, then allow runtime environment values to override.

    Passing environ explicitly keeps unit tests deterministic.
    """

    values: dict[str, str] = {}

    if env_file is not None:
        env_path = Path(env_file)

        if env_path.exists():
            file_values = dotenv_values(env_path)

            values.update(
                {
                    str(key): str(value)
                    for key, value in file_values.items()
                    if value is not None
                }
            )

    runtime_environment = (
        os.environ
        if environ is None
        else environ
    )

    values.update(
        {
            str(key): str(value)
            for key, value in runtime_environment.items()
            if value is not None
        }
    )

    return values


def _parse_float(
    environment: Mapping[str, str],
    name: str,
    default: float,
) -> float:
    raw_value = environment.get(name)

    if raw_value is None or not raw_value.strip():
        return default

    try:
        return float(raw_value)
    except ValueError as exc:
        raise ConfigurationError(
            f"{name} must be a valid number.",
            code="invalid_float_setting",
        ) from exc


def _parse_integer(
    environment: Mapping[str, str],
    name: str,
    default: int,
) -> int:
    raw_value = environment.get(name)

    if raw_value is None or not raw_value.strip():
        return default

    try:
        return int(raw_value)
    except ValueError as exc:
        raise ConfigurationError(
            f"{name} must be a valid integer.",
            code="invalid_integer_setting",
        ) from exc


def load_gemini_config(
    *,
    env_file: str | Path | None = ".env",
    environ: Mapping[str, str] | None = None,
    require_api_key: bool = False,
) -> GeminiConfig:
    """
    Load and validate Gemini settings.

    Runtime environment values override values from the .env file.
    GOOGLE_API_KEY follows the Google Gen AI SDK precedence rule over
    GEMINI_API_KEY when both are present.
    """

    environment = _load_environment(
        env_file=env_file,
        environ=environ,
    )

    google_api_key = environment.get(
        "GOOGLE_API_KEY",
        "",
    ).strip()
    gemini_api_key = environment.get(
        "GEMINI_API_KEY",
        "",
    ).strip()

    warnings: list[str] = []

    if google_api_key and gemini_api_key:
        warnings.append(
            (
                "Both GOOGLE_API_KEY and GEMINI_API_KEY are set. "
                "GOOGLE_API_KEY takes precedence."
            )
        )

    if google_api_key:
        api_key = google_api_key
        api_key_source = "GOOGLE_API_KEY"
    elif gemini_api_key:
        api_key = gemini_api_key
        api_key_source = "GEMINI_API_KEY"
    else:
        api_key = None
        api_key_source = None

    config = GeminiConfig(
        api_key=api_key,
        api_key_source=api_key_source,
        model=environment.get(
            "GEMINI_MODEL",
            DEFAULT_GEMINI_MODEL,
        ),
        temperature=_parse_float(
            environment,
            "GEMINI_TEMPERATURE",
            DEFAULT_TEMPERATURE,
        ),
        max_output_tokens=_parse_integer(
            environment,
            "GEMINI_MAX_OUTPUT_TOKENS",
            DEFAULT_MAX_OUTPUT_TOKENS,
        ),
        max_conversation_messages=_parse_integer(
            environment,
            "GEMINI_MAX_CONVERSATION_MESSAGES",
            DEFAULT_MAX_CONVERSATION_MESSAGES,
        ),
        max_tool_rounds=_parse_integer(
            environment,
            "GEMINI_MAX_TOOL_ROUNDS",
            DEFAULT_MAX_TOOL_ROUNDS,
        ),
        request_timeout_seconds=_parse_float(
            environment,
            "GEMINI_REQUEST_TIMEOUT_SECONDS",
            DEFAULT_REQUEST_TIMEOUT_SECONDS,
        ),
        default_explanation_style=parse_explanation_style(
            environment.get(
                "DATASENTRY_DEFAULT_EXPLANATION_STYLE",
                ExplanationStyle.BUSINESS_FRIENDLY.value,
            )
        ),
        default_analysis_focus=parse_analysis_focus(
            environment.get(
                "DATASENTRY_DEFAULT_ANALYSIS_FOCUS",
                AnalysisFocus.GENERAL_DATA_QUALITY.value,
            )
        ),
        configuration_warnings=tuple(warnings),
    )

    if require_api_key:
        config.require_api_key()

    return config
