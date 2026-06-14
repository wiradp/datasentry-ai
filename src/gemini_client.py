from __future__ import annotations

import json
import math
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Sequence

import httpx
from google import genai
from google.genai import errors, types

from src.config import (
    AnalysisFocus,
    ConfigurationError,
    ExplanationStyle,
    GeminiConfig,
    parse_analysis_focus,
    parse_explanation_style,
)
from src.prompts import build_system_instruction
from src.tools import MAX_RESULT_LIMIT, AuditToolbox


class GeminiClientError(RuntimeError):
    """Raised for invalid local Gemini-client usage."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "gemini_client_error",
    ) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class GeminiClientState:
    """Secret-free state suitable for Streamlit diagnostics."""

    model: str
    history_message_count: int
    max_conversation_messages: int
    max_tool_rounds: int
    request_timeout_seconds: float
    explanation_style: str
    analysis_focus: str
    api_key_configured: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "history_message_count": self.history_message_count,
            "max_conversation_messages": (
                self.max_conversation_messages
            ),
            "max_tool_rounds": self.max_tool_rounds,
            "request_timeout_seconds": (
                self.request_timeout_seconds
            ),
            "explanation_style": self.explanation_style,
            "analysis_focus": self.analysis_focus,
            "api_key_configured": self.api_key_configured,
        }


_TOOL_DECLARATION_SPECS: tuple[dict[str, Any], ...] = (
    {
        "name": "get_dataset_overview",
        "description": (
            "Return audited file metadata, dataset dimensions, column "
            "groups by inferred type, and basic missing/duplicate rates."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
    {
        "name": "get_quality_summary",
        "description": (
            "Return the heuristic quality score, score band, final "
            "readiness status, component scores, issue counts, gates, "
            "limitations, and disclaimer."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
    {
        "name": "get_missing_value_report",
        "description": (
            "Return audited missing-value issues with optional column, "
            "minimum percentage, and result-limit filters."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": MAX_RESULT_LIMIT,
                    "description": (
                        "Maximum number of matching issues to return."
                    ),
                },
                "min_missing_percentage": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 100,
                    "description": (
                        "Only return columns at or above this missing "
                        "percentage."
                    ),
                },
                "column_name": {
                    "type": "string",
                    "description": (
                        "Optional exact or uniquely case-insensitive "
                        "audited column name."
                    ),
                },
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "get_duplicate_report",
        "description": (
            "Return the exact full-row duplicate assessment and evidence."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
    {
        "name": "get_column_quality_report",
        "description": (
            "Return normalized issues, recommendations, and scoring "
            "diagnostics for one audited column."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "column_name": {
                    "type": "string",
                    "description": (
                        "Exact or uniquely case-insensitive audited "
                        "column name."
                    ),
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": MAX_RESULT_LIMIT,
                    "description": (
                        "Maximum number of issues and recommendations "
                        "to return."
                    ),
                },
            },
            "required": ["column_name"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_priority_issues",
        "description": (
            "Return prioritized normalized audit issues, optionally "
            "filtered by severity, check name, or column."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": MAX_RESULT_LIMIT,
                },
                "minimum_severity": {
                    "type": "string",
                    "enum": [
                        "CRITICAL",
                        "HIGH",
                        "MEDIUM",
                        "LOW",
                    ],
                    "description": (
                        "Return this severity and all more severe issues."
                    ),
                },
                "check_name": {
                    "type": "string",
                    "description": (
                        "Optional normalized audit check name."
                    ),
                },
                "column_name": {
                    "type": "string",
                    "description": (
                        "Optional exact or uniquely case-insensitive "
                        "audited column name."
                    ),
                },
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "get_ml_readiness_report",
        "description": (
            "Return a bounded generic feature-quality view for machine "
            "learning preparation, with explicit capability limits."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": MAX_RESULT_LIMIT,
                },
            },
            "additionalProperties": False,
        },
    },
)


def _json_safe(value: Any) -> Any:
    """Convert SDK, enum, numpy-like, and mapping values to strict JSON."""

    if value is None:
        return None

    if isinstance(value, Enum):
        return _json_safe(value.value)

    if isinstance(value, (str, bool, int)):
        return value

    if isinstance(value, float):
        return value if math.isfinite(value) else None

    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]

    if hasattr(value, "model_dump"):
        try:
            return _json_safe(
                value.model_dump(
                    mode="json",
                    exclude_none=True,
                )
            )
        except (TypeError, ValueError):
            return _json_safe(
                value.model_dump(
                    exclude_none=True,
                )
            )

    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except (TypeError, ValueError):
            pass

    return str(value)


def _strict_json_copy(value: Any) -> Any:
    safe_value = _json_safe(value)
    payload = json.dumps(
        safe_value,
        ensure_ascii=False,
        allow_nan=False,
    )
    return json.loads(payload)


def build_audit_tool_declarations() -> types.Tool:
    """
    Build explicit read-only function declarations for Gemini.

    Explicit declarations keep execution under DataSentry's validated
    dispatcher instead of enabling SDK-managed Python function execution.
    """

    declarations = [
        types.FunctionDeclaration(
            name=spec["name"],
            description=spec["description"],
            parameters_json_schema=deepcopy(
                spec["parameters"]
            ),
        )
        for spec in _TOOL_DECLARATION_SPECS
    ]

    return types.Tool(
        function_declarations=declarations
    )


def _text_content(
    role: str,
    text: str,
) -> types.Content:
    return types.Content(
        role=role,
        parts=[types.Part(text=text)],
    )


def _extract_response_parts(
    response: Any,
) -> tuple[
    Any | None,
    list[Any],
    str,
    str | None,
]:
    """
    Extract first candidate content, function calls, text, and finish reason.
    """

    candidates = getattr(response, "candidates", None) or []

    if not candidates:
        return None, [], "", None

    candidate = candidates[0]
    content = getattr(candidate, "content", None)
    finish_reason_value = _json_safe(
        getattr(candidate, "finish_reason", None)
    )
    finish_reason = (
        str(finish_reason_value)
        if finish_reason_value is not None
        else None
    )

    if content is None:
        return None, [], "", finish_reason

    function_calls: list[Any] = []
    text_parts: list[str] = []

    for part in getattr(content, "parts", None) or []:
        function_call = getattr(
            part,
            "function_call",
            None,
        )

        if function_call is not None:
            function_calls.append(function_call)

        text = getattr(part, "text", None)
        thought = bool(getattr(part, "thought", False))

        if isinstance(text, str) and text.strip() and not thought:
            text_parts.append(text.strip())

    return (
        content,
        function_calls,
        "\n".join(text_parts).strip(),
        finish_reason,
    )


def _usage_metadata(response: Any) -> dict[str, Any]:
    """
    Return stable token fields across google-genai metadata variants.

    google-genai 2.8 uses candidates_token_count for generated tokens;
    the public result exposes the stable response_token_count name.
    """

    usage = getattr(response, "usage_metadata", None)

    if usage is None:
        return {}

    def read(*names: str) -> Any:
        for name in names:
            value = getattr(usage, name, None)

            if value is not None:
                return _json_safe(value)

        return None

    normalized = {
        "prompt_token_count": read(
            "prompt_token_count"
        ),
        "cached_content_token_count": read(
            "cached_content_token_count"
        ),
        "response_token_count": read(
            "response_token_count",
            "candidates_token_count",
        ),
        "tool_use_prompt_token_count": read(
            "tool_use_prompt_token_count"
        ),
        "thoughts_token_count": read(
            "thoughts_token_count"
        ),
        "total_token_count": read(
            "total_token_count"
        ),
    }

    return {
        key: value
        for key, value in normalized.items()
        if value is not None
    }


def _api_status_code(error: BaseException) -> int | None:
    code = getattr(error, "code", None)

    if isinstance(code, int):
        return code

    response = getattr(error, "response", None)
    status_code = getattr(response, "status_code", None)

    return status_code if isinstance(status_code, int) else None


def _mapped_error(
    error: BaseException,
) -> tuple[str, str, bool, int | None]:
    """Map SDK/network exceptions to user-safe error categories."""

    status_code = _api_status_code(error)
    error_text = str(error).casefold()
    class_name = type(error).__name__.casefold()

    if isinstance(error, (TimeoutError, httpx.TimeoutException)):
        return (
            "timeout_error",
            (
                "The Gemini request timed out. Try again or reduce the "
                "conversation length."
            ),
            True,
            status_code,
        )

    if status_code in {401, 403} or any(
        token in error_text
        for token in (
            "api key",
            "authentication",
            "permission denied",
            "unauthenticated",
        )
    ):
        return (
            "authentication_error",
            (
                "Gemini authentication failed. Verify the API key and "
                "its permissions."
            ),
            False,
            status_code,
        )

    if status_code == 429 or "rate limit" in error_text:
        return (
            "rate_limit_error",
            (
                "Gemini rate or quota limits were reached. Wait briefly "
                "and try again."
            ),
            True,
            status_code,
        )

    if status_code in {500, 502, 503, 504} or isinstance(
        error,
        errors.ServerError,
    ):
        return (
            "service_unavailable",
            (
                "The Gemini service is temporarily unavailable. "
                "Try again later."
            ),
            True,
            status_code,
        )

    if isinstance(error, errors.ClientError) or (
        status_code is not None and 400 <= status_code < 500
    ):
        return (
            "request_error",
            (
                "Gemini rejected the request. Review the model, prompt, "
                "tool arguments, and configured limits."
            ),
            False,
            status_code,
        )

    if "connect" in class_name or "network" in error_text:
        return (
            "network_error",
            (
                "A network error prevented communication with Gemini."
            ),
            True,
            status_code,
        )

    return (
        "gemini_api_error",
        (
            "Gemini could not complete the request because of an "
            "unexpected API or network error."
        ),
        False,
        status_code,
    )


_BLOCKED_FINISH_REASONS = {
    "SAFETY",
    "BLOCKLIST",
    "PROHIBITED_CONTENT",
    "SPII",
    "IMAGE_SAFETY",
    "RECITATION",
}


class DataSentryGeminiClient:
    """
    Stateful Gemini client with a bounded manual read-only tool loop.

    Persistent history stores only user and final assistant text. Internal
    function-call parts and tool responses remain local to the active request,
    preserving required thought signatures without retaining stale evidence.
    """

    def __init__(
        self,
        *,
        config: GeminiConfig,
        toolbox: AuditToolbox,
        explanation_style: str | ExplanationStyle | None = None,
        analysis_focus: str | AnalysisFocus | None = None,
        client: Any | None = None,
    ) -> None:
        if not isinstance(config, GeminiConfig):
            raise GeminiClientError(
                "config must be a GeminiConfig instance.",
                code="invalid_config",
            )

        if not isinstance(toolbox, AuditToolbox):
            raise GeminiClientError(
                "toolbox must be an AuditToolbox instance.",
                code="invalid_toolbox",
            )

        self._config = config
        self._toolbox = toolbox
        self._explanation_style = parse_explanation_style(
            explanation_style
            if explanation_style is not None
            else config.default_explanation_style
        )
        self._analysis_focus = parse_analysis_focus(
            analysis_focus
            if analysis_focus is not None
            else config.default_analysis_focus
        )
        self._history: list[dict[str, str]] = []
        self._owns_client = client is None

        if client is None:
            api_key = config.require_api_key()
            timeout_milliseconds = int(
                round(
                    config.request_timeout_seconds
                    * 1000
                )
            )
            self._client = genai.Client(
                api_key=api_key,
                http_options=types.HttpOptions(
                    timeout=timeout_milliseconds
                ),
            )
        else:
            if not hasattr(client, "models") or not hasattr(
                client.models,
                "generate_content",
            ):
                raise GeminiClientError(
                    (
                        "Injected client must expose "
                        "models.generate_content()."
                    ),
                    code="invalid_injected_client",
                )

            self._client = client

        self._tool_declarations = (
            build_audit_tool_declarations()
        )

    @property
    def history_message_count(self) -> int:
        return len(self._history)

    def get_history(self) -> list[dict[str, str]]:
        """Return an independent Streamlit-safe conversation history."""

        return deepcopy(self._history)

    def reset_history(self) -> None:
        """Clear text history without changing the active audit report."""

        self._history.clear()

    def get_state(self) -> dict[str, Any]:
        state = GeminiClientState(
            model=self._config.model,
            history_message_count=len(self._history),
            max_conversation_messages=(
                self._config.max_conversation_messages
            ),
            max_tool_rounds=self._config.max_tool_rounds,
            request_timeout_seconds=(
                self._config.request_timeout_seconds
            ),
            explanation_style=(
                self._explanation_style.value
            ),
            analysis_focus=self._analysis_focus.value,
            api_key_configured=self._config.has_api_key,
        )

        return state.to_dict()

    def set_response_profile(
        self,
        *,
        explanation_style: str | ExplanationStyle,
        analysis_focus: str | AnalysisFocus,
    ) -> None:
        """Change response style and focus for future turns."""

        self._explanation_style = (
            parse_explanation_style(
                explanation_style
            )
        )
        self._analysis_focus = parse_analysis_focus(
            analysis_focus
        )

    def close(self) -> None:
        """Close an internally created SDK client."""

        if self._owns_client and hasattr(
            self._client,
            "close",
        ):
            self._client.close()

    def __enter__(self) -> "DataSentryGeminiClient":
        return self

    def __exit__(
        self,
        exc_type: Any,
        exc: Any,
        traceback: Any,
    ) -> None:
        self.close()

    def _system_instruction(
        self,
        *,
        explanation_style: ExplanationStyle,
        analysis_focus: AnalysisFocus,
    ) -> str:
        return build_system_instruction(
            explanation_style=explanation_style,
            analysis_focus=analysis_focus,
        )

    def _generation_config(
        self,
        *,
        explanation_style: ExplanationStyle,
        analysis_focus: AnalysisFocus,
    ) -> types.GenerateContentConfig:
        return types.GenerateContentConfig(
            system_instruction=self._system_instruction(
                explanation_style=explanation_style,
                analysis_focus=analysis_focus,
            ),
            temperature=self._config.temperature,
            max_output_tokens=(
                self._config.max_output_tokens
            ),
            tools=[self._tool_declarations],
            automatic_function_calling=(
                types.AutomaticFunctionCallingConfig(
                    disable=True
                )
            ),
        )

    def _history_as_contents(
        self,
    ) -> list[types.Content]:
        contents: list[types.Content] = []

        for message in self._history:
            role = (
                "model"
                if message["role"] == "assistant"
                else "user"
            )
            contents.append(
                _text_content(
                    role,
                    message["content"],
                )
            )

        return contents

    def _append_history_turn(
        self,
        *,
        user_text: str,
        assistant_text: str,
    ) -> None:
        self._history.extend(
            [
                {
                    "role": "user",
                    "content": user_text,
                },
                {
                    "role": "assistant",
                    "content": assistant_text,
                },
            ]
        )

        maximum = (
            self._config.max_conversation_messages
        )

        if len(self._history) > maximum:
            self._history = self._history[-maximum:]

        while (
            self._history
            and self._history[0]["role"] != "user"
        ):
            self._history.pop(0)

    def _error_result(
        self,
        *,
        code: str,
        message: str,
        retryable: bool,
        status_code: int | None = None,
        tool_calls: Sequence[Mapping[str, Any]] = (),
        tool_rounds: int = 0,
        finish_reason: str | None = None,
    ) -> dict[str, Any]:
        return _strict_json_copy(
            {
                "ok": False,
                "message": message,
                "text": None,
                "model": self._config.model,
                "finish_reason": finish_reason,
                "tool_rounds": tool_rounds,
                "tool_call_count": len(tool_calls),
                "tool_calls": list(tool_calls),
                "usage": {},
                "history_message_count": len(
                    self._history
                ),
                "error": {
                    "code": code,
                    "retryable": retryable,
                    "status_code": status_code,
                },
            }
        )

    def send_message(
        self,
        user_message: str,
        *,
        explanation_style: str | ExplanationStyle | None = None,
        analysis_focus: str | AnalysisFocus | None = None,
    ) -> dict[str, Any]:
        """
        Send one turn through a bounded manual function-calling loop.

        Expected API, timeout, rate-limit, authentication, blocked,
        empty-response, and tool-limit conditions are returned as structured
        JSON-safe results instead of escaping into the Streamlit interface.
        """

        if not isinstance(user_message, str):
            return self._error_result(
                code="invalid_user_message_type",
                message="The user message must be a string.",
                retryable=False,
            )

        normalized_message = user_message.strip()

        if not normalized_message:
            return self._error_result(
                code="empty_user_message",
                message="Enter a question before sending it to Gemini.",
                retryable=False,
            )

        try:
            active_style = parse_explanation_style(
                explanation_style
                if explanation_style is not None
                else self._explanation_style
            )
            active_focus = parse_analysis_focus(
                analysis_focus
                if analysis_focus is not None
                else self._analysis_focus
            )
        except ConfigurationError as error:
            return self._error_result(
                code=error.code,
                message=str(error),
                retryable=False,
            )

        contents = self._history_as_contents()
        contents.append(
            _text_content(
                "user",
                normalized_message,
            )
        )
        generation_config = self._generation_config(
            explanation_style=active_style,
            analysis_focus=active_focus,
        )

        tool_trace: list[dict[str, Any]] = []
        tool_rounds = 0
        last_finish_reason: str | None = None

        while True:
            try:
                response = (
                    self._client.models.generate_content(
                        model=self._config.model,
                        contents=contents,
                        config=generation_config,
                    )
                )
            except Exception as error:
                (
                    code,
                    message,
                    retryable,
                    status_code,
                ) = _mapped_error(error)

                return self._error_result(
                    code=code,
                    message=message,
                    retryable=retryable,
                    status_code=status_code,
                    tool_calls=tool_trace,
                    tool_rounds=tool_rounds,
                    finish_reason=last_finish_reason,
                )

            (
                model_content,
                function_calls,
                text,
                finish_reason,
            ) = _extract_response_parts(response)
            last_finish_reason = finish_reason

            if function_calls:
                if (
                    tool_rounds
                    >= self._config.max_tool_rounds
                ):
                    return self._error_result(
                        code="tool_round_limit_exceeded",
                        message=(
                            "Gemini requested more tool rounds than "
                            "the configured safety limit."
                        ),
                        retryable=False,
                        tool_calls=tool_trace,
                        tool_rounds=tool_rounds,
                        finish_reason=finish_reason,
                    )

                if model_content is None:
                    return self._error_result(
                        code="invalid_function_call_response",
                        message=(
                            "Gemini returned function calls without "
                            "model content."
                        ),
                        retryable=True,
                        tool_calls=tool_trace,
                        tool_rounds=tool_rounds,
                        finish_reason=finish_reason,
                    )

                # Preserve the original model parts, including thought
                # signatures needed by thinking models.
                contents.append(model_content)

                response_parts: list[types.Part] = []

                for function_call in function_calls:
                    name = getattr(
                        function_call,
                        "name",
                        None,
                    )
                    call_id = getattr(
                        function_call,
                        "id",
                        None,
                    )
                    arguments = getattr(
                        function_call,
                        "args",
                        None,
                    )

                    if isinstance(arguments, Mapping):
                        tool_arguments: Any = dict(
                            arguments
                        )
                    elif arguments is None:
                        tool_arguments = {}
                    else:
                        tool_arguments = arguments

                    tool_result = (
                        self._toolbox.dispatch_tool(
                            str(name or ""),
                            tool_arguments,
                        )
                    )

                    tool_trace.append(
                        {
                            "round": tool_rounds + 1,
                            "name": name,
                            "call_id": call_id,
                            "arguments": tool_arguments,
                            "result": tool_result,
                        }
                    )

                    response_parts.append(
                        types.Part(
                            function_response=(
                                types.FunctionResponse(
                                    name=str(name or ""),
                                    id=(
                                        str(call_id)
                                        if call_id is not None
                                        else None
                                    ),
                                    response=tool_result,
                                )
                            )
                        )
                    )

                contents.append(
                    types.Content(
                        role="user",
                        parts=response_parts,
                    )
                )
                tool_rounds += 1
                continue

            if text:
                self._append_history_turn(
                    user_text=normalized_message,
                    assistant_text=text,
                )

                return _strict_json_copy(
                    {
                        "ok": True,
                        "message": (
                            "Gemini response completed."
                        ),
                        "text": text,
                        "model": self._config.model,
                        "finish_reason": finish_reason,
                        "response_id": getattr(
                            response,
                            "response_id",
                            None,
                        ),
                        "model_version": getattr(
                            response,
                            "model_version",
                            None,
                        ),
                        "tool_rounds": tool_rounds,
                        "tool_call_count": len(
                            tool_trace
                        ),
                        "tool_calls": tool_trace,
                        "usage": _usage_metadata(
                            response
                        ),
                        "history_message_count": len(
                            self._history
                        ),
                        "response_profile": {
                            "explanation_style": (
                                active_style.value
                            ),
                            "analysis_focus": (
                                active_focus.value
                            ),
                        },
                        "error": None,
                    }
                )

            normalized_finish = (
                finish_reason.upper()
                if isinstance(finish_reason, str)
                else ""
            )

            if normalized_finish in _BLOCKED_FINISH_REASONS:
                return self._error_result(
                    code="response_blocked",
                    message=(
                        "Gemini did not return an answer because the "
                        "response was blocked by a safety or content "
                        "policy."
                    ),
                    retryable=False,
                    tool_calls=tool_trace,
                    tool_rounds=tool_rounds,
                    finish_reason=finish_reason,
                )

            if normalized_finish == "MAX_TOKENS":
                return self._error_result(
                    code="response_truncated",
                    message=(
                        "Gemini reached the output-token limit before "
                        "returning a usable answer."
                    ),
                    retryable=True,
                    tool_calls=tool_trace,
                    tool_rounds=tool_rounds,
                    finish_reason=finish_reason,
                )

            return self._error_result(
                code="empty_response",
                message=(
                    "Gemini returned no text and no function call. "
                    "Try rephrasing the question."
                ),
                retryable=True,
                tool_calls=tool_trace,
                tool_rounds=tool_rounds,
                finish_reason=finish_reason,
            )


def build_gemini_client(
    *,
    config: GeminiConfig,
    toolbox: AuditToolbox,
    explanation_style: str | ExplanationStyle | None = None,
    analysis_focus: str | AnalysisFocus | None = None,
    client: Any | None = None,
) -> DataSentryGeminiClient:
    """Build a validated DataSentry Gemini client."""

    return DataSentryGeminiClient(
        config=config,
        toolbox=toolbox,
        explanation_style=explanation_style,
        analysis_focus=analysis_focus,
        client=client,
    )
