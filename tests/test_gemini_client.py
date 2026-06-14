from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import pytest
from google.genai import errors, types

from src.config import (
    AnalysisFocus,
    ExplanationStyle,
    GeminiConfig,
)
from src.gemini_client import (
    DataSentryGeminiClient,
    GeminiClientError,
    build_audit_tool_declarations,
    build_gemini_client,
)
from src.report_builder import build_audit_report
from src.tools import AuditToolbox


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_PATH = (
    PROJECT_ROOT
    / "data"
    / "sample_dirty_customers.csv"
)
FIXED_TIME = "2026-06-14T12:00:00Z"


class FakeModels:
    def __init__(
        self,
        responses: list[Any],
    ) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def generate_content(
        self,
        *,
        model: str,
        contents: Any,
        config: Any,
    ) -> Any:
        self.calls.append(
            {
                "model": model,
                "contents": deepcopy(contents),
                "config": config,
            }
        )

        if not self.responses:
            raise AssertionError("No fake response remains.")

        response = self.responses.pop(0)

        if isinstance(response, BaseException):
            raise response

        return response


class FakeClient:
    def __init__(
        self,
        responses: list[Any],
    ) -> None:
        self.models = FakeModels(responses)
        self.closed = False

    def close(self) -> None:
        self.closed = True


def text_response(
    text: str,
    *,
    finish_reason: str = "STOP",
    prompt_tokens: int = 20,
    response_tokens: int = 10,
) -> types.GenerateContentResponse:
    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(
                    role="model",
                    parts=[types.Part(text=text)],
                ),
                finish_reason=finish_reason,
            )
        ],
        response_id="response-123",
        model_version="gemini-test",
        usage_metadata=(
            types.GenerateContentResponseUsageMetadata(
                prompt_token_count=prompt_tokens,
                candidates_token_count=response_tokens,
                total_token_count=(
                    prompt_tokens + response_tokens
                ),
            )
        ),
    )


def function_response(
    calls: list[
        tuple[str, dict[str, Any], str | None]
    ],
    *,
    thought_signature: bytes | None = None,
) -> types.GenerateContentResponse:
    parts: list[types.Part] = []

    for index, (name, arguments, call_id) in enumerate(
        calls
    ):
        parts.append(
            types.Part(
                function_call=types.FunctionCall(
                    name=name,
                    args=arguments,
                    id=call_id,
                ),
                thought_signature=(
                    thought_signature
                    if index == 0
                    else None
                ),
            )
        )

    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(
                    role="model",
                    parts=parts,
                ),
                finish_reason="STOP",
            )
        ]
    )


def empty_response(
    *,
    finish_reason: str = "STOP",
) -> types.GenerateContentResponse:
    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(
                    role="model",
                    parts=[],
                ),
                finish_reason=finish_reason,
            )
        ]
    )


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_PATH)


@pytest.fixture
def sample_report(
    sample_dataframe: pd.DataFrame,
) -> dict[str, Any]:
    return build_audit_report(
        sample_dataframe,
        generated_at_utc=FIXED_TIME,
    )


@pytest.fixture
def toolbox(
    sample_report: dict[str, Any],
) -> AuditToolbox:
    return AuditToolbox(sample_report)


@pytest.fixture
def config() -> GeminiConfig:
    return GeminiConfig(
        api_key=None,
        model="gemini-2.5-flash",
        temperature=0.2,
        max_output_tokens=2048,
        max_conversation_messages=4,
        max_tool_rounds=2,
        request_timeout_seconds=30.0,
        default_explanation_style=(
            ExplanationStyle.BUSINESS_FRIENDLY
        ),
        default_analysis_focus=(
            AnalysisFocus.GENERAL_DATA_QUALITY
        ),
    )


def assert_json_safe(value: object) -> None:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
    )
    assert "NaN" not in payload
    assert "Infinity" not in payload


def test_builds_all_explicit_tool_declarations() -> None:
    tool = build_audit_tool_declarations()
    declarations = tool.function_declarations or []

    assert [item.name for item in declarations] == [
        "get_dataset_overview",
        "get_quality_summary",
        "get_missing_value_report",
        "get_duplicate_report",
        "get_column_quality_report",
        "get_priority_issues",
        "get_ml_readiness_report",
    ]

    column_tool = next(
        item
        for item in declarations
        if item.name == "get_column_quality_report"
    )
    schema = column_tool.parameters_json_schema

    assert schema["required"] == ["column_name"]
    assert schema["additionalProperties"] is False


def test_direct_text_response_is_streamlit_safe(
    toolbox: AuditToolbox,
    config: GeminiConfig,
) -> None:
    fake = FakeClient(
        [text_response("The dataset needs cleaning.")]
    )
    client = DataSentryGeminiClient(
        config=config,
        toolbox=toolbox,
        client=fake,
    )

    result = client.send_message(
        "Is this dataset ready?"
    )

    assert result["ok"] is True
    assert result["text"] == (
        "The dataset needs cleaning."
    )
    assert result["tool_rounds"] == 0
    assert result["tool_call_count"] == 0
    assert result["usage"]["response_token_count"] == 10
    assert result["usage"]["total_token_count"] == 30
    assert result["response_profile"] == {
        "explanation_style": "business-friendly",
        "analysis_focus": "general-data-quality",
    }
    assert result["history_message_count"] == 2
    assert_json_safe(result)


def test_generation_config_disables_automatic_calls(
    toolbox: AuditToolbox,
    config: GeminiConfig,
) -> None:
    fake = FakeClient([text_response("Done")])
    client = DataSentryGeminiClient(
        config=config,
        toolbox=toolbox,
        client=fake,
    )

    client.send_message("Summarize the audit.")

    generation_config = fake.models.calls[0]["config"]

    assert generation_config.temperature == 0.2
    assert generation_config.max_output_tokens == 2048
    assert (
        generation_config.automatic_function_calling.disable
        is True
    )
    assert generation_config.system_instruction
    assert "only source of truth" in (
        generation_config.system_instruction
    )
    assert len(generation_config.tools) == 1


def test_single_function_call_loop(
    toolbox: AuditToolbox,
    config: GeminiConfig,
) -> None:
    first = function_response(
        [
            (
                "get_quality_summary",
                {},
                "call-quality",
            )
        ],
        thought_signature=b"signature",
    )
    fake = FakeClient(
        [
            first,
            text_response(
                "The score is 92.02 and cleaning is required."
            ),
        ]
    )
    client = DataSentryGeminiClient(
        config=config,
        toolbox=toolbox,
        client=fake,
    )

    result = client.send_message(
        "What is the quality score?"
    )

    assert result["ok"] is True
    assert result["tool_rounds"] == 1
    assert result["tool_call_count"] == 1
    assert result["tool_calls"][0]["name"] == (
        "get_quality_summary"
    )
    assert result["tool_calls"][0]["result"]["ok"] is True
    assert result["tool_calls"][0]["result"]["data"][
        "overall_score"
    ] == 92.02
    assert len(fake.models.calls) == 2

    second_contents = fake.models.calls[1]["contents"]
    model_content = second_contents[-2]
    tool_content = second_contents[-1]

    assert (
        model_content.parts[0].thought_signature
        == b"signature"
    )
    function_result = (
        tool_content.parts[0].function_response
    )
    assert function_result.name == "get_quality_summary"
    assert function_result.id == "call-quality"
    assert function_result.response["ok"] is True


def test_parallel_function_calls_are_returned_in_order(
    toolbox: AuditToolbox,
    config: GeminiConfig,
) -> None:
    fake = FakeClient(
        [
            function_response(
                [
                    (
                        "get_dataset_overview",
                        {},
                        "call-1",
                    ),
                    (
                        "get_duplicate_report",
                        {},
                        "call-2",
                    ),
                ]
            ),
            text_response(
                "Overview and duplicate report ready."
            ),
        ]
    )
    client = build_gemini_client(
        config=config,
        toolbox=toolbox,
        client=fake,
    )

    result = client.send_message(
        "Give me dimensions and duplicate evidence."
    )

    assert result["ok"] is True
    assert result["tool_rounds"] == 1
    assert [
        item["name"]
        for item in result["tool_calls"]
    ] == [
        "get_dataset_overview",
        "get_duplicate_report",
    ]

    response_parts = fake.models.calls[1][
        "contents"
    ][-1].parts

    assert [
        part.function_response.id
        for part in response_parts
    ] == ["call-1", "call-2"]


def test_unknown_model_tool_is_safely_dispatched(
    toolbox: AuditToolbox,
    config: GeminiConfig,
) -> None:
    fake = FakeClient(
        [
            function_response(
                [
                    (
                        "delete_rows",
                        {"column": "age"},
                        "bad-call",
                    )
                ]
            ),
            text_response(
                "That operation is not available."
            ),
        ]
    )
    client = DataSentryGeminiClient(
        config=config,
        toolbox=toolbox,
        client=fake,
    )

    result = client.send_message(
        "Delete invalid rows."
    )

    assert result["ok"] is True
    tool_result = result["tool_calls"][0]["result"]

    assert tool_result["ok"] is False
    assert tool_result["error"]["code"] == "unknown_tool"


def test_tool_round_limit_is_enforced(
    toolbox: AuditToolbox,
    config: GeminiConfig,
) -> None:
    limited_config = GeminiConfig(
        model=config.model,
        max_tool_rounds=1,
        max_conversation_messages=4,
    )
    fake = FakeClient(
        [
            function_response(
                [
                    (
                        "get_quality_summary",
                        {},
                        "call-1",
                    )
                ]
            ),
            function_response(
                [
                    (
                        "get_priority_issues",
                        {"limit": 1},
                        "call-2",
                    )
                ]
            ),
        ]
    )
    client = DataSentryGeminiClient(
        config=limited_config,
        toolbox=toolbox,
        client=fake,
    )

    result = client.send_message(
        "Analyze the dataset."
    )

    assert result["ok"] is False
    assert result["error"]["code"] == (
        "tool_round_limit_exceeded"
    )
    assert result["tool_rounds"] == 1
    assert result["tool_call_count"] == 1
    assert client.get_history() == []


def test_history_is_bounded_to_complete_recent_turns(
    toolbox: AuditToolbox,
    config: GeminiConfig,
) -> None:
    fake = FakeClient(
        [
            text_response("Answer one."),
            text_response("Answer two."),
            text_response("Answer three."),
        ]
    )
    client = DataSentryGeminiClient(
        config=config,
        toolbox=toolbox,
        client=fake,
    )

    client.send_message("Question one.")
    client.send_message("Question two.")
    client.send_message("Question three.")

    assert client.get_history() == [
        {
            "role": "user",
            "content": "Question two.",
        },
        {
            "role": "assistant",
            "content": "Answer two.",
        },
        {
            "role": "user",
            "content": "Question three.",
        },
        {
            "role": "assistant",
            "content": "Answer three.",
        },
    ]

    third_contents = fake.models.calls[2]["contents"]

    assert [
        content.role
        for content in third_contents
    ] == [
        "user",
        "model",
        "user",
        "model",
        "user",
    ]


def test_reset_history(
    toolbox: AuditToolbox,
    config: GeminiConfig,
) -> None:
    fake = FakeClient([text_response("Answer")])
    client = DataSentryGeminiClient(
        config=config,
        toolbox=toolbox,
        client=fake,
    )

    client.send_message("Question")
    assert client.history_message_count == 2

    client.reset_history()

    assert client.history_message_count == 0
    assert client.get_history() == []


def test_response_profile_can_be_overridden_per_turn(
    toolbox: AuditToolbox,
    config: GeminiConfig,
) -> None:
    fake = FakeClient([text_response("Technical answer")])
    client = DataSentryGeminiClient(
        config=config,
        toolbox=toolbox,
        client=fake,
    )

    result = client.send_message(
        "Explain this.",
        explanation_style="technical",
        analysis_focus="ml",
    )

    assert result["response_profile"] == {
        "explanation_style": "technical",
        "analysis_focus": "machine-learning-readiness",
    }
    system_instruction = fake.models.calls[0][
        "config"
    ].system_instruction
    assert "Explanation style: technical" in (
        system_instruction
    )
    assert (
        "Analysis focus: machine-learning-readiness"
        in system_instruction
    )


def test_set_response_profile_updates_future_turns(
    toolbox: AuditToolbox,
    config: GeminiConfig,
) -> None:
    fake = FakeClient([text_response("Simple answer")])
    client = DataSentryGeminiClient(
        config=config,
        toolbox=toolbox,
        client=fake,
    )

    client.set_response_profile(
        explanation_style="beginner",
        analysis_focus="ml",
    )
    result = client.send_message("Explain.")

    assert result["response_profile"] == {
        "explanation_style": "beginner-friendly",
        "analysis_focus": "machine-learning-readiness",
    }


@pytest.mark.parametrize(
    ("message", "expected_code"),
    [
        ("", "empty_user_message"),
        ("   ", "empty_user_message"),
        (None, "invalid_user_message_type"),
    ],
)
def test_invalid_user_message_does_not_call_api(
    toolbox: AuditToolbox,
    config: GeminiConfig,
    message: Any,
    expected_code: str,
) -> None:
    fake = FakeClient([])
    client = DataSentryGeminiClient(
        config=config,
        toolbox=toolbox,
        client=fake,
    )

    result = client.send_message(message)

    assert result["ok"] is False
    assert result["error"]["code"] == expected_code
    assert fake.models.calls == []


def test_empty_response_is_structured_error(
    toolbox: AuditToolbox,
    config: GeminiConfig,
) -> None:
    fake = FakeClient([empty_response()])
    client = DataSentryGeminiClient(
        config=config,
        toolbox=toolbox,
        client=fake,
    )

    result = client.send_message("Answer me.")

    assert result["ok"] is False
    assert result["error"]["code"] == "empty_response"
    assert result["error"]["retryable"] is True
    assert client.get_history() == []


@pytest.mark.parametrize(
    "finish_reason",
    [
        "SAFETY",
        "BLOCKLIST",
        "PROHIBITED_CONTENT",
    ],
)
def test_blocked_response_is_structured_error(
    toolbox: AuditToolbox,
    config: GeminiConfig,
    finish_reason: str,
) -> None:
    fake = FakeClient(
        [
            empty_response(
                finish_reason=finish_reason
            )
        ]
    )
    client = DataSentryGeminiClient(
        config=config,
        toolbox=toolbox,
        client=fake,
    )

    result = client.send_message("Answer me.")

    assert result["ok"] is False
    assert result["error"]["code"] == "response_blocked"
    assert result["finish_reason"] == finish_reason


def test_max_tokens_without_text_is_truncated_error(
    toolbox: AuditToolbox,
    config: GeminiConfig,
) -> None:
    fake = FakeClient(
        [empty_response(finish_reason="MAX_TOKENS")]
    )
    client = DataSentryGeminiClient(
        config=config,
        toolbox=toolbox,
        client=fake,
    )

    result = client.send_message("Answer me.")

    assert result["error"]["code"] == "response_truncated"
    assert result["error"]["retryable"] is True


def test_timeout_is_mapped(
    toolbox: AuditToolbox,
    config: GeminiConfig,
) -> None:
    request = httpx.Request(
        "POST",
        "https://generativelanguage.googleapis.com",
    )
    fake = FakeClient(
        [
            httpx.ReadTimeout(
                "timed out",
                request=request,
            )
        ]
    )
    client = DataSentryGeminiClient(
        config=config,
        toolbox=toolbox,
        client=fake,
    )

    result = client.send_message("Question")

    assert result["ok"] is False
    assert result["error"] == {
        "code": "timeout_error",
        "retryable": True,
        "status_code": None,
    }


@pytest.mark.parametrize(
    (
        "exception",
        "expected_code",
        "retryable",
        "status_code",
    ),
    [
        (
            errors.ClientError(
                401,
                {
                    "error": {
                        "message": "Invalid API key"
                    }
                },
            ),
            "authentication_error",
            False,
            401,
        ),
        (
            errors.ClientError(
                429,
                {
                    "error": {
                        "message": "Rate limit exceeded"
                    }
                },
            ),
            "rate_limit_error",
            True,
            429,
        ),
        (
            errors.ServerError(
                503,
                {
                    "error": {
                        "message": "Unavailable"
                    }
                },
            ),
            "service_unavailable",
            True,
            503,
        ),
        (
            errors.ClientError(
                400,
                {
                    "error": {
                        "message": "Bad request"
                    }
                },
            ),
            "request_error",
            False,
            400,
        ),
    ],
)
def test_api_errors_are_mapped(
    toolbox: AuditToolbox,
    config: GeminiConfig,
    exception: BaseException,
    expected_code: str,
    retryable: bool,
    status_code: int,
) -> None:
    fake = FakeClient([exception])
    client = DataSentryGeminiClient(
        config=config,
        toolbox=toolbox,
        client=fake,
    )

    result = client.send_message("Question")

    assert result["ok"] is False
    assert result["error"] == {
        "code": expected_code,
        "retryable": retryable,
        "status_code": status_code,
    }
    assert "Invalid API key" not in result["message"]


def test_unexpected_error_is_sanitized(
    toolbox: AuditToolbox,
    config: GeminiConfig,
) -> None:
    fake = FakeClient(
        [RuntimeError("secret internal detail")]
    )
    client = DataSentryGeminiClient(
        config=config,
        toolbox=toolbox,
        client=fake,
    )

    result = client.send_message("Question")

    assert result["error"]["code"] == "gemini_api_error"
    assert "secret internal detail" not in result["message"]


def test_injected_client_does_not_require_api_key(
    toolbox: AuditToolbox,
    config: GeminiConfig,
) -> None:
    fake = FakeClient([text_response("Answer")])

    client = build_gemini_client(
        config=config,
        toolbox=toolbox,
        client=fake,
    )

    assert client.send_message("Question")["ok"] is True


def test_real_client_factory_uses_key_and_millisecond_timeout(
    monkeypatch: pytest.MonkeyPatch,
    toolbox: AuditToolbox,
) -> None:
    captured: dict[str, Any] = {}

    class FactoryClient:
        def __init__(
            self,
            *,
            api_key: str,
            http_options: types.HttpOptions,
        ) -> None:
            captured["api_key"] = api_key
            captured["http_options"] = http_options
            self.models = FakeModels([])

        def close(self) -> None:
            captured["closed"] = True

    monkeypatch.setattr(
        "src.gemini_client.genai.Client",
        FactoryClient,
    )

    config = GeminiConfig(
        api_key="secret-key",
        request_timeout_seconds=12.5,
    )
    client = DataSentryGeminiClient(
        config=config,
        toolbox=toolbox,
    )

    assert captured["api_key"] == "secret-key"
    assert captured["http_options"].timeout == 12_500

    client.close()

    assert captured["closed"] is True


def test_missing_key_rejected_when_building_real_client(
    toolbox: AuditToolbox,
    config: GeminiConfig,
) -> None:
    with pytest.raises(
        Exception,
        match="API key is not configured",
    ):
        DataSentryGeminiClient(
            config=config,
            toolbox=toolbox,
        )


def test_invalid_constructor_inputs(
    toolbox: AuditToolbox,
    config: GeminiConfig,
) -> None:
    with pytest.raises(
        GeminiClientError,
        match="GeminiConfig",
    ):
        DataSentryGeminiClient(
            config="invalid",  # type: ignore[arg-type]
            toolbox=toolbox,
            client=FakeClient([]),
        )

    with pytest.raises(
        GeminiClientError,
        match="AuditToolbox",
    ):
        DataSentryGeminiClient(
            config=config,
            toolbox="invalid",  # type: ignore[arg-type]
            client=FakeClient([]),
        )

    with pytest.raises(
        GeminiClientError,
        match="models.generate_content",
    ):
        DataSentryGeminiClient(
            config=config,
            toolbox=toolbox,
            client=object(),
        )


def test_get_state_is_secret_free(
    toolbox: AuditToolbox,
) -> None:
    config = GeminiConfig(
        api_key="very-secret",
        model="gemini-2.5-flash",
    )
    client = DataSentryGeminiClient(
        config=config,
        toolbox=toolbox,
        client=FakeClient([]),
    )

    state = client.get_state()

    assert state["api_key_configured"] is True
    assert "very-secret" not in str(state)
    assert "api_key" not in state


def test_failed_turn_does_not_modify_existing_history(
    toolbox: AuditToolbox,
    config: GeminiConfig,
) -> None:
    fake = FakeClient(
        [
            text_response("First answer"),
            RuntimeError("failure"),
        ]
    )
    client = DataSentryGeminiClient(
        config=config,
        toolbox=toolbox,
        client=fake,
    )

    client.send_message("First question")
    existing = client.get_history()

    result = client.send_message("Second question")

    assert result["ok"] is False
    assert client.get_history() == existing


def test_original_report_remains_unchanged_after_tool_loop(
    sample_report: dict[str, Any],
    config: GeminiConfig,
) -> None:
    original = deepcopy(sample_report)
    toolbox = AuditToolbox(sample_report)
    fake = FakeClient(
        [
            function_response(
                [
                    (
                        "get_priority_issues",
                        {"limit": 3},
                        "call-1",
                    )
                ]
            ),
            text_response("Done"),
        ]
    )
    client = DataSentryGeminiClient(
        config=config,
        toolbox=toolbox,
        client=fake,
    )

    result = client.send_message("Top issues?")

    assert result["ok"] is True
    assert sample_report == original
