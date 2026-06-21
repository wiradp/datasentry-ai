from __future__ import annotations

import hashlib
import inspect
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Mapping, Sequence

import pandas as pd
import streamlit as st

from src import data_loader as data_loader_module
from src.config import (
    ConfigurationError,
    GeminiConfig,
    available_analysis_focuses,
    available_explanation_styles,
    load_gemini_config,
)
from src.gemini_client import DataSentryGeminiClient
from src.report_builder import (
    ReportBuilderConfig,
    audit_report_to_json,
    build_audit_report,
)
from src.response_formatter import format_copilot_response
from src.tools import AuditToolbox


LOGGER = logging.getLogger(__name__)

APP_TITLE = "DataSentry AI"
APP_SUBTITLE = "Gemini-Powered Data Quality Copilot"
PREVIEW_ROW_LIMIT = 25
MAX_PREVIEW_ROW_LIMIT = 100

DASHBOARD_TAB_LABELS = (
    "Summary",
    "Issues",
    "Recommendations",
    "Report",
    "AI Copilot",
)

_STYLE_LABELS = {
    "beginner-friendly": "Beginner friendly",
    "business-friendly": "Business friendly",
    "technical": "Technical",
}

_FOCUS_LABELS = {
    "general-data-quality": "General data quality",
    "machine-learning-readiness": "Machine-learning readiness",
}

_LOADER_FUNCTION_CANDIDATES = (
    "load_csv",
    "load_csv_file",
    "load_csv_data",
    "load_csv_safely",
    "read_csv_safely",
)

_SESSION_DEFAULTS: dict[str, Any] = {
    "active_upload_key": None,
    "audit_artifacts": None,
    "gemini_client": None,
    "gemini_client_report_id": None,
    "chat_records": [],
    "last_audit_error": None,
}


class AppOrchestrationError(RuntimeError):
    """Raised when app-level orchestration cannot continue safely."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "app_orchestration_error",
    ) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class AuditArtifacts:
    """
    Session-safe output of one deterministic audit.

    Only a bounded preview is retained. Gemini receives the structured report
    through AuditToolbox and never receives the source DataFrame.
    """

    file_key: str
    report: dict[str, Any]
    preview: pd.DataFrame

    @property
    def report_id(self) -> str:
        return str(self.report["report_id"])


def _read_value(
    source: Mapping[str, Any] | Any,
    *names: str,
    default: Any = None,
) -> Any:
    if isinstance(source, Mapping):
        for name in names:
            if name in source:
                return source[name]
        return default

    for name in names:
        if hasattr(source, name):
            return getattr(source, name)

    return default


def resolve_csv_loader(
    module: ModuleType = data_loader_module,
) -> Callable[..., Any]:
    """
    Resolve the project's public CSV loader without bypassing validation.

    Named public candidates are preferred. A conservative fallback accepts
    exactly one public function whose name contains both "csv" and
    "load"/"read".
    """

    for name in _LOADER_FUNCTION_CANDIDATES:
        candidate = getattr(module, name, None)

        if callable(candidate) and not inspect.isclass(candidate):
            return candidate

    discovered: list[Callable[..., Any]] = []

    for name, value in vars(module).items():
        normalized = name.casefold()

        if name.startswith("_") or not inspect.isfunction(value):
            continue

        if "csv" in normalized and (
            "load" in normalized or "read" in normalized
        ):
            discovered.append(value)

    if len(discovered) == 1:
        return discovered[0]

    available = sorted(
        name
        for name, value in vars(module).items()
        if callable(value) and not name.startswith("_")
    )

    raise AppOrchestrationError(
        (
            "No unambiguous public CSV loader was found in "
            "src.data_loader."
        ),
        code="csv_loader_not_found",
    ) from None


def extract_dataframe(
    loader_result: Any,
) -> pd.DataFrame:
    """Extract the validated DataFrame from a loader result."""

    if isinstance(loader_result, pd.DataFrame):
        dataframe = loader_result
    else:
        dataframe = _read_value(
            loader_result,
            "dataframe",
            "df",
            "data",
            default=None,
        )

    if not isinstance(dataframe, pd.DataFrame):
        raise AppOrchestrationError(
            (
                "The CSV loader result does not expose a pandas "
                "DataFrame."
            ),
            code="loader_dataframe_missing",
        )

    if dataframe.empty:
        raise AppOrchestrationError(
            "The validated CSV contains no data rows.",
            code="empty_dataframe",
        )

    return dataframe


def uploaded_file_bytes(
    uploaded_file: Any,
) -> bytes:
    """Read uploaded bytes without depending on a specific Streamlit class."""

    if uploaded_file is None:
        raise AppOrchestrationError(
            "No CSV file was provided.",
            code="missing_upload",
        )

    if hasattr(uploaded_file, "getvalue"):
        value = uploaded_file.getvalue()

        if isinstance(value, bytes):
            return value

    if not hasattr(uploaded_file, "read"):
        raise AppOrchestrationError(
            "The uploaded object is not file-like.",
            code="invalid_upload_object",
        )

    original_position: int | None = None

    if hasattr(uploaded_file, "tell"):
        try:
            original_position = uploaded_file.tell()
        except (OSError, ValueError):
            original_position = None

    if hasattr(uploaded_file, "seek"):
        uploaded_file.seek(0)

    value = uploaded_file.read()

    if (
        original_position is not None
        and hasattr(uploaded_file, "seek")
    ):
        uploaded_file.seek(original_position)

    if not isinstance(value, bytes):
        raise AppOrchestrationError(
            "The uploaded file could not be read as bytes.",
            code="invalid_upload_bytes",
        )

    return value


def uploaded_file_key(
    uploaded_file: Any,
) -> str:
    """Return a deterministic key for detecting a changed upload."""

    payload = uploaded_file_bytes(uploaded_file)
    file_name = str(
        getattr(uploaded_file, "name", "uploaded.csv")
    )

    digest = hashlib.sha256()
    digest.update(file_name.encode("utf-8", errors="replace"))
    digest.update(b"\0")
    digest.update(payload)

    return digest.hexdigest()


def _prepare_upload_for_loader(
    uploaded_file: Any,
) -> None:
    if hasattr(uploaded_file, "seek"):
        uploaded_file.seek(0)


def build_audit_artifacts(
    uploaded_file: Any,
    *,
    loader_callable: Callable[..., Any] | None = None,
    preview_rows: int = PREVIEW_ROW_LIMIT,
) -> AuditArtifacts:
    """
    Load one CSV through src.data_loader and build its deterministic report.
    """

    if (
        isinstance(preview_rows, bool)
        or not isinstance(preview_rows, int)
        or not 1 <= preview_rows <= MAX_PREVIEW_ROW_LIMIT
    ):
        raise AppOrchestrationError(
            (
                f"preview_rows must be an integer between 1 and "
                f"{MAX_PREVIEW_ROW_LIMIT}."
            ),
            code="invalid_preview_limit",
        )

    file_key = uploaded_file_key(uploaded_file)
    loader = loader_callable or resolve_csv_loader()

    _prepare_upload_for_loader(uploaded_file)
    loader_result = loader(uploaded_file)
    dataframe = extract_dataframe(loader_result)

    report = build_audit_report(
        dataframe,
        file_metadata=loader_result,
        report_config=ReportBuilderConfig(
            include_raw_check_results=False,
        ),
    )

    return AuditArtifacts(
        file_key=file_key,
        report=report,
        preview=dataframe.head(preview_rows).copy(deep=True),
    )


def component_scores_frame(
    report: Mapping[str, Any],
) -> pd.DataFrame:
    """Return component scores in a dashboard-friendly table."""

    scores = report.get("component_scores", {})
    weights = report.get("score_weights", {})
    contributions = report.get(
        "weighted_contributions",
        {},
    )

    rows = [
        {
            "component": str(component),
            "score": float(score),
            "weight": (
                float(weights[component])
                if component in weights
                else None
            ),
            "weighted_contribution": (
                float(contributions[component])
                if component in contributions
                else None
            ),
        }
        for component, score in scores.items()
    ]

    return pd.DataFrame(
        rows,
        columns=[
            "component",
            "score",
            "weight",
            "weighted_contribution",
        ],
    )


def format_component_label(
    component_name: str,
) -> str:
    """Convert a snake-case component name into a readable chart label."""

    words = (
        str(component_name)
        .strip()
        .replace("_", " ")
        .split()
    )

    if not words:
        return "Unnamed component"

    return " ".join(
        word.capitalize()
        for word in words
    )


def component_chart_frame(
    report: Mapping[str, Any],
) -> pd.DataFrame:
    """
    Return component scores with labels optimized for a horizontal chart.
    """

    frame = component_scores_frame(report)

    if frame.empty:
        return pd.DataFrame(
            columns=[
                "component_label",
                "score",
            ]
        )

    return pd.DataFrame(
        {
            "component_label": frame[
                "component"
            ].map(format_component_label),
            "score": frame["score"],
        }
    )


def issues_frame(
    report: Mapping[str, Any],
) -> pd.DataFrame:
    """Return normalized issues with stable display columns."""

    rows = []

    for issue in report.get("issues", []):
        rows.append(
            {
                "rank": issue.get("priority_rank"),
                "severity": issue.get("severity"),
                "check": issue.get("check_name"),
                "issue_type": issue.get("issue_type"),
                "column": issue.get("column"),
                "count": issue.get("count"),
                "percentage": issue.get("percentage"),
                "evidence": issue.get("evidence"),
                "recommendation": issue.get(
                    "recommendation"
                ),
                "limitation": issue.get("limitation"),
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "rank",
            "severity",
            "check",
            "issue_type",
            "column",
            "count",
            "percentage",
            "evidence",
            "recommendation",
            "limitation",
        ],
    )


def recommendations_frame(
    report: Mapping[str, Any],
) -> pd.DataFrame:
    """Return prioritized recommendations with stable display columns."""

    rows = []

    for item in report.get(
        "prioritized_recommendations",
        [],
    ):
        rows.append(
            {
                "rank": item.get("priority_rank"),
                "priority": item.get("priority_level"),
                "severity": item.get("severity"),
                "column": item.get("column"),
                "issue_type": item.get("issue_type"),
                "action": item.get("action"),
                "rationale": item.get("rationale"),
                "source_checks": ", ".join(
                    item.get("source_checks", [])
                ),
                "related_issues": ", ".join(
                    item.get("related_issue_ids", [])
                ),
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "rank",
            "priority",
            "severity",
            "column",
            "issue_type",
            "action",
            "rationale",
            "source_checks",
            "related_issues",
        ],
    )


def filter_issue_records(
    issues: Sequence[Mapping[str, Any]],
    *,
    severities: Sequence[str] | None = None,
    check_name: str | None = None,
    column_name: str | None = None,
) -> list[dict[str, Any]]:
    """Filter normalized issue records without mutating the report."""

    severity_set = (
        {str(value).upper() for value in severities}
        if severities is not None
        else None
    )

    return [
        dict(issue)
        for issue in issues
        if (
            severity_set is None
            or str(issue.get("severity")).upper()
            in severity_set
        )
        and (
            check_name is None
            or issue.get("check_name") == check_name
        )
        and (
            column_name is None
            or issue.get("column") == column_name
        )
    ]


def report_download_name(
    report: Mapping[str, Any],
) -> str:
    """Build a safe, readable JSON download filename."""

    metadata = report.get("file_metadata", {})
    file_name = str(
        metadata.get("file_name") or "uploaded.csv"
    )
    stem = Path(file_name).stem or "uploaded"
    safe_stem = "".join(
        character
        if character.isalnum() or character in {"-", "_"}
        else "_"
        for character in stem
    ).strip("_")

    return (
        f"datasentry_audit_{safe_stem or 'uploaded'}.json"
    )


def dashboard_tab_state_key(
    report_id: str,
) -> str:
    """
    Return a report-scoped Streamlit key for the dashboard tabs.

    Streamlit preserves the selected tab across reruns when tabs use
    a key together with on_change="rerun".
    """

    normalized_report_id = str(report_id).strip()

    if not normalized_report_id:
        raise AppOrchestrationError(
            "report_id cannot be empty.",
            code="empty_report_id",
        )

    safe_report_id = "".join(
        character
        if character.isalnum() or character in {"-", "_"}
        else "_"
        for character in normalized_report_id
    )

    return f"dashboard_tab_{safe_report_id}"


def readiness_display(
    readiness_status: str,
) -> tuple[str, str]:
    """Return a user label and Streamlit callout level."""

    normalized = str(readiness_status).upper()
    labels = {
        "READY_WITH_MINOR_REVIEW": (
            "Ready with minor review",
            "success",
        ),
        "NEEDS_CLEANING": (
            "Needs cleaning",
            "warning",
        ),
        "SIGNIFICANT_QUALITY_ISSUES": (
            "Significant quality issues",
            "error",
        ),
        "NOT_READY": (
            "Not ready",
            "error",
        ),
    }

    return labels.get(
        normalized,
        (
            normalized.replace("_", " ").title(),
            "info",
        ),
    )


def _initialize_session_state() -> None:
    for key, default in _SESSION_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = (
                list(default)
                if isinstance(default, list)
                else default
            )


def _close_session_client() -> None:
    client = st.session_state.get("gemini_client")

    if client is not None:
        try:
            client.close()
        except Exception:
            LOGGER.exception(
                "Failed to close the Gemini client cleanly."
            )

    st.session_state["gemini_client"] = None
    st.session_state["gemini_client_report_id"] = None


def _reset_audit_state(
    *,
    active_upload_key: str | None,
) -> None:
    _close_session_client()
    st.session_state["active_upload_key"] = (
        active_upload_key
    )
    st.session_state["audit_artifacts"] = None
    st.session_state["chat_records"] = []
    st.session_state["last_audit_error"] = None


def _secret_aware_environment() -> dict[str, str]:
    environment = {
        str(key): str(value)
        for key, value in os.environ.items()
    }

    keys = (
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "GEMINI_MODEL",
        "GEMINI_TEMPERATURE",
        "GEMINI_MAX_OUTPUT_TOKENS",
        "GEMINI_MAX_CONVERSATION_MESSAGES",
        "GEMINI_MAX_TOOL_ROUNDS",
        "GEMINI_REQUEST_TIMEOUT_SECONDS",
        "DATASENTRY_DEFAULT_EXPLANATION_STYLE",
        "DATASENTRY_DEFAULT_ANALYSIS_FOCUS",
    )

    for key in keys:
        if environment.get(key):
            continue

        try:
            value = st.secrets[key]
        except Exception:
            continue

        if value is not None and str(value).strip():
            environment[key] = str(value)

    return environment


def _load_app_config() -> tuple[
    GeminiConfig | None,
    str | None,
]:
    try:
        config = load_gemini_config(
            environ=_secret_aware_environment(),
            require_api_key=False,
        )
    except ConfigurationError as error:
        return None, str(error)

    return config, None


def _render_sidebar(
    config: GeminiConfig | None,
    config_error: str | None,
) -> tuple[str, str]:
    st.sidebar.header("Copilot settings")

    style = st.sidebar.selectbox(
        "Explanation style",
        options=available_explanation_styles(),
        index=1,
        format_func=lambda value: _STYLE_LABELS[
            value
        ],
        help=(
            "Controls how Gemini communicates the deterministic "
            "audit evidence."
        ),
    )

    focus = st.sidebar.selectbox(
        "Analysis focus",
        options=available_analysis_focuses(),
        index=0,
        format_func=lambda value: _FOCUS_LABELS[
            value
        ],
        help=(
            "Changes interpretation emphasis, not the underlying "
            "deterministic audit."
        ),
    )

    st.sidebar.divider()
    st.sidebar.subheader("Gemini status")

    if config_error is not None:
        st.sidebar.error(config_error)
    elif config is None:
        st.sidebar.error(
            "Gemini configuration is unavailable."
        )
    else:
        st.sidebar.caption(f"Model: `{config.model}`")

        if config.has_api_key:
            st.sidebar.success("API key configured")
        else:
            st.sidebar.warning(
                "API key is not configured. The deterministic "
                "audit remains available, but chat is disabled."
            )

        for warning in config.configuration_warnings:
            st.sidebar.warning(warning)

    st.sidebar.divider()
    st.sidebar.caption(
        "The quality score is heuristic. DataSentry does not "
        "modify or save a cleaned dataset."
    )

    return style, focus


def _sync_upload_state(
    uploaded_file: Any | None,
) -> str | None:
    if uploaded_file is None:
        if st.session_state.get(
            "active_upload_key"
        ) is not None:
            _reset_audit_state(
                active_upload_key=None
            )

        return None

    current_key = uploaded_file_key(uploaded_file)

    if (
        st.session_state.get("active_upload_key")
        != current_key
    ):
        _reset_audit_state(
            active_upload_key=current_key
        )

    return current_key


def _display_readiness_callout(
    summary: Mapping[str, Any],
) -> None:
    label, level = readiness_display(
        str(summary["readiness_status"])
    )
    message = f"**{label}.** {summary['headline']}"

    getattr(st, level)(message)


def _render_summary_tab(
    artifacts: AuditArtifacts,
) -> None:
    report = artifacts.report
    summary = report["summary"]
    overview = report["overview"]
    metadata = report["file_metadata"]

    _display_readiness_callout(summary)

    metric_columns = st.columns(5)
    metric_columns[0].metric(
        "Quality score",
        f"{summary['overall_score']:.2f}",
    )
    metric_columns[1].metric(
        "Rows",
        f"{overview['row_count']:,}",
    )
    metric_columns[2].metric(
        "Columns",
        f"{overview['column_count']:,}",
    )
    metric_columns[3].metric(
        "Issues",
        f"{summary['issue_count_total']:,}",
    )
    metric_columns[4].metric(
        "Highest severity",
        summary["highest_severity"],
    )

    st.subheader("Component scores")
    component_frame = component_scores_frame(report)

    if component_frame.empty:
        st.info("No component scores are available.")
    else:
        chart_frame = component_chart_frame(
            report
        )
        st.bar_chart(
            chart_frame,
            x="component_label",
            y="score",
            x_label="Component",
            y_label="Quality score",
            horizontal=True,
            sort=False,
            width="stretch",
            height=320,
        )
        st.dataframe(
            component_frame,
            hide_index=True,
            use_container_width=True,
        )

    st.subheader("Issue severity")
    counts = summary["issue_counts"]
    severity_columns = st.columns(4)

    for column, severity in zip(
        severity_columns,
        ("CRITICAL", "HIGH", "MEDIUM", "LOW"),
    ):
        column.metric(
            severity.title(),
            int(counts.get(severity, 0)),
        )

    st.subheader(
        f"Data preview — first {len(artifacts.preview)} rows"
    )
    st.dataframe(
        artifacts.preview,
        hide_index=True,
        use_container_width=True,
    )

    with st.expander("File and audit metadata"):
        st.json(
            {
                "report_id": report["report_id"],
                "generated_at_utc": report[
                    "generated_at_utc"
                ],
                "file_metadata": metadata,
                "overview": overview,
            },
            expanded=False,
        )

    warnings = metadata.get("loader_warnings", [])

    if warnings:
        with st.expander(
            f"Loader warnings ({len(warnings)})"
        ):
            for warning in warnings:
                st.warning(str(warning))


def _render_issues_tab(
    artifacts: AuditArtifacts,
) -> None:
    report = artifacts.report
    issues = report["issues"]

    if not issues:
        st.success(
            "No issues were produced by the current generic checks."
        )
        return

    report_id = artifacts.report_id
    severity_options = [
        severity
        for severity in (
            "CRITICAL",
            "HIGH",
            "MEDIUM",
            "LOW",
        )
        if any(
            issue["severity"] == severity
            for issue in issues
        )
    ]
    check_options = sorted(
        {
            str(issue["check_name"])
            for issue in issues
        }
    )
    column_options = sorted(
        {
            str(issue["column"])
            for issue in issues
            if issue.get("column") is not None
        }
    )

    filter_columns = st.columns(3)
    selected_severities = filter_columns[
        0
    ].multiselect(
        "Severity",
        options=severity_options,
        default=severity_options,
        key=f"severity_filter_{report_id}",
    )
    selected_check = filter_columns[1].selectbox(
        "Check",
        options=["All", *check_options],
        key=f"check_filter_{report_id}",
    )
    selected_column = filter_columns[
        2
    ].selectbox(
        "Column",
        options=["All", *column_options],
        key=f"column_filter_{report_id}",
    )

    filtered = filter_issue_records(
        issues,
        severities=selected_severities,
        check_name=(
            None
            if selected_check == "All"
            else selected_check
        ),
        column_name=(
            None
            if selected_column == "All"
            else selected_column
        ),
    )
    frame = issues_frame({"issues": filtered})

    st.caption(
        f"Showing {len(filtered)} of {len(issues)} issues."
    )
    st.dataframe(
        frame,
        hide_index=True,
        use_container_width=True,
    )

    st.download_button(
        "Download filtered issues as CSV",
        data=frame.to_csv(index=False).encode("utf-8"),
        file_name="datasentry_filtered_issues.csv",
        mime="text/csv",
        disabled=frame.empty,
        key=f"download_issues_{report_id}",
    )


def _render_recommendations_tab(
    artifacts: AuditArtifacts,
) -> None:
    frame = recommendations_frame(
        artifacts.report
    )

    if frame.empty:
        st.success(
            "No prioritized remediation actions were generated."
        )
        return

    st.caption(
        "Recommendations are heuristic and require human or "
        "domain review before execution."
    )
    st.dataframe(
        frame,
        hide_index=True,
        use_container_width=True,
    )


def _render_report_tab(
    artifacts: AuditArtifacts,
) -> None:
    report_json = audit_report_to_json(
        artifacts.report,
        indent=2,
    )

    st.download_button(
        "Download structured audit report",
        data=report_json.encode("utf-8"),
        file_name=report_download_name(
            artifacts.report
        ),
        mime="application/json",
        key=f"download_report_{artifacts.report_id}",
    )

    with st.expander("View JSON report"):
        st.code(
            report_json,
            language="json",
        )


def _get_or_create_gemini_client(
    *,
    config: GeminiConfig,
    artifacts: AuditArtifacts,
    explanation_style: str,
    analysis_focus: str,
) -> DataSentryGeminiClient:
    existing = st.session_state.get(
        "gemini_client"
    )
    existing_report_id = st.session_state.get(
        "gemini_client_report_id"
    )

    if (
        existing is not None
        and existing_report_id
        == artifacts.report_id
    ):
        existing.set_response_profile(
            explanation_style=explanation_style,
            analysis_focus=analysis_focus,
        )
        return existing

    _close_session_client()

    client = DataSentryGeminiClient(
        config=config,
        toolbox=AuditToolbox(
            artifacts.report
        ),
        explanation_style=explanation_style,
        analysis_focus=analysis_focus,
    )
    st.session_state["gemini_client"] = client
    st.session_state[
        "gemini_client_report_id"
    ] = artifacts.report_id

    return client


def _append_chat_record(
    *,
    role: str,
    content: str,
    kind: str = "message",
    details: Mapping[str, Any] | None = None,
) -> None:
    st.session_state["chat_records"].append(
        {
            "role": role,
            "content": content,
            "kind": kind,
            "details": (
                dict(details)
                if details is not None
                else None
            ),
        }
    )


def _render_chat_history() -> None:
    for record in st.session_state[
        "chat_records"
    ]:
        with st.chat_message(record["role"]):
            if record["kind"] == "error":
                st.error(record["content"])
            else:
                st.markdown(record["content"])

            details = record.get("details")

            if details:
                with st.expander(
                    "Tool activity and response metadata"
                ):
                    st.json(details)


def _render_copilot_tab(
    artifacts: AuditArtifacts,
    *,
    config: GeminiConfig | None,
    config_error: str | None,
    explanation_style: str,
    analysis_focus: str,
) -> None:
    st.caption(
        "Gemini can only access the structured audit through "
        "registered read-only tools."
    )

    if st.button(
        "Clear conversation",
        key=f"clear_chat_{artifacts.report_id}",
        disabled=not st.session_state[
            "chat_records"
        ],
    ):
        client = st.session_state.get(
            "gemini_client"
        )

        if client is not None:
            client.reset_history()

        st.session_state["chat_records"] = []
        st.rerun()

    _render_chat_history()

    if config_error is not None:
        st.error(config_error)
        return

    if config is None or not config.has_api_key:
        st.warning(
            "Configure GEMINI_API_KEY in `.env`, the runtime "
            "environment, or Streamlit secrets to enable chat."
        )
        return

    prompt = st.chat_input(
        "Ask about the audited dataset",
        key=f"chat_input_{artifacts.report_id}",
    )

    if not prompt:
        return

    _append_chat_record(
        role="user",
        content=prompt,
    )

    try:
        client = _get_or_create_gemini_client(
            config=config,
            artifacts=artifacts,
            explanation_style=explanation_style,
            analysis_focus=analysis_focus,
        )

        with st.spinner(
            "Gemini is consulting the read-only audit tools..."
        ):
            result = client.send_message(
                prompt,
                explanation_style=explanation_style,
                analysis_focus=analysis_focus,
            )
    except Exception as error:
        LOGGER.exception(
            "Gemini chat orchestration failed."
        )
        safe_message = (
            str(error)
            if isinstance(error, ConfigurationError)
            else (
                "The copilot could not be initialized. "
                "Review the Gemini configuration and try again."
            )
        )
        _append_chat_record(
            role="assistant",
            content=safe_message,
            kind="error",
        )
        st.rerun()
        return

    if result["ok"]:
        assistant_text = format_copilot_response(
            str(result["text"]),
            explanation_style=explanation_style,
        )

        details = {
            "model": result.get("model"),
            "finish_reason": result.get(
                "finish_reason"
            ),
            "tool_rounds": result.get(
                "tool_rounds"
            ),
            "tool_call_count": result.get(
                "tool_call_count"
            ),
            "tool_calls": result.get(
                "tool_calls",
                [],
            ),
            "usage": result.get("usage", {}),
            "response_profile": result.get(
                "response_profile"
            ),
        }
        _append_chat_record(
            role="assistant",
            content=assistant_text,
            details=details,
        )
    else:
        _append_chat_record(
            role="assistant",
            content=str(result["message"]),
            kind="error",
            details={
                "error": result.get("error"),
                "tool_calls": result.get(
                    "tool_calls",
                    [],
                ),
            },
        )

    st.rerun()


def _render_dashboard(
    artifacts: AuditArtifacts,
    *,
    config: GeminiConfig | None,
    config_error: str | None,
    explanation_style: str,
    analysis_focus: str,
) -> None:
    st.divider()
    st.header("Audit results")

    tab_state_key = dashboard_tab_state_key(
        artifacts.report_id
    )

    summary_tab, issues_tab, actions_tab, report_tab, ai_tab = (
        st.tabs(
            list(DASHBOARD_TAB_LABELS),
            default="Summary",
            key=tab_state_key,
            on_change="rerun",
        )
    )

    if summary_tab.open:
        with summary_tab:
            _render_summary_tab(artifacts)

    if issues_tab.open:
        with issues_tab:
            _render_issues_tab(artifacts)

    if actions_tab.open:
        with actions_tab:
            _render_recommendations_tab(
                artifacts
            )

    if report_tab.open:
        with report_tab:
            _render_report_tab(artifacts)

    if ai_tab.open:
        with ai_tab:
            _render_copilot_tab(
                artifacts,
                config=config,
                config_error=config_error,
                explanation_style=explanation_style,
                analysis_focus=analysis_focus,
            )


def _safe_audit_error(
    error: BaseException,
) -> str:
    validation_error = getattr(
        data_loader_module,
        "CSVValidationError",
        None,
    )

    if (
        isinstance(validation_error, type)
        and isinstance(error, validation_error)
    ):
        return str(error)

    if isinstance(
        error,
        AppOrchestrationError,
    ):
        return str(error)

    LOGGER.exception(
        "Deterministic audit failed."
    )

    return (
        "The audit could not be completed. Review the CSV format "
        "and application logs, then try again."
    )


def main() -> None:
    st.set_page_config(
        page_title=f"{APP_TITLE} — {APP_SUBTITLE}",
        page_icon="🛡️",
        layout="wide",
    )

    _initialize_session_state()
    config, config_error = _load_app_config()
    explanation_style, analysis_focus = (
        _render_sidebar(
            config,
            config_error,
        )
    )

    st.title(APP_TITLE)
    st.subheader(APP_SUBTITLE)
    st.write(
        "Upload one CSV to run a deterministic, read-only quality "
        "audit. Gemini explains the resulting evidence through "
        "registered tools; it does not receive the source DataFrame."
    )

    uploaded_file = st.file_uploader(
        "Upload a CSV file",
        type=["csv"],
        accept_multiple_files=False,
        help=(
            "The file is validated by src.data_loader before any "
            "quality checks run."
        ),
    )

    current_key = _sync_upload_state(
        uploaded_file
    )

    run_audit = st.button(
        "Run deterministic audit",
        type="primary",
        disabled=uploaded_file is None,
    )

    if run_audit and uploaded_file is not None:
        try:
            with st.spinner(
                "Validating CSV and running quality checks..."
            ):
                artifacts = build_audit_artifacts(
                    uploaded_file
                )

            st.session_state[
                "audit_artifacts"
            ] = artifacts
            st.session_state[
                "active_upload_key"
            ] = current_key
            st.session_state[
                "last_audit_error"
            ] = None
            st.session_state[
                "chat_records"
            ] = []
            _close_session_client()
            st.success(
                "Deterministic audit completed."
            )
        except Exception as error:
            message = _safe_audit_error(error)
            st.session_state[
                "last_audit_error"
            ] = message
            st.session_state[
                "audit_artifacts"
            ] = None
            _close_session_client()

    audit_error = st.session_state.get(
        "last_audit_error"
    )

    if audit_error:
        st.error(audit_error)

    artifacts = st.session_state.get(
        "audit_artifacts"
    )

    if artifacts is None:
        if uploaded_file is None:
            st.info(
                "Upload a CSV file to begin."
            )
        else:
            st.info(
                "The selected file has not been audited yet."
            )
        return

    _render_dashboard(
        artifacts,
        config=config,
        config_error=config_error,
        explanation_style=explanation_style,
        analysis_focus=analysis_focus,
    )


if __name__ == "__main__":
    main()
