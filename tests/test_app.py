from __future__ import annotations

from io import BytesIO
from types import ModuleType, SimpleNamespace
from typing import Any

import pandas as pd
import pytest

import app


class FakeUploadedFile(BytesIO):
    def __init__(
        self,
        payload: bytes,
        *,
        name: str = "sample.csv",
    ) -> None:
        super().__init__(payload)
        self.name = name
        self.size = len(payload)

    def getvalue(self) -> bytes:
        return super().getvalue()


class FakeLoaderResult:
    def __init__(
        self,
        dataframe: pd.DataFrame,
        *,
        file_name: str = "sample.csv",
        warnings: list[str] | None = None,
    ) -> None:
        self.dataframe = dataframe
        self.file_name = file_name
        self.file_size_bytes = 123
        self.encoding = "utf-8"
        self.delimiter = ","
        self.fingerprint_sha256 = "a" * 64
        self.row_count = len(dataframe)
        self.column_count = len(dataframe.columns)
        self.warnings = warnings or []


@pytest.fixture
def sample_payload() -> bytes:
    with open(
        "data/sample_dirty_customers.csv",
        "rb",
    ) as handle:
        return handle.read()


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    return pd.read_csv(
        "data/sample_dirty_customers.csv"
    )


def test_uploaded_file_key_is_deterministic(
    sample_payload: bytes,
) -> None:
    first = FakeUploadedFile(
        sample_payload,
        name="customers.csv",
    )
    second = FakeUploadedFile(
        sample_payload,
        name="customers.csv",
    )

    assert app.uploaded_file_key(
        first
    ) == app.uploaded_file_key(second)


def test_uploaded_file_key_changes_with_name_or_content() -> None:
    base = FakeUploadedFile(
        b"a,b\n1,2\n",
        name="first.csv",
    )
    changed_name = FakeUploadedFile(
        b"a,b\n1,2\n",
        name="second.csv",
    )
    changed_content = FakeUploadedFile(
        b"a,b\n1,3\n",
        name="first.csv",
    )

    assert app.uploaded_file_key(base) != (
        app.uploaded_file_key(changed_name)
    )
    assert app.uploaded_file_key(base) != (
        app.uploaded_file_key(changed_content)
    )


def test_resolve_csv_loader_prefers_named_public_function() -> None:
    module = ModuleType("fake_loader")

    def load_csv(source: Any) -> Any:
        return source

    def csv_load_fallback(source: Any) -> Any:
        return source

    module.load_csv = load_csv
    module.csv_load_fallback = csv_load_fallback

    assert app.resolve_csv_loader(module) is load_csv


def test_resolve_csv_loader_accepts_one_conservative_fallback() -> None:
    module = ModuleType("fake_loader")

    def validated_csv_reader(source: Any) -> Any:
        return source

    module.validated_csv_reader = validated_csv_reader

    assert (
        app.resolve_csv_loader(module)
        is validated_csv_reader
    )


def test_resolve_csv_loader_rejects_ambiguity() -> None:
    module = ModuleType("fake_loader")

    def first_csv_reader(source: Any) -> Any:
        return source

    def second_csv_loader(source: Any) -> Any:
        return source

    module.first_csv_reader = first_csv_reader
    module.second_csv_loader = second_csv_loader

    with pytest.raises(
        app.AppOrchestrationError,
        match="No unambiguous public CSV loader",
    ):
        app.resolve_csv_loader(module)


def test_extract_dataframe_supports_object_and_mapping(
    sample_dataframe: pd.DataFrame,
) -> None:
    object_result = SimpleNamespace(
        dataframe=sample_dataframe
    )
    mapping_result = {"df": sample_dataframe}

    assert app.extract_dataframe(
        object_result
    ) is sample_dataframe
    assert app.extract_dataframe(
        mapping_result
    ) is sample_dataframe


def test_extract_dataframe_rejects_missing_or_empty_data() -> None:
    with pytest.raises(
        app.AppOrchestrationError,
        match="does not expose",
    ):
        app.extract_dataframe({"value": 1})

    with pytest.raises(
        app.AppOrchestrationError,
        match="no data rows",
    ):
        app.extract_dataframe(
            pd.DataFrame(columns=["a"])
        )


def test_build_audit_artifacts_runs_full_pipeline(
    sample_payload: bytes,
) -> None:
    uploaded = FakeUploadedFile(
        sample_payload,
        name="sample_dirty_customers.csv",
    )

    def fake_loader(
        source: FakeUploadedFile,
    ) -> FakeLoaderResult:
        source.seek(0)
        dataframe = pd.read_csv(source)

        return FakeLoaderResult(
            dataframe,
            file_name=source.name,
            warnings=[
                "Example loader warning."
            ],
        )

    artifacts = app.build_audit_artifacts(
        uploaded,
        loader_callable=fake_loader,
        preview_rows=7,
    )

    assert artifacts.report["summary"][
        "overall_score"
    ] == 92.02
    assert artifacts.report["summary"][
        "readiness_status"
    ] == "NEEDS_CLEANING"
    assert artifacts.report["summary"][
        "issue_count_total"
    ] == 27
    assert len(artifacts.preview) == 7
    assert (
        artifacts.report["file_metadata"][
            "file_name"
        ]
        == "sample_dirty_customers.csv"
    )
    assert artifacts.report["file_metadata"][
        "loader_warnings"
    ] == ["Example loader warning."]
    assert "quality_check_results" not in (
        artifacts.report
    )
    assert not hasattr(
        artifacts,
        "dataframe",
    )


@pytest.mark.parametrize(
    "preview_rows",
    [0, 101, 1.5, True],
)
def test_build_audit_artifacts_rejects_invalid_preview_limit(
    sample_payload: bytes,
    preview_rows: Any,
) -> None:
    uploaded = FakeUploadedFile(sample_payload)

    with pytest.raises(
        app.AppOrchestrationError,
        match="preview_rows",
    ):
        app.build_audit_artifacts(
            uploaded,
            loader_callable=lambda source: None,
            preview_rows=preview_rows,
        )


def test_component_scores_frame_has_expected_values(
    sample_payload: bytes,
) -> None:
    uploaded = FakeUploadedFile(sample_payload)

    def fake_loader(source: FakeUploadedFile) -> Any:
        source.seek(0)
        dataframe = pd.read_csv(source)
        return FakeLoaderResult(dataframe)

    report = app.build_audit_artifacts(
        uploaded,
        loader_callable=fake_loader,
    ).report
    frame = app.component_scores_frame(report)

    assert set(frame["component"]) == {
        "completeness",
        "duplicate_free",
        "schema_usability",
        "categorical_consistency",
        "outlier_risk",
    }
    assert frame.loc[
        frame["component"] == "completeness",
        "score",
    ].iloc[0] == 97.56


def test_issue_and_recommendation_frames_are_stable(
    sample_payload: bytes,
) -> None:
    uploaded = FakeUploadedFile(sample_payload)

    def fake_loader(source: FakeUploadedFile) -> Any:
        source.seek(0)
        dataframe = pd.read_csv(source)
        return FakeLoaderResult(dataframe)

    report = app.build_audit_artifacts(
        uploaded,
        loader_callable=fake_loader,
    ).report
    issues = app.issues_frame(report)
    recommendations = app.recommendations_frame(
        report
    )

    assert len(issues) == 27
    assert issues.iloc[0]["severity"] == "HIGH"
    assert not recommendations.empty
    assert recommendations.iloc[0]["priority"] == "P1"


def test_filter_issue_records_is_read_only() -> None:
    issues = [
        {
            "severity": "HIGH",
            "check_name": "missing_values",
            "column": "age",
        },
        {
            "severity": "LOW",
            "check_name": "numeric_outliers",
            "column": "income",
        },
    ]
    original = [dict(item) for item in issues]

    filtered = app.filter_issue_records(
        issues,
        severities=["HIGH"],
        check_name="missing_values",
        column_name="age",
    )

    assert filtered == [issues[0]]
    assert issues == original
    assert filtered[0] is not issues[0]


def test_report_download_name_is_sanitized() -> None:
    report = {
        "file_metadata": {
            "file_name": "customer data (final).csv"
        }
    }

    assert app.report_download_name(report) == (
        "datasentry_audit_customer_data__final.json"
    )


@pytest.mark.parametrize(
    ("status", "label", "level"),
    [
        (
            "READY_WITH_MINOR_REVIEW",
            "Ready with minor review",
            "success",
        ),
        (
            "NEEDS_CLEANING",
            "Needs cleaning",
            "warning",
        ),
        (
            "NOT_READY",
            "Not ready",
            "error",
        ),
    ],
)
def test_readiness_display(
    status: str,
    label: str,
    level: str,
) -> None:
    assert app.readiness_display(status) == (
        label,
        level,
    )


def test_dashboard_tab_labels_are_stable() -> None:
    assert app.DASHBOARD_TAB_LABELS == (
        "Summary",
        "Issues",
        "Recommendations",
        "Report",
        "AI Copilot",
    )


def test_dashboard_tab_state_key_is_report_scoped() -> None:
    first = app.dashboard_tab_state_key(
        "DSA-report-001"
    )
    second = app.dashboard_tab_state_key(
        "DSA-report-002"
    )

    assert first == "dashboard_tab_DSA-report-001"
    assert second == "dashboard_tab_DSA-report-002"
    assert first != second


def test_dashboard_tab_state_key_sanitizes_unsafe_characters() -> None:
    assert app.dashboard_tab_state_key(
        "DSA report/001"
    ) == "dashboard_tab_DSA_report_001"


def test_dashboard_tab_state_key_rejects_empty_id() -> None:
    with pytest.raises(
        app.AppOrchestrationError,
        match="report_id cannot be empty",
    ):
        app.dashboard_tab_state_key("   ")


@pytest.mark.parametrize(
    ("raw_name", "expected_label"),
    [
        (
            "categorical_consistency",
            "Categorical Consistency",
        ),
        (
            "duplicate_free",
            "Duplicate Free",
        ),
        (
            "schema_usability",
            "Schema Usability",
        ),
        (
            "  outlier_risk  ",
            "Outlier Risk",
        ),
    ],
)
def test_format_component_label(
    raw_name: str,
    expected_label: str,
) -> None:
    assert app.format_component_label(
        raw_name
    ) == expected_label


def test_component_chart_frame_uses_readable_labels(
    sample_payload: bytes,
) -> None:
    uploaded = FakeUploadedFile(sample_payload)

    def fake_loader(source: FakeUploadedFile) -> Any:
        source.seek(0)
        dataframe = pd.read_csv(source)
        return FakeLoaderResult(dataframe)

    report = app.build_audit_artifacts(
        uploaded,
        loader_callable=fake_loader,
    ).report
    frame = app.component_chart_frame(report)

    assert list(frame.columns) == [
        "component_label",
        "score",
    ]
    assert list(frame["component_label"]) == [
        "Completeness",
        "Duplicate Free",
        "Schema Usability",
        "Categorical Consistency",
        "Outlier Risk",
    ]
    assert frame["score"].between(
        0,
        100,
    ).all()


def test_component_chart_frame_handles_empty_report() -> None:
    frame = app.component_chart_frame(
        {"component_scores": {}}
    )

    assert frame.empty
    assert list(frame.columns) == [
        "component_label",
        "score",
    ]
