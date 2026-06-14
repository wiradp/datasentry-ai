from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd
import pytest

from src.quality_checks import run_all_quality_checks
from src.quality_score import calculate_quality_score
from src.report_builder import (
    ReportBuilderConfig,
    ReportBuilderError,
    audit_report_to_json,
    build_audit_report,
    build_audit_report_from_results,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_PATH = (
    PROJECT_ROOT
    / "data"
    / "sample_dirty_customers.csv"
)
FIXED_TIME = datetime(
    2026,
    6,
    14,
    12,
    0,
    tzinfo=timezone.utc,
)


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_PATH)


@pytest.fixture
def clean_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": list(range(20, 40)),
            "city": [
                "Jakarta",
                "Bandung",
                "Surabaya",
                "Medan",
            ] * 5,
            "purchases": [
                float(100 + index * 50)
                for index in range(20)
            ],
        }
    )


@pytest.fixture
def sample_metadata() -> dict[str, object]:
    return {
        "file_name": "sample_dirty_customers.csv",
        "file_size_bytes": 19822,
        "encoding": "utf-8-sig",
        "delimiter": ",",
        "fingerprint_sha256": "a" * 64,
        "warnings": [
            "1 kolom 'Unnamed' terdeteksi."
        ],
        "row_count": 208,
        "column_count": 13,
    }


def test_builds_expected_top_level_structure(
    sample_dataframe: pd.DataFrame,
    sample_metadata: dict[str, object],
) -> None:
    report = build_audit_report(
        sample_dataframe,
        file_metadata=sample_metadata,
        generated_at_utc=FIXED_TIME,
    )

    assert set(report) >= {
        "report_version",
        "report_id",
        "generated_at_utc",
        "file_metadata",
        "summary",
        "overview",
        "component_scores",
        "issues",
        "prioritized_recommendations",
        "methodology",
        "limitations",
        "disclaimer",
        "quality_check_results",
    }


def test_sample_report_matches_expected_score_and_status(
    sample_dataframe: pd.DataFrame,
) -> None:
    report = build_audit_report(
        sample_dataframe,
        generated_at_utc=FIXED_TIME,
    )

    assert report["summary"]["overall_score"] == 92.02
    assert report["summary"]["score_band_status"] == (
        "READY_WITH_MINOR_REVIEW"
    )
    assert report["summary"]["readiness_status"] == (
        "NEEDS_CLEANING"
    )
    assert report["summary"][
        "status_adjusted_by_gates"
    ] is True


def test_issue_table_matches_score_counts(
    sample_dataframe: pd.DataFrame,
) -> None:
    report = build_audit_report(
        sample_dataframe,
        generated_at_utc=FIXED_TIME,
    )

    assert len(report["issues"]) == 27
    assert report["summary"]["issue_count_total"] == 27
    assert report["summary"]["issue_counts"] == {
        "CRITICAL": 0,
        "HIGH": 14,
        "MEDIUM": 7,
        "LOW": 6,
    }


def test_issue_ids_and_priority_ranks_are_sequential(
    sample_dataframe: pd.DataFrame,
) -> None:
    report = build_audit_report(
        sample_dataframe,
        generated_at_utc=FIXED_TIME,
    )

    assert [
        issue["issue_id"]
        for issue in report["issues"]
    ] == [
        f"ISSUE-{index:03d}"
        for index in range(1, 28)
    ]

    assert [
        issue["priority_rank"]
        for issue in report["issues"]
    ] == list(range(1, 28))


def test_issues_are_sorted_by_severity(
    sample_dataframe: pd.DataFrame,
) -> None:
    report = build_audit_report(
        sample_dataframe,
        generated_at_utc=FIXED_TIME,
    )

    severity_rank = {
        "CRITICAL": 0,
        "HIGH": 1,
        "MEDIUM": 2,
        "LOW": 3,
    }

    ranks = [
        severity_rank[issue["severity"]]
        for issue in report["issues"]
    ]

    assert ranks == sorted(ranks)


def test_pass_missing_columns_are_not_issues(
    sample_dataframe: pd.DataFrame,
) -> None:
    report = build_audit_report(
        sample_dataframe,
        generated_at_utc=FIXED_TIME,
    )

    missing_columns = {
        issue["column"]
        for issue in report["issues"]
        if issue["check_name"] == "missing_values"
    }

    assert missing_columns == {
        "age",
        "annual_income",
        "membership_type",
        "monthly_visits",
        "referral_code",
        "signup_date",
    }


def test_duplicate_issue_is_normalized(
    sample_dataframe: pd.DataFrame,
) -> None:
    report = build_audit_report(
        sample_dataframe,
        generated_at_utc=FIXED_TIME,
    )

    duplicates = [
        issue
        for issue in report["issues"]
        if issue["check_name"] == "duplicates"
    ]

    assert len(duplicates) == 1
    assert duplicates[0]["column"] is None
    assert duplicates[0]["count"] == 8
    assert duplicates[0]["percentage"] == 3.85
    assert duplicates[0]["issue_type"] == (
        "EXACT_DUPLICATE_ROWS"
    )


def test_file_metadata_is_preserved(
    sample_dataframe: pd.DataFrame,
    sample_metadata: dict[str, object],
) -> None:
    report = build_audit_report(
        sample_dataframe,
        file_metadata=sample_metadata,
        generated_at_utc=FIXED_TIME,
    )

    metadata = report["file_metadata"]

    assert metadata["file_name"] == (
        "sample_dirty_customers.csv"
    )
    assert metadata["fingerprint_sha256"] == "a" * 64
    assert metadata["row_count"] == 208
    assert metadata["column_count"] == 13
    assert metadata["loader_warnings"] == [
        "1 kolom 'Unnamed' terdeteksi."
    ]


@dataclass
class LoaderResultLike:
    file_name: str
    file_size_bytes: int
    encoding: str
    delimiter: str
    fingerprint_sha256: str
    warnings: list[str]
    row_count: int
    column_count: int


def test_accepts_loader_result_like_metadata(
    sample_dataframe: pd.DataFrame,
) -> None:
    metadata = LoaderResultLike(
        file_name="customers.csv",
        file_size_bytes=100,
        encoding="utf-8",
        delimiter=",",
        fingerprint_sha256="b" * 64,
        warnings=[],
        row_count=208,
        column_count=13,
    )

    report = build_audit_report(
        sample_dataframe,
        file_metadata=metadata,
        generated_at_utc=FIXED_TIME,
    )

    assert report["file_metadata"]["file_name"] == (
        "customers.csv"
    )
    assert report["file_metadata"][
        "fingerprint_sha256"
    ] == "b" * 64


def test_rejects_metadata_dimension_mismatch(
    sample_dataframe: pd.DataFrame,
) -> None:
    with pytest.raises(
        ReportBuilderError,
        match="row_count does not match",
    ):
        build_audit_report(
            sample_dataframe,
            file_metadata={
                "row_count": 999,
                "column_count": 13,
            },
            generated_at_utc=FIXED_TIME,
        )


def test_prioritized_recommendations_respect_limit(
    sample_dataframe: pd.DataFrame,
) -> None:
    report = build_audit_report(
        sample_dataframe,
        generated_at_utc=FIXED_TIME,
        report_config=ReportBuilderConfig(
            max_prioritized_recommendations=5,
        ),
    )

    recommendations = report[
        "prioritized_recommendations"
    ]

    assert len(recommendations) == 5
    assert [
        item["priority_rank"]
        for item in recommendations
    ] == [1, 2, 3, 4, 5]
    assert len(
        {
            (
                item["column"],
                item["action"],
            )
            for item in recommendations
        }
    ) == 5


def test_clean_dataset_has_no_issues(
    clean_dataframe: pd.DataFrame,
) -> None:
    report = build_audit_report(
        clean_dataframe,
        generated_at_utc=FIXED_TIME,
    )

    assert report["summary"]["overall_score"] == 100.0
    assert report["summary"]["readiness_status"] == (
        "READY_WITH_MINOR_REVIEW"
    )
    assert report["summary"]["issue_count_total"] == 0
    assert report["summary"]["highest_severity"] == "PASS"
    assert report["issues"] == []
    assert report["prioritized_recommendations"] == []


def test_report_is_json_serializable_without_nan(
    sample_dataframe: pd.DataFrame,
) -> None:
    report = build_audit_report(
        sample_dataframe,
        generated_at_utc=FIXED_TIME,
    )

    payload = audit_report_to_json(report)

    parsed = json.loads(payload)

    assert parsed["report_id"] == report["report_id"]
    assert "NaN" not in payload
    assert "Infinity" not in payload
    assert (
        parsed["quality_check_results"]
        ["data_type_warnings"][1]
        ["parse_success_count"]
        is None
    )


def test_report_is_deterministic_with_fixed_time(
    sample_dataframe: pd.DataFrame,
) -> None:
    first = build_audit_report(
        sample_dataframe,
        generated_at_utc=FIXED_TIME,
    )
    second = build_audit_report(
        sample_dataframe,
        generated_at_utc=FIXED_TIME,
    )

    assert first == second
    assert first["generated_at_utc"] == (
        "2026-06-14T12:00:00Z"
    )


def test_build_from_precomputed_results_matches_direct_build(
    sample_dataframe: pd.DataFrame,
    sample_metadata: dict[str, object],
) -> None:
    checks = run_all_quality_checks(sample_dataframe)
    score = calculate_quality_score(checks)

    precomputed = build_audit_report_from_results(
        checks,
        score,
        file_metadata=sample_metadata,
        generated_at_utc=FIXED_TIME,
    )
    direct = build_audit_report(
        sample_dataframe,
        file_metadata=sample_metadata,
        generated_at_utc=FIXED_TIME,
    )

    assert precomputed == direct


def test_builder_does_not_mutate_inputs(
    sample_dataframe: pd.DataFrame,
) -> None:
    dataframe_copy = sample_dataframe.copy(deep=True)
    checks = run_all_quality_checks(sample_dataframe)
    score = calculate_quality_score(checks)
    checks_copy = deepcopy(checks)
    score_copy = deepcopy(score)

    build_audit_report_from_results(
        checks,
        score,
        generated_at_utc=FIXED_TIME,
    )

    pd.testing.assert_frame_equal(
        sample_dataframe,
        dataframe_copy,
    )
    assert checks["overview"] == checks_copy["overview"]
    assert checks["duplicates"] == checks_copy["duplicates"]
    assert score == score_copy

    for key, value in checks.items():
        if isinstance(value, pd.DataFrame):
            pd.testing.assert_frame_equal(
                value,
                checks_copy[key],
            )


def test_rejects_missing_check_result(
    sample_dataframe: pd.DataFrame,
) -> None:
    checks = run_all_quality_checks(sample_dataframe)
    score = calculate_quality_score(checks)
    checks.pop("data_type_warnings")

    with pytest.raises(
        ReportBuilderError,
        match="Missing quality-check results",
    ):
        build_audit_report_from_results(
            checks,
            score,
            generated_at_utc=FIXED_TIME,
        )


def test_can_exclude_raw_check_results(
    clean_dataframe: pd.DataFrame,
) -> None:
    report = build_audit_report(
        clean_dataframe,
        generated_at_utc=FIXED_TIME,
        report_config=ReportBuilderConfig(
            include_raw_check_results=False,
        ),
    )

    assert "quality_check_results" not in report
