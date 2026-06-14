from copy import deepcopy
import inspect
import json
from pathlib import Path

import pandas as pd
import pytest

from src.report_builder import (
    ReportBuilderConfig,
    build_audit_report,
)
from src.tools import (
    AUDIT_TOOL_NAMES,
    MAX_RESULT_LIMIT,
    AuditToolError,
    AuditToolbox,
    build_audit_toolbox,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_PATH = (
    PROJECT_ROOT
    / "data"
    / "sample_dirty_customers.csv"
)
FIXED_TIME = "2026-06-14T12:00:00Z"


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
def sample_report(
    sample_dataframe: pd.DataFrame,
) -> dict[str, object]:
    return build_audit_report(
        sample_dataframe,
        file_metadata={
            "file_name": "sample_dirty_customers.csv",
            "row_count": 208,
            "column_count": 13,
            "encoding": "utf-8-sig",
            "delimiter": ",",
            "fingerprint_sha256": "a" * 64,
            "warnings": [
                "An unnamed index-like column was detected."
            ],
        },
        generated_at_utc=FIXED_TIME,
    )


@pytest.fixture
def clean_report(
    clean_dataframe: pd.DataFrame,
) -> dict[str, object]:
    return build_audit_report(
        clean_dataframe,
        generated_at_utc=FIXED_TIME,
    )


@pytest.fixture
def toolbox(
    sample_report: dict[str, object],
) -> AuditToolbox:
    return build_audit_toolbox(sample_report)


def assert_strict_json_safe(value: object) -> None:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
    )
    assert "NaN" not in payload
    assert "Infinity" not in payload


def test_tool_registry_has_all_required_tools(
    toolbox: AuditToolbox,
) -> None:
    registry = toolbox.get_tool_registry()

    assert tuple(registry) == AUDIT_TOOL_NAMES
    assert all(callable(function) for function in registry.values())


def test_all_tools_return_json_safe_dictionaries(
    toolbox: AuditToolbox,
) -> None:
    results = [
        toolbox.get_dataset_overview(),
        toolbox.get_quality_summary(),
        toolbox.get_missing_value_report(),
        toolbox.get_duplicate_report(),
        toolbox.get_column_quality_report("customer_id"),
        toolbox.get_priority_issues(),
        toolbox.get_ml_readiness_report(),
    ]

    for result in results:
        assert isinstance(result, dict)
        assert result["ok"] is True
        assert_strict_json_safe(result)


def test_dataset_overview_returns_dimensions_and_columns(
    toolbox: AuditToolbox,
) -> None:
    result = toolbox.get_dataset_overview()
    data = result["data"]

    assert data["dimensions"] == {
        "row_count": 208,
        "column_count": 13,
        "total_cells": 2704,
    }
    assert data["column_type_counts"] == {
        "numeric": 4,
        "categorical": 9,
        "datetime": 0,
        "boolean": 0,
        "other": 0,
    }
    assert data["file"]["file_name"] == (
        "sample_dirty_customers.csv"
    )
    assert data["basic_quality"][
        "missing_percentage"
    ] == 2.44


def test_quality_summary_returns_score_and_readiness(
    toolbox: AuditToolbox,
) -> None:
    result = toolbox.get_quality_summary()
    data = result["data"]

    assert data["overall_score"] == 92.02
    assert data["score_band_status"] == (
        "READY_WITH_MINOR_REVIEW"
    )
    assert data["readiness_status"] == (
        "NEEDS_CLEANING"
    )
    assert data["status_adjusted_by_gates"] is True
    assert data["issue_counts"] == {
        "CRITICAL": 0,
        "HIGH": 14,
        "MEDIUM": 7,
        "LOW": 6,
    }


def test_missing_report_supports_limit_and_filters(
    toolbox: AuditToolbox,
) -> None:
    result = toolbox.get_missing_value_report(
        limit=2,
        min_missing_percentage=5.0,
    )
    data = result["data"]

    assert data["returned_count"] == 2
    assert data["total_matching_count"] == 3
    assert data["truncated"] is True
    assert {
        item["column"]
        for item in data["items"]
    } <= {
        "annual_income",
        "age",
        "referral_code",
    }
    assert all(
        item["percentage"] >= 5.0
        for item in data["items"]
    )


def test_missing_report_can_return_clear_empty_result(
    toolbox: AuditToolbox,
) -> None:
    result = toolbox.get_missing_value_report(
        column_name="country"
    )

    assert result["ok"] is True
    assert result["data"]["returned_count"] == 0
    assert result["data"]["items"] == []
    assert "No missing-value issues" in result["message"]
    assert "'country'" in result["message"]


@pytest.mark.parametrize(
    "limit",
    [0, MAX_RESULT_LIMIT + 1, 1.5, True],
)
def test_missing_report_rejects_invalid_limit(
    toolbox: AuditToolbox,
    limit: object,
) -> None:
    result = toolbox.get_missing_value_report(
        limit=limit,  # type: ignore[arg-type]
    )

    assert result["ok"] is False
    assert result["data"] is None
    assert result["error"]["code"] in {
        "invalid_limit_type",
        "limit_out_of_range",
    }


def test_missing_report_rejects_invalid_percentage(
    toolbox: AuditToolbox,
) -> None:
    result = toolbox.get_missing_value_report(
        min_missing_percentage=101.0
    )

    assert result["ok"] is False
    assert result["error"]["code"] == (
        "percentage_out_of_range"
    )


def test_duplicate_report_returns_exact_duplicate_evidence(
    toolbox: AuditToolbox,
) -> None:
    result = toolbox.get_duplicate_report()
    data = result["data"]

    assert data["has_exact_duplicates"] is True
    assert data["duplicate_rows"] == 8
    assert data["duplicate_percentage"] == 3.85
    assert data["issue"]["check_name"] == "duplicates"


def test_duplicate_report_has_clear_clean_message(
    clean_report: dict[str, object],
) -> None:
    toolbox = AuditToolbox(clean_report)
    result = toolbox.get_duplicate_report()

    assert result["ok"] is True
    assert result["data"]["has_exact_duplicates"] is False
    assert result["data"]["issue"] is None
    assert "No extra exact duplicate rows" in (
        result["message"]
    )


def test_column_report_returns_all_customer_id_evidence(
    toolbox: AuditToolbox,
) -> None:
    result = toolbox.get_column_quality_report(
        "customer_id"
    )
    data = result["data"]

    assert data["column_name"] == "customer_id"
    assert data["column_type"] == "categorical"
    assert data["is_clean_in_current_audit"] is False
    assert data["issue_count"] == 2
    assert {
        issue["check_name"]
        for issue in data["issues"]
    } == {
        "potential_identifiers",
        "high_cardinality_categories",
    }
    assert data["scoring_diagnostics"][
        "schema_penalty"
    ] == 50.0


def test_column_resolution_is_case_insensitive_when_unique(
    toolbox: AuditToolbox,
) -> None:
    result = toolbox.get_column_quality_report(
        "ANNUAL_INCOME"
    )

    assert result["ok"] is True
    assert result["data"]["column_name"] == (
        "annual_income"
    )


def test_column_report_returns_clear_clean_result(
    toolbox: AuditToolbox,
) -> None:
    result = toolbox.get_column_quality_report("churn")

    assert result["ok"] is True
    assert result["data"]["is_clean_in_current_audit"] is True
    assert result["data"]["issue_count"] == 0
    assert result["data"]["issues"] == []
    assert "No quality issues were found" in result["message"]


def test_column_report_rejects_unknown_column(
    toolbox: AuditToolbox,
) -> None:
    result = toolbox.get_column_quality_report(
        "not_a_real_column"
    )

    assert result["ok"] is False
    assert result["error"]["code"] == "unknown_column"
    assert "customer_id" in result["error"]["details"][
        "available_columns"
    ]


def test_priority_issues_respect_limit_and_severity(
    toolbox: AuditToolbox,
) -> None:
    result = toolbox.get_priority_issues(
        limit=5,
        minimum_severity="HIGH",
    )
    data = result["data"]

    assert data["returned_count"] == 5
    assert data["total_matching_count"] == 14
    assert data["truncated"] is True
    assert all(
        item["severity"] in {"CRITICAL", "HIGH"}
        for item in data["items"]
    )


def test_priority_issues_can_filter_check_and_column(
    toolbox: AuditToolbox,
) -> None:
    result = toolbox.get_priority_issues(
        check_name="category-consistency",
        column_name="city",
    )
    data = result["data"]

    assert data["total_matching_count"] == 3
    assert all(
        item["check_name"] == "category_consistency"
        and item["column"] == "city"
        for item in data["items"]
    )


def test_priority_issues_reject_unknown_severity(
    toolbox: AuditToolbox,
) -> None:
    result = toolbox.get_priority_issues(
        minimum_severity="urgent"
    )

    assert result["ok"] is False
    assert result["error"]["code"] == "unknown_severity"


def test_priority_issues_reject_unknown_check_name(
    toolbox: AuditToolbox,
) -> None:
    result = toolbox.get_priority_issues(
        check_name="schema_drift"
    )

    assert result["ok"] is False
    assert result["error"]["code"] == (
        "unknown_check_name"
    )


def test_priority_issues_return_clear_empty_result(
    clean_report: dict[str, object],
) -> None:
    result = AuditToolbox(
        clean_report
    ).get_priority_issues()

    assert result["ok"] is True
    assert result["data"]["items"] == []
    assert "No quality issues matched" in result["message"]


def test_ml_readiness_report_is_bounded_and_honest(
    toolbox: AuditToolbox,
) -> None:
    result = toolbox.get_ml_readiness_report(limit=4)
    data = result["data"]

    assert data["assessment_scope"] == (
        "GENERIC_FEATURE_QUALITY_ONLY"
    )
    assert data["returned_priority_item_count"] == 4
    assert data["priority_items_truncated"] is True
    assert data["high_or_critical_issue_count"] == 14
    assert data["capability_boundaries"] == {
        "target_column_assessed": False,
        "target_balance_assessed": False,
        "data_leakage_assessed": False,
        "train_validation_split_assessed": False,
        "model_performance_assessed": False,
        "fairness_assessed": False,
        "deployment_readiness_assessed": False,
    }
    assert "not a machine-learning readiness certification" in (
        data["interpretation"]
    )


def test_clean_ml_report_keeps_capability_warning(
    clean_report: dict[str, object],
) -> None:
    result = AuditToolbox(
        clean_report
    ).get_ml_readiness_report()

    assert result["ok"] is True
    assert result["data"]["feature_quality_issue_count"] == 0
    assert result["data"]["priority_items"] == []
    assert "evidence are still unavailable" in result["message"]


def test_tools_do_not_mutate_input_report(
    sample_report: dict[str, object],
) -> None:
    original = deepcopy(sample_report)
    toolbox = AuditToolbox(sample_report)

    toolbox.get_dataset_overview()
    toolbox.get_quality_summary()
    toolbox.get_missing_value_report(limit=3)
    toolbox.get_duplicate_report()
    toolbox.get_column_quality_report("signup_date")
    toolbox.get_priority_issues(limit=3)
    toolbox.get_ml_readiness_report(limit=3)

    assert sample_report == original


def test_toolbox_uses_independent_report_copy(
    sample_report: dict[str, object],
) -> None:
    toolbox = AuditToolbox(sample_report)

    sample_report["overview"]["row_count"] = 999  # type: ignore[index]
    sample_report["issues"].clear()  # type: ignore[union-attr]

    overview = toolbox.get_dataset_overview()
    priorities = toolbox.get_priority_issues(limit=1)

    assert overview["data"]["dimensions"]["row_count"] == 208
    assert priorities["data"]["total_matching_count"] == 27


def test_tool_signatures_do_not_expose_dataframe_or_report(
    toolbox: AuditToolbox,
) -> None:
    for function in toolbox.get_tool_functions():
        parameters = inspect.signature(function).parameters

        assert "dataframe" not in parameters
        assert "audit_report" not in parameters
        assert "report" not in parameters


def test_dispatch_executes_registered_tool(
    toolbox: AuditToolbox,
) -> None:
    result = toolbox.dispatch_tool(
        "get_priority_issues",
        {
            "limit": 2,
            "minimum_severity": "HIGH",
        },
    )

    assert result["ok"] is True
    assert result["tool_name"] == "get_priority_issues"
    assert result["data"]["returned_count"] == 2


def test_dispatch_rejects_unknown_tool(
    toolbox: AuditToolbox,
) -> None:
    result = toolbox.dispatch_tool(
        "delete_rows",
        {},
    )

    assert result["ok"] is False
    assert result["error"]["code"] == "unknown_tool"
    assert "delete_rows" not in result["error"]["details"][
        "allowed_tools"
    ]


def test_dispatch_rejects_invalid_arguments(
    toolbox: AuditToolbox,
) -> None:
    result = toolbox.dispatch_tool(
        "get_column_quality_report",
        {"unknown_argument": "age"},
    )

    assert result["ok"] is False
    assert result["error"]["code"] == (
        "tool_argument_binding_error"
    )


def test_toolbox_can_use_report_without_raw_check_results(
    sample_dataframe: pd.DataFrame,
) -> None:
    report = build_audit_report(
        sample_dataframe,
        generated_at_utc=FIXED_TIME,
        report_config=ReportBuilderConfig(
            include_raw_check_results=False,
        ),
    )

    assert "quality_check_results" not in report

    toolbox = AuditToolbox(report)

    assert toolbox.get_missing_value_report()["ok"] is True
    assert toolbox.get_duplicate_report()["ok"] is True


def test_invalid_report_is_rejected_at_construction(
    sample_report: dict[str, object],
) -> None:
    invalid_report = deepcopy(sample_report)
    invalid_report.pop("issues")

    with pytest.raises(
        AuditToolError,
        match="missing required fields",
    ):
        AuditToolbox(invalid_report)
