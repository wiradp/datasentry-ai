from __future__ import annotations

from dataclasses import dataclass
from math import isclose
from typing import Any, Mapping

import pandas as pd

from src.quality_checks import (
    QualityCheckConfig,
    run_all_quality_checks,
)


class QualityScoreError(ValueError):
    """Raised when quality-check results cannot be scored safely."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "quality_score_error",
    ) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class QualityScoreConfig:
    """Weights, thresholds, and penalties for heuristic scoring."""

    completeness_weight: float = 0.35
    duplicate_free_weight: float = 0.20
    schema_usability_weight: float = 0.15
    categorical_consistency_weight: float = 0.15
    outlier_risk_weight: float = 0.15

    low_penalty: float = 10.0
    medium_penalty: float = 25.0
    high_penalty: float = 50.0
    critical_penalty: float = 100.0

    ready_min_score: float = 85.0
    needs_cleaning_min_score: float = 70.0
    significant_issues_min_score: float = 50.0

    apply_readiness_gates: bool = True

    def __post_init__(self) -> None:
        weights = self.weights

        if any(weight < 0 for weight in weights.values()):
            raise ValueError("Quality-score weights cannot be negative.")

        if not isclose(
            sum(weights.values()),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError(
                "Quality-score weights must sum to 1.0."
            )

        penalties = self.severity_penalties

        if not (
            0.0
            <= penalties["PASS"]
            <= penalties["LOW"]
            <= penalties["MEDIUM"]
            <= penalties["HIGH"]
            <= penalties["CRITICAL"]
            <= 100.0
        ):
            raise ValueError(
                "Severity penalties must be ordered between 0 and 100."
            )

        if not (
            0.0
            <= self.significant_issues_min_score
            <= self.needs_cleaning_min_score
            <= self.ready_min_score
            <= 100.0
        ):
            raise ValueError(
                "Readiness thresholds must be ordered between 0 and 100."
            )

    @property
    def weights(self) -> dict[str, float]:
        return {
            "completeness": self.completeness_weight,
            "duplicate_free": self.duplicate_free_weight,
            "schema_usability": self.schema_usability_weight,
            "categorical_consistency": (
                self.categorical_consistency_weight
            ),
            "outlier_risk": self.outlier_risk_weight,
        }

    @property
    def severity_penalties(self) -> dict[str, float]:
        return {
            "PASS": 0.0,
            "LOW": self.low_penalty,
            "MEDIUM": self.medium_penalty,
            "HIGH": self.high_penalty,
            "CRITICAL": self.critical_penalty,
        }


_REQUIRED_CHECK_KEYS = {
    "overview",
    "missing_values",
    "duplicates",
    "constant_columns",
    "near_constant_columns",
    "potential_identifiers",
    "high_cardinality_categories",
    "numeric_outliers",
    "category_consistency",
    "data_type_warnings",
}

_SCHEMA_REPORT_KEYS = (
    "constant_columns",
    "near_constant_columns",
    "potential_identifiers",
    "high_cardinality_categories",
    "data_type_warnings",
)

_DATAFRAME_REPORT_KEYS = (
    "missing_values",
    "constant_columns",
    "near_constant_columns",
    "potential_identifiers",
    "high_cardinality_categories",
    "numeric_outliers",
    "category_consistency",
    "data_type_warnings",
)

_SEVERITY_ORDER = (
    "CRITICAL",
    "HIGH",
    "MEDIUM",
    "LOW",
    "PASS",
)



def _clamp_score(value: float) -> float:
    """Clamp a score to the inclusive 0–100 range."""

    return round(max(0.0, min(100.0, float(value))), 2)



def _validate_check_results(
    check_results: Mapping[str, Any],
) -> None:
    """Validate the contract produced by run_all_quality_checks."""

    if not isinstance(check_results, Mapping):
        raise QualityScoreError(
            "check_results must be a mapping.",
            code="invalid_check_results_type",
        )

    missing_keys = sorted(
        _REQUIRED_CHECK_KEYS - set(check_results)
    )

    if missing_keys:
        raise QualityScoreError(
            (
                "Missing quality-check results: "
                f"{', '.join(missing_keys)}"
            ),
            code="missing_check_results",
        )

    overview = check_results["overview"]
    duplicates = check_results["duplicates"]

    if not isinstance(overview, Mapping):
        raise QualityScoreError(
            "overview must be a mapping.",
            code="invalid_overview",
        )

    if not isinstance(duplicates, Mapping):
        raise QualityScoreError(
            "duplicates must be a mapping.",
            code="invalid_duplicate_report",
        )

    required_overview_fields = {
        "row_count",
        "column_count",
        "missing_percentage",
        "numeric_column_count",
        "categorical_column_count",
    }

    missing_overview_fields = sorted(
        required_overview_fields - set(overview)
    )

    if missing_overview_fields:
        raise QualityScoreError(
            (
                "Overview is missing fields: "
                f"{', '.join(missing_overview_fields)}"
            ),
            code="invalid_overview",
        )

    if "duplicate_percentage" not in duplicates:
        raise QualityScoreError(
            "Duplicate report is missing duplicate_percentage.",
            code="invalid_duplicate_report",
        )

    if int(overview["row_count"]) <= 0:
        raise QualityScoreError(
            "Overview row_count must be greater than zero.",
            code="invalid_row_count",
        )

    if int(overview["column_count"]) <= 0:
        raise QualityScoreError(
            "Overview column_count must be greater than zero.",
            code="invalid_column_count",
        )

    for key in _DATAFRAME_REPORT_KEYS:
        if not isinstance(check_results[key], pd.DataFrame):
            raise QualityScoreError(
                f"{key} must be a pandas DataFrame.",
                code="invalid_dataframe_report",
            )



def _require_columns(
    report: pd.DataFrame,
    required_columns: set[str],
    *,
    report_name: str,
) -> None:
    """Require report columns while allowing empty DataFrames."""

    missing_columns = sorted(
        required_columns - set(report.columns)
    )

    if missing_columns:
        raise QualityScoreError(
            (
                f"{report_name} is missing columns: "
                f"{', '.join(missing_columns)}"
            ),
            code="invalid_report_schema",
        )



def _normalized_severity(value: Any) -> str:
    """Normalize and validate a severity label."""

    severity = str(value).strip().upper()

    if severity not in _SEVERITY_ORDER:
        raise QualityScoreError(
            f"Unknown severity label: {value!r}",
            code="unknown_severity",
        )

    return severity



def _completeness_score(
    overview: Mapping[str, Any],
) -> float:
    """Score the share of non-missing cells."""

    missing_percentage = float(
        overview["missing_percentage"]
    )

    return _clamp_score(100.0 - missing_percentage)



def _duplicate_free_score(
    duplicate_report: Mapping[str, Any],
) -> float:
    """Score the share of rows that are not extra duplicates."""

    duplicate_percentage = float(
        duplicate_report["duplicate_percentage"]
    )

    return _clamp_score(100.0 - duplicate_percentage)



def _schema_usability_score(
    check_results: Mapping[str, Any],
    *,
    column_count: int,
    config: QualityScoreConfig,
) -> tuple[float, dict[str, float]]:
    """
    Score schema usability using the maximum issue penalty per column.

    Using the maximum prevents overlapping checks from repeatedly
    penalizing the same column.
    """

    penalties_by_column: dict[str, float] = {}
    severity_penalties = config.severity_penalties

    for report_name in _SCHEMA_REPORT_KEYS:
        report = check_results[report_name]

        _require_columns(
            report,
            {"column", "severity"},
            report_name=report_name,
        )

        for row in report[["column", "severity"]].itertuples(
            index=False
        ):
            column = str(row.column)
            severity = _normalized_severity(row.severity)
            penalty = severity_penalties[severity]

            penalties_by_column[column] = max(
                penalties_by_column.get(column, 0.0),
                penalty,
            )

    total_penalty = sum(penalties_by_column.values())
    average_penalty = total_penalty / column_count

    return (
        _clamp_score(100.0 - average_penalty),
        {
            column: round(penalty, 2)
            for column, penalty in sorted(
                penalties_by_column.items()
            )
        },
    )



def _categorical_consistency_score(
    category_report: pd.DataFrame,
    *,
    categorical_column_count: int,
) -> tuple[float, dict[str, float]]:
    """Score category-label consistency across categorical columns."""

    _require_columns(
        category_report,
        {"column", "affected_percentage"},
        report_name="category_consistency",
    )

    if categorical_column_count <= 0:
        return 100.0, {}

    affected_by_column: dict[str, float] = {}

    for row in category_report[
        ["column", "affected_percentage"]
    ].itertuples(index=False):
        column = str(row.column)
        affected_by_column[column] = min(
            100.0,
            affected_by_column.get(column, 0.0)
            + float(row.affected_percentage),
        )

    average_affected_percentage = (
        sum(affected_by_column.values())
        / categorical_column_count
    )

    return (
        _clamp_score(100.0 - average_affected_percentage),
        {
            column: round(percentage, 2)
            for column, percentage in sorted(
                affected_by_column.items()
            )
        },
    )



def _outlier_risk_score(
    outlier_report: pd.DataFrame,
    *,
    numeric_column_count: int,
) -> tuple[float, dict[str, float]]:
    """Score the observed IQR-outlier burden across numeric columns."""

    _require_columns(
        outlier_report,
        {"column", "outlier_percentage"},
        report_name="numeric_outliers",
    )

    if numeric_column_count <= 0:
        return 100.0, {}

    outlier_by_column: dict[str, float] = {}

    for row in outlier_report[
        ["column", "outlier_percentage"]
    ].itertuples(index=False):
        column = str(row.column)
        outlier_by_column[column] = min(
            100.0,
            max(
                outlier_by_column.get(column, 0.0),
                float(row.outlier_percentage),
            ),
        )

    average_outlier_percentage = (
        sum(outlier_by_column.values())
        / numeric_column_count
    )

    return (
        _clamp_score(100.0 - average_outlier_percentage),
        {
            column: round(percentage, 2)
            for column, percentage in sorted(
                outlier_by_column.items()
            )
        },
    )



def _count_issues_by_severity(
    check_results: Mapping[str, Any],
) -> dict[str, int]:
    """Count issue records by severity across all check reports."""

    counts = {
        severity: 0
        for severity in _SEVERITY_ORDER
        if severity != "PASS"
    }

    missing_report = check_results["missing_values"]

    _require_columns(
        missing_report,
        {"severity", "status"},
        report_name="missing_values",
    )

    issue_rows = missing_report.loc[
        missing_report["status"].astype(str).str.upper()
        != "PASS"
    ]

    for value in issue_rows["severity"].tolist():
        severity = _normalized_severity(value)
        if severity != "PASS":
            counts[severity] += 1

    duplicate_report = check_results["duplicates"]

    if str(duplicate_report.get("status", "")).upper() != "PASS":
        severity = _normalized_severity(
            duplicate_report.get("severity", "PASS")
        )
        if severity != "PASS":
            counts[severity] += 1

    for report_name in (
        "constant_columns",
        "near_constant_columns",
        "potential_identifiers",
        "high_cardinality_categories",
        "numeric_outliers",
        "category_consistency",
        "data_type_warnings",
    ):
        report = check_results[report_name]

        _require_columns(
            report,
            {"severity"},
            report_name=report_name,
        )

        for value in report["severity"].tolist():
            severity = _normalized_severity(value)
            if severity != "PASS":
                counts[severity] += 1

    return counts



def _score_band_status(
    score: float,
    config: QualityScoreConfig,
) -> str:
    """Map the overall numeric score to its base score band."""

    if score >= config.ready_min_score:
        return "READY_WITH_MINOR_REVIEW"

    if score >= config.needs_cleaning_min_score:
        return "NEEDS_CLEANING"

    if score >= config.significant_issues_min_score:
        return "SIGNIFICANT_QUALITY_ISSUES"

    return "NOT_READY"



def _apply_readiness_gates(
    base_status: str,
    issue_counts: Mapping[str, int],
    *,
    enabled: bool,
) -> tuple[str, list[str]]:
    """Apply conservative gates that prevent an inflated ready status."""

    if not enabled:
        return base_status, []

    gates: list[str] = []
    status = base_status

    if issue_counts.get("CRITICAL", 0) > 0:
        status = "NOT_READY"
        gates.append(
            "Critical issues force the readiness status to NOT_READY."
        )

    elif (
        issue_counts.get("HIGH", 0) > 0
        and base_status == "READY_WITH_MINOR_REVIEW"
    ):
        status = "NEEDS_CLEANING"
        gates.append(
            "High-severity issues prevent a READY_WITH_MINOR_REVIEW status."
        )

    return status, gates



def calculate_quality_score(
    check_results: Mapping[str, Any],
    *,
    config: QualityScoreConfig | None = None,
) -> dict[str, Any]:
    """
    Calculate a deterministic heuristic data-quality score.

    The function consumes the output of run_all_quality_checks and does
    not mutate the reports or the source DataFrame.
    """

    _validate_check_results(check_results)

    active_config = config or QualityScoreConfig()
    overview = check_results["overview"]

    column_count = int(overview["column_count"])
    categorical_column_count = int(
        overview["categorical_column_count"]
    )
    numeric_column_count = int(
        overview["numeric_column_count"]
    )

    completeness = _completeness_score(overview)
    duplicate_free = _duplicate_free_score(
        check_results["duplicates"]
    )

    schema_usability, schema_penalties = (
        _schema_usability_score(
            check_results,
            column_count=column_count,
            config=active_config,
        )
    )

    categorical_consistency, category_burden = (
        _categorical_consistency_score(
            check_results["category_consistency"],
            categorical_column_count=(
                categorical_column_count
            ),
        )
    )

    outlier_risk, outlier_burden = (
        _outlier_risk_score(
            check_results["numeric_outliers"],
            numeric_column_count=numeric_column_count,
        )
    )

    component_scores = {
        "completeness": completeness,
        "duplicate_free": duplicate_free,
        "schema_usability": schema_usability,
        "categorical_consistency": (
            categorical_consistency
        ),
        "outlier_risk": outlier_risk,
    }

    weights = active_config.weights

    weighted_contributions = {
        component: round(
            component_scores[component] * weight,
            4,
        )
        for component, weight in weights.items()
    }

    overall_score = _clamp_score(
        sum(weighted_contributions.values())
    )

    issue_counts = _count_issues_by_severity(
        check_results
    )

    score_band_status = _score_band_status(
        overall_score,
        active_config,
    )

    readiness_status, readiness_gates = (
        _apply_readiness_gates(
            score_band_status,
            issue_counts,
            enabled=active_config.apply_readiness_gates,
        )
    )

    return {
        "overall_score": overall_score,
        "score_band_status": score_band_status,
        "readiness_status": readiness_status,
        "status_adjusted_by_gates": (
            readiness_status != score_band_status
        ),
        "component_scores": component_scores,
        "weighted_contributions": weighted_contributions,
        "weights": {
            component: round(weight, 4)
            for component, weight in weights.items()
        },
        "issue_counts": issue_counts,
        "issue_count_total": int(
            sum(issue_counts.values())
        ),
        "readiness_gates": readiness_gates,
        "diagnostics": {
            "schema_penalty_by_column": (
                schema_penalties
            ),
            "category_affected_percentage_by_column": (
                category_burden
            ),
            "outlier_percentage_by_column": (
                outlier_burden
            ),
        },
        "methodology": {
            "type": "HEURISTIC",
            "completeness": (
                "100 minus the percentage of missing cells."
            ),
            "duplicate_free": (
                "100 minus the percentage of extra exact duplicate rows."
            ),
            "schema_usability": (
                "100 minus the average maximum severity penalty per column."
            ),
            "categorical_consistency": (
                "100 minus the average affected percentage across all "
                "categorical columns."
            ),
            "outlier_risk": (
                "100 minus the average IQR-outlier percentage across all "
                "numeric columns."
            ),
        },
        "disclaimer": (
            "This score is a heuristic prioritization aid and does not "
            "replace domain validation, source-system checks, or human review."
        ),
    }



def calculate_dataframe_quality_score(
    dataframe: pd.DataFrame,
    *,
    check_config: QualityCheckConfig | None = None,
    score_config: QualityScoreConfig | None = None,
) -> dict[str, Any]:
    """Run all deterministic checks and calculate the quality score."""

    check_results = run_all_quality_checks(
        dataframe,
        config=check_config,
    )

    return calculate_quality_score(
        check_results,
        config=score_config,
    )
