from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.quality_checks import (
    QualityCheckConfig,
    run_all_quality_checks,
)
from src.quality_score import (
    QualityScoreConfig,
    calculate_quality_score,
)


class ReportBuilderError(ValueError):
    """Raised when an audit report cannot be built safely."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "report_builder_error",
    ) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class ReportBuilderConfig:
    """Configuration for audit-report assembly."""

    report_version: str = "1.0"
    max_prioritized_recommendations: int = 10
    include_raw_check_results: bool = True

    def __post_init__(self) -> None:
        if not self.report_version.strip():
            raise ValueError("report_version cannot be empty.")

        if self.max_prioritized_recommendations < 0:
            raise ValueError(
                "max_prioritized_recommendations cannot be negative."
            )


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

_REQUIRED_SCORE_KEYS = {
    "overall_score",
    "score_band_status",
    "readiness_status",
    "status_adjusted_by_gates",
    "component_scores",
    "weighted_contributions",
    "weights",
    "issue_counts",
    "issue_count_total",
    "readiness_gates",
    "diagnostics",
    "methodology",
    "disclaimer",
}

_SEVERITY_RANK = {
    "CRITICAL": 0,
    "HIGH": 1,
    "MEDIUM": 2,
    "LOW": 3,
    "PASS": 4,
}

_PRIORITY_LABEL = {
    "CRITICAL": "P0",
    "HIGH": "P1",
    "MEDIUM": "P2",
    "LOW": "P3",
}

_CHECK_ORDER = {
    "missing_values": 0,
    "duplicates": 1,
    "constant_columns": 2,
    "near_constant_columns": 3,
    "potential_identifiers": 4,
    "high_cardinality_categories": 5,
    "numeric_outliers": 6,
    "category_consistency": 7,
    "data_type_warnings": 8,
}


def _json_safe(value: Any) -> Any:
    """Recursively convert pandas/numpy values into JSON-safe values."""

    if value is None:
        return None

    if isinstance(value, (str, bool, int)):
        return value

    if isinstance(value, float):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)

    if isinstance(value, np.generic):
        return _json_safe(value.item())

    if isinstance(value, (pd.Timestamp, datetime)):
        timestamp = value
        if isinstance(timestamp, pd.Timestamp):
            timestamp = timestamp.to_pydatetime()

        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)

        return timestamp.astimezone(timezone.utc).isoformat().replace(
            "+00:00",
            "Z",
        )

    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, pd.DataFrame):
        return [
            _json_safe(record)
            for record in value.to_dict(orient="records")
        ]

    if isinstance(value, pd.Series):
        return [_json_safe(item) for item in value.tolist()]

    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]

    if pd.isna(value):
        return None

    return str(value)


def _normalize_generated_at(
    generated_at_utc: datetime | str | None,
) -> str:
    """Return an ISO-8601 UTC timestamp."""

    if generated_at_utc is None:
        timestamp = datetime.now(timezone.utc)

    elif isinstance(generated_at_utc, str):
        normalized = generated_at_utc.strip().replace("Z", "+00:00")

        try:
            timestamp = datetime.fromisoformat(normalized)
        except ValueError as exc:
            raise ReportBuilderError(
                "generated_at_utc must be a valid ISO-8601 timestamp.",
                code="invalid_generated_at",
            ) from exc

    elif isinstance(generated_at_utc, datetime):
        timestamp = generated_at_utc

    else:
        raise ReportBuilderError(
            "generated_at_utc must be datetime, ISO string, or None.",
            code="invalid_generated_at_type",
        )

    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)

    return timestamp.astimezone(timezone.utc).isoformat().replace(
        "+00:00",
        "Z",
    )


def _get_metadata_value(
    metadata: Mapping[str, Any] | Any | None,
    *names: str,
    default: Any = None,
) -> Any:
    """Read metadata from a mapping or loader-result-like object."""

    if metadata is None:
        return default

    if isinstance(metadata, Mapping):
        for name in names:
            if name in metadata:
                return metadata[name]
        return default

    for name in names:
        if hasattr(metadata, name):
            return getattr(metadata, name)

    return default


def _normalize_file_metadata(
    metadata: Mapping[str, Any] | Any | None,
    *,
    overview: Mapping[str, Any],
) -> dict[str, Any]:
    """Normalize optional loader metadata and validate dimensions."""

    row_count = int(overview["row_count"])
    column_count = int(overview["column_count"])

    metadata_row_count = _get_metadata_value(
        metadata,
        "row_count",
        default=None,
    )
    metadata_column_count = _get_metadata_value(
        metadata,
        "column_count",
        default=None,
    )

    if (
        metadata_row_count is not None
        and int(metadata_row_count) != row_count
    ):
        raise ReportBuilderError(
            "File metadata row_count does not match the audit overview.",
            code="metadata_row_count_mismatch",
        )

    if (
        metadata_column_count is not None
        and int(metadata_column_count) != column_count
    ):
        raise ReportBuilderError(
            "File metadata column_count does not match the audit overview.",
            code="metadata_column_count_mismatch",
        )

    warnings = _get_metadata_value(
        metadata,
        "warnings",
        "loader_warnings",
        default=[],
    )

    if warnings is None:
        warnings = []
    elif isinstance(warnings, str):
        warnings = [warnings]
    else:
        warnings = list(warnings)

    normalized = {
        "file_name": _get_metadata_value(
            metadata,
            "file_name",
            "name",
            default="uploaded.csv",
        ),
        "file_size_bytes": _get_metadata_value(
            metadata,
            "file_size_bytes",
            default=None,
        ),
        "encoding": _get_metadata_value(
            metadata,
            "encoding",
            default=None,
        ),
        "delimiter": _get_metadata_value(
            metadata,
            "delimiter",
            default=None,
        ),
        "fingerprint_sha256": _get_metadata_value(
            metadata,
            "fingerprint_sha256",
            "fingerprint",
            default=None,
        ),
        "loader_warnings": warnings,
        "row_count": row_count,
        "column_count": column_count,
    }

    return _json_safe(normalized)


def _validate_results(
    check_results: Mapping[str, Any],
    quality_score: Mapping[str, Any],
) -> None:
    """Validate the contracts consumed by the report builder."""

    if not isinstance(check_results, Mapping):
        raise ReportBuilderError(
            "check_results must be a mapping.",
            code="invalid_check_results",
        )

    if not isinstance(quality_score, Mapping):
        raise ReportBuilderError(
            "quality_score must be a mapping.",
            code="invalid_quality_score",
        )

    missing_checks = sorted(
        _REQUIRED_CHECK_KEYS - set(check_results)
    )

    if missing_checks:
        raise ReportBuilderError(
            (
                "Missing quality-check results: "
                f"{', '.join(missing_checks)}"
            ),
            code="missing_check_results",
        )

    missing_score_fields = sorted(
        _REQUIRED_SCORE_KEYS - set(quality_score)
    )

    if missing_score_fields:
        raise ReportBuilderError(
            (
                "Missing quality-score fields: "
                f"{', '.join(missing_score_fields)}"
            ),
            code="missing_quality_score_fields",
        )

    if not isinstance(check_results["overview"], Mapping):
        raise ReportBuilderError(
            "overview must be a mapping.",
            code="invalid_overview",
        )

    if not isinstance(check_results["duplicates"], Mapping):
        raise ReportBuilderError(
            "duplicates must be a mapping.",
            code="invalid_duplicates",
        )

    for key in _REQUIRED_CHECK_KEYS - {
        "overview",
        "duplicates",
    }:
        if not isinstance(check_results[key], pd.DataFrame):
            raise ReportBuilderError(
                f"{key} must be a pandas DataFrame.",
                code="invalid_dataframe_report",
            )


def _severity(value: Any) -> str:
    """Normalize a severity label."""

    normalized = str(value).strip().upper()

    if normalized not in _SEVERITY_RANK:
        raise ReportBuilderError(
            f"Unknown severity label: {value!r}",
            code="unknown_severity",
        )

    return normalized


def _safe_number(value: Any) -> int | float | None:
    """Return a finite Python number or None."""

    converted = _json_safe(value)

    if isinstance(converted, (int, float)):
        return converted

    return None


def _issue_record(
    *,
    check_name: str,
    issue_type: str,
    severity: Any,
    status: Any,
    column: Any = None,
    count: Any = None,
    percentage: Any = None,
    evidence: str,
    recommendation: Any,
    method: str,
    limitation: str,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create one normalized issue record."""

    severity_label = _severity(severity)

    return {
        "check_name": check_name,
        "issue_type": issue_type,
        "column": (
            None
            if column is None or pd.isna(column)
            else str(column)
        ),
        "severity": severity_label,
        "priority_level": _PRIORITY_LABEL[severity_label],
        "status": str(status).strip().upper(),
        "count": _safe_number(count),
        "percentage": _safe_number(percentage),
        "evidence": evidence,
        "recommendation": str(recommendation),
        "method": method,
        "limitation": limitation,
        "details": _json_safe(details or {}),
    }


def _normalize_missing_issues(
    report: pd.DataFrame,
    *,
    row_count: int,
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []

    for row in report.to_dict(orient="records"):
        if str(row.get("status", "")).upper() == "PASS":
            continue

        count = int(row["missing_count"])
        percentage = float(row["missing_percentage"])

        issues.append(
            _issue_record(
                check_name="missing_values",
                issue_type="MISSING_VALUES",
                column=row["column"],
                severity=row["severity"],
                status=row["status"],
                count=count,
                percentage=percentage,
                evidence=(
                    f"{count} of {row_count} values are missing "
                    f"({percentage:.2f}%)."
                ),
                recommendation=row["recommendation"],
                method="Null-value scan using pandas isna().",
                limitation=(
                    "This check measures missingness but cannot determine "
                    "why values are missing or whether they are acceptable."
                ),
                details={
                    "dtype": row.get("dtype"),
                    "non_missing_count": row.get(
                        "non_missing_count"
                    ),
                },
            )
        )

    return issues


def _normalize_duplicate_issue(
    report: Mapping[str, Any],
    *,
    row_count: int,
) -> list[dict[str, Any]]:
    if str(report.get("status", "")).upper() == "PASS":
        return []

    count = int(report["duplicate_rows"])
    percentage = float(report["duplicate_percentage"])

    return [
        _issue_record(
            check_name="duplicates",
            issue_type="EXACT_DUPLICATE_ROWS",
            column=None,
            severity=report["severity"],
            status=report["status"],
            count=count,
            percentage=percentage,
            evidence=(
                f"{count} extra exact duplicate rows were found among "
                f"{row_count} rows ({percentage:.2f}%)."
            ),
            recommendation=report["recommendation"],
            method="Exact full-row comparison using pandas duplicated().",
            limitation=(
                "Near-duplicates and legitimate repeated events are not "
                "distinguished by this check."
            ),
            details={
                "rows_in_duplicate_groups": report.get(
                    "rows_in_duplicate_groups"
                ),
                "duplicate_group_count": report.get(
                    "duplicate_group_count"
                ),
                "example_duplicate_indices": report.get(
                    "example_duplicate_indices",
                    [],
                ),
            },
        )
    ]


def _normalize_constant_issues(
    report: pd.DataFrame,
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []

    for row in report.to_dict(orient="records"):
        value = row.get("constant_value")

        issues.append(
            _issue_record(
                check_name="constant_columns",
                issue_type=str(row["status"]).upper(),
                column=row["column"],
                severity=row["severity"],
                status=row["status"],
                count=row.get("unique_non_null"),
                percentage=100.0,
                evidence=(
                    "The column has no analytical variation. "
                    f"Observed non-null value: {value!r}."
                    if str(row["status"]).upper() == "CONSTANT"
                    else "The column contains no non-null values."
                ),
                recommendation=row["recommendation"],
                method="Non-null unique-value count.",
                limitation=(
                    "A constant column may still have operational or "
                    "governance value even when it adds no variation."
                ),
                details={
                    "dtype": row.get("dtype"),
                    "missing_count": row.get("missing_count"),
                    "missing_percentage": row.get(
                        "missing_percentage"
                    ),
                    "constant_value": value,
                },
            )
        )

    return issues


def _normalize_near_constant_issues(
    report: pd.DataFrame,
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []

    for row in report.to_dict(orient="records"):
        percentage = float(row["dominance_percentage"])

        issues.append(
            _issue_record(
                check_name="near_constant_columns",
                issue_type="NEAR_CONSTANT_COLUMN",
                column=row["column"],
                severity=row["severity"],
                status=row["status"],
                count=row["dominant_count"],
                percentage=percentage,
                evidence=(
                    f"Value {row['dominant_value']!r} represents "
                    f"{percentage:.2f}% of non-null records."
                ),
                recommendation=row["recommendation"],
                method="Dominant-value share among non-null values.",
                limitation=(
                    "Rare values may be valid and important despite the "
                    "column's low variation."
                ),
                details={
                    "dtype": row.get("dtype"),
                    "unique_non_null": row.get(
                        "unique_non_null"
                    ),
                    "dominant_value": row.get(
                        "dominant_value"
                    ),
                    "non_null_count": row.get(
                        "non_null_count"
                    ),
                },
            )
        )

    return issues


def _normalize_identifier_issues(
    report: pd.DataFrame,
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []

    for row in report.to_dict(orient="records"):
        percentage = float(row["unique_percentage"])

        issues.append(
            _issue_record(
                check_name="potential_identifiers",
                issue_type="POTENTIAL_IDENTIFIER",
                column=row["column"],
                severity=row["severity"],
                status=row["status"],
                count=row["unique_non_null"],
                percentage=percentage,
                evidence=(
                    f"{percentage:.2f}% of non-null values are unique; "
                    f"detection reason: {row['detection_reason']}."
                ),
                recommendation=row["recommendation"],
                method=(
                    "Heuristic based on column-name patterns and "
                    "non-null uniqueness."
                ),
                limitation=(
                    "High uniqueness does not prove that a column is an "
                    "identifier; business meaning must be verified."
                ),
                details={
                    "dtype": row.get("dtype"),
                    "non_null_count": row.get(
                        "non_null_count"
                    ),
                    "name_hint": row.get("name_hint"),
                    "detection_reason": row.get(
                        "detection_reason"
                    ),
                },
            )
        )

    return issues


def _normalize_high_cardinality_issues(
    report: pd.DataFrame,
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []

    for row in report.to_dict(orient="records"):
        percentage = float(row["unique_percentage"])

        issues.append(
            _issue_record(
                check_name="high_cardinality_categories",
                issue_type="HIGH_CARDINALITY_CATEGORY",
                column=row["column"],
                severity=row["severity"],
                status=row["status"],
                count=row["unique_non_null"],
                percentage=percentage,
                evidence=(
                    f"The column has {row['unique_non_null']} unique "
                    f"non-null values ({percentage:.2f}%)."
                ),
                recommendation=row["recommendation"],
                method=(
                    "Unique-count and unique-ratio thresholds for "
                    "categorical columns."
                ),
                limitation=(
                    "High cardinality may be appropriate for dates, "
                    "identifiers, or naturally diverse categories."
                ),
                details={
                    "dtype": row.get("dtype"),
                    "non_null_count": row.get(
                        "non_null_count"
                    ),
                    "potential_identifier_name": row.get(
                        "potential_identifier_name"
                    ),
                },
            )
        )

    return issues


def _normalize_outlier_issues(
    report: pd.DataFrame,
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []

    for row in report.to_dict(orient="records"):
        percentage = float(row["outlier_percentage"])

        issues.append(
            _issue_record(
                check_name="numeric_outliers",
                issue_type="NUMERIC_OUTLIERS",
                column=row["column"],
                severity=row["severity"],
                status=row["status"],
                count=row["outlier_count"],
                percentage=percentage,
                evidence=(
                    f"{row['outlier_count']} values fall outside the "
                    f"IQR bounds [{row['lower_bound']:.4g}, "
                    f"{row['upper_bound']:.4g}] "
                    f"({percentage:.2f}%)."
                ),
                recommendation=row["recommendation"],
                method=(
                    f"IQR rule with multiplier {row['iqr_multiplier']}."
                ),
                limitation=(
                    "IQR flags statistical extremes, not domain errors, "
                    "and may miss invalid values inside the statistical "
                    "bounds."
                ),
                details={
                    "dtype": row.get("dtype"),
                    "q1": row.get("q1"),
                    "q3": row.get("q3"),
                    "iqr": row.get("iqr"),
                    "minimum_outlier": row.get(
                        "minimum_outlier"
                    ),
                    "maximum_outlier": row.get(
                        "maximum_outlier"
                    ),
                    "example_outlier_values": row.get(
                        "example_outlier_values",
                        [],
                    ),
                    "infinite_count": row.get(
                        "infinite_count"
                    ),
                },
            )
        )

    return issues


def _normalize_category_issues(
    report: pd.DataFrame,
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []

    for row in report.to_dict(orient="records"):
        percentage = float(row["affected_percentage"])
        variants = _json_safe(row.get("variants", []))

        issues.append(
            _issue_record(
                check_name="category_consistency",
                issue_type="CATEGORY_LABEL_INCONSISTENCY",
                column=row["column"],
                severity=row["severity"],
                status=row["status"],
                count=row["affected_count"],
                percentage=percentage,
                evidence=(
                    f"{row['variant_count']} variants map to normalized "
                    f"label {row['normalized_value']!r}, affecting "
                    f"{percentage:.2f}% of non-null records."
                ),
                recommendation=row["recommendation"],
                method=(
                    "Case-folding and whitespace normalization of "
                    "categorical labels."
                ),
                limitation=(
                    "This check only detects case and whitespace "
                    "differences; semantic synonyms are not inferred."
                ),
                details={
                    "dtype": row.get("dtype"),
                    "normalized_value": row.get(
                        "normalized_value"
                    ),
                    "variants": variants,
                    "variant_frequencies": row.get(
                        "variant_frequencies",
                        {},
                    ),
                    "whitespace_variant_detected": row.get(
                        "whitespace_variant_detected"
                    ),
                    "case_variant_detected": row.get(
                        "case_variant_detected"
                    ),
                },
            )
        )

    return issues


def _data_type_impact_percentage(
    row: Mapping[str, Any],
) -> float | None:
    """Choose a useful prioritization percentage for type warnings."""

    status = str(row.get("status", "")).upper()

    if status == "UNNAMED_COLUMN":
        return 100.0

    if status == "MIXED_DATETIME_FORMATS":
        return _safe_number(
            row.get("parse_success_percentage")
        )

    invalid_percentage = _safe_number(
        row.get("invalid_percentage")
    )

    if invalid_percentage is not None:
        return float(invalid_percentage)

    return _safe_number(row.get("parse_success_percentage"))


def _normalize_data_type_issues(
    report: pd.DataFrame,
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []

    for row in report.to_dict(orient="records"):
        status = str(row["status"]).upper()
        percentage = _data_type_impact_percentage(row)

        if status == "UNNAMED_COLUMN":
            evidence = (
                "The column name matches a common exported-index "
                "artifact pattern."
            )

        elif status == "NUMERIC_LIKE_TEXT":
            evidence = (
                f"{float(row['parse_success_percentage']):.2f}% of "
                "non-null text values can be parsed as numeric; "
                f"{int(row['invalid_count'])} values cannot."
            )

        elif status == "MIXED_DATETIME_FORMATS":
            evidence = (
                f"{int(row['detected_format_count'])} date formats were "
                f"detected with "
                f"{float(row['parse_success_percentage']):.2f}% "
                "parse coverage."
            )

        else:
            evidence = (
                f"{float(row['parse_success_percentage']):.2f}% of "
                "non-null text values match a datetime pattern."
            )

        issues.append(
            _issue_record(
                check_name="data_type_warnings",
                issue_type=status,
                column=row["column"],
                severity=row["severity"],
                status=row["status"],
                count=row.get("invalid_count"),
                percentage=percentage,
                evidence=evidence,
                recommendation=row["recommendation"],
                method=(
                    "Column-name heuristics and conservative numeric/"
                    "datetime parsing patterns."
                ),
                limitation=(
                    "Type inference is heuristic and does not replace "
                    "an explicit schema or domain contract."
                ),
                details={
                    "dtype": row.get("dtype"),
                    "inferred_type": row.get(
                        "inferred_type"
                    ),
                    "non_null_count": row.get(
                        "non_null_count"
                    ),
                    "parse_success_count": row.get(
                        "parse_success_count"
                    ),
                    "parse_success_percentage": row.get(
                        "parse_success_percentage"
                    ),
                    "invalid_count": row.get(
                        "invalid_count"
                    ),
                    "invalid_percentage": row.get(
                        "invalid_percentage"
                    ),
                    "detected_formats": row.get(
                        "detected_formats",
                        [],
                    ),
                    "example_invalid_values": row.get(
                        "example_invalid_values",
                        [],
                    ),
                },
            )
        )

    return issues


def _build_issue_table(
    check_results: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Normalize every non-PASS check result into one issue table."""

    row_count = int(check_results["overview"]["row_count"])

    issues = [
        *_normalize_missing_issues(
            check_results["missing_values"],
            row_count=row_count,
        ),
        *_normalize_duplicate_issue(
            check_results["duplicates"],
            row_count=row_count,
        ),
        *_normalize_constant_issues(
            check_results["constant_columns"]
        ),
        *_normalize_near_constant_issues(
            check_results["near_constant_columns"]
        ),
        *_normalize_identifier_issues(
            check_results["potential_identifiers"]
        ),
        *_normalize_high_cardinality_issues(
            check_results[
                "high_cardinality_categories"
            ]
        ),
        *_normalize_outlier_issues(
            check_results["numeric_outliers"]
        ),
        *_normalize_category_issues(
            check_results["category_consistency"]
        ),
        *_normalize_data_type_issues(
            check_results["data_type_warnings"]
        ),
    ]

    def sort_key(issue: Mapping[str, Any]) -> tuple[Any, ...]:
        percentage = issue.get("percentage")
        count = issue.get("count")

        return (
            _SEVERITY_RANK[issue["severity"]],
            -(
                float(percentage)
                if isinstance(percentage, (int, float))
                else -1.0
            ),
            -(
                float(count)
                if isinstance(count, (int, float))
                else -1.0
            ),
            _CHECK_ORDER[issue["check_name"]],
            issue.get("column") or "",
            issue["issue_type"],
        )

    issues.sort(key=sort_key)

    for index, issue in enumerate(issues, start=1):
        issue["issue_id"] = f"ISSUE-{index:03d}"
        issue["priority_rank"] = index

    return issues


def _count_issues(
    issues: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    counts = {
        "CRITICAL": 0,
        "HIGH": 0,
        "MEDIUM": 0,
        "LOW": 0,
    }

    for issue in issues:
        counts[issue["severity"]] += 1

    return counts


def _build_prioritized_recommendations(
    issues: Sequence[Mapping[str, Any]],
    *,
    max_items: int,
) -> list[dict[str, Any]]:
    """Create a concise, de-duplicated action list."""

    if max_items == 0:
        return []

    grouped: dict[
        tuple[str | None, str],
        dict[str, Any],
    ] = {}

    for issue in issues:
        key = (
            issue.get("column"),
            issue["recommendation"],
        )

        if key not in grouped:
            grouped[key] = {
                "severity": issue["severity"],
                "column": issue.get("column"),
                "issue_type": issue["issue_type"],
                "action": issue["recommendation"],
                "rationale": issue["evidence"],
                "related_issue_ids": [issue["issue_id"]],
                "source_checks": [issue["check_name"]],
                "_first_rank": issue["priority_rank"],
            }
            continue

        item = grouped[key]
        item["related_issue_ids"].append(issue["issue_id"])

        if issue["check_name"] not in item["source_checks"]:
            item["source_checks"].append(issue["check_name"])

        if (
            _SEVERITY_RANK[issue["severity"]]
            < _SEVERITY_RANK[item["severity"]]
        ):
            item["severity"] = issue["severity"]

    recommendations = sorted(
        grouped.values(),
        key=lambda item: (
            _SEVERITY_RANK[item["severity"]],
            item["_first_rank"],
            item.get("column") or "",
        ),
    )[:max_items]

    for index, item in enumerate(recommendations, start=1):
        item["recommendation_id"] = f"REC-{index:03d}"
        item["priority_rank"] = index
        item["priority_level"] = _PRIORITY_LABEL[
            item["severity"]
        ]
        item.pop("_first_rank", None)

    return recommendations


def _headline(readiness_status: str) -> str:
    messages = {
        "READY_WITH_MINOR_REVIEW": (
            "The dataset is broadly ready, with minor review still "
            "recommended."
        ),
        "NEEDS_CLEANING": (
            "The dataset requires targeted cleaning before downstream "
            "analysis or machine learning."
        ),
        "SIGNIFICANT_QUALITY_ISSUES": (
            "The dataset has significant quality issues that should be "
            "resolved before use."
        ),
        "NOT_READY": (
            "The dataset is not ready for downstream use without "
            "substantial remediation."
        ),
    }

    return messages.get(
        readiness_status,
        "The dataset requires human review.",
    )


def _report_id(
    *,
    generated_at_utc: str,
    metadata: Mapping[str, Any],
    score: float,
) -> str:
    source = "|".join(
        [
            str(metadata.get("fingerprint_sha256") or ""),
            str(metadata.get("file_name") or ""),
            generated_at_utc,
            f"{score:.2f}",
        ]
    )

    return f"DSA-{sha256(source.encode('utf-8')).hexdigest()[:12]}"


def build_audit_report_from_results(
    check_results: Mapping[str, Any],
    quality_score: Mapping[str, Any],
    *,
    file_metadata: Mapping[str, Any] | Any | None = None,
    generated_at_utc: datetime | str | None = None,
    config: ReportBuilderConfig | None = None,
) -> dict[str, Any]:
    """
    Build one JSON-safe audit report from precomputed results.

    The function does not mutate the supplied quality-check or score
    results.
    """

    _validate_results(check_results, quality_score)
    active_config = config or ReportBuilderConfig()

    generated_at = _normalize_generated_at(
        generated_at_utc
    )
    overview = deepcopy(check_results["overview"])

    metadata = _normalize_file_metadata(
        file_metadata,
        overview=overview,
    )

    issues = _build_issue_table(check_results)
    issue_counts = _count_issues(issues)

    expected_counts = {
        key: int(value)
        for key, value in quality_score["issue_counts"].items()
    }

    if issue_counts != expected_counts:
        raise ReportBuilderError(
            (
                "Normalized issue counts do not match quality-score "
                "issue counts."
            ),
            code="issue_count_mismatch",
        )

    if len(issues) != int(quality_score["issue_count_total"]):
        raise ReportBuilderError(
            (
                "Normalized issue total does not match the quality-score "
                "issue total."
            ),
            code="issue_total_mismatch",
        )

    recommendations = _build_prioritized_recommendations(
        issues,
        max_items=(
            active_config.max_prioritized_recommendations
        ),
    )

    readiness_status = str(
        quality_score["readiness_status"]
    )

    summary = {
        "headline": _headline(readiness_status),
        "overall_score": float(
            quality_score["overall_score"]
        ),
        "score_band_status": quality_score[
            "score_band_status"
        ],
        "readiness_status": readiness_status,
        "status_adjusted_by_gates": bool(
            quality_score["status_adjusted_by_gates"]
        ),
        "issue_count_total": len(issues),
        "issue_counts": issue_counts,
        "prioritized_recommendation_count": len(
            recommendations
        ),
        "highest_severity": (
            issues[0]["severity"]
            if issues
            else "PASS"
        ),
        "readiness_gates": deepcopy(
            quality_score["readiness_gates"]
        ),
    }

    report: dict[str, Any] = {
        "report_version": active_config.report_version,
        "report_id": _report_id(
            generated_at_utc=generated_at,
            metadata=metadata,
            score=float(quality_score["overall_score"]),
        ),
        "generated_at_utc": generated_at,
        "file_metadata": metadata,
        "summary": summary,
        "overview": deepcopy(overview),
        "component_scores": deepcopy(
            quality_score["component_scores"]
        ),
        "weighted_contributions": deepcopy(
            quality_score["weighted_contributions"]
        ),
        "score_weights": deepcopy(
            quality_score["weights"]
        ),
        "issues": issues,
        "prioritized_recommendations": recommendations,
        "scoring_diagnostics": deepcopy(
            quality_score["diagnostics"]
        ),
        "methodology": {
            "scoring": deepcopy(
                quality_score["methodology"]
            ),
            "issue_prioritization": (
                "Issues are ordered by severity, affected percentage, "
                "affected count, and deterministic check order."
            ),
            "recommendation_deduplication": (
                "Recommendations with the same action for the same "
                "column are grouped."
            ),
            "report_type": "DETERMINISTIC_HEURISTIC_AUDIT",
        },
        "limitations": [
            (
                "The audit uses generic heuristics and does not know "
                "source-system or business-domain rules."
            ),
            (
                "Statistical outliers are not automatically errors, and "
                "invalid domain values may remain inside statistical "
                "bounds."
            ),
            (
                "Category-consistency checks cover case and whitespace "
                "differences, not semantic synonyms."
            ),
            (
                "Readiness is a prioritization aid and must be confirmed "
                "through human and domain review."
            ),
        ],
        "disclaimer": quality_score["disclaimer"],
    }

    if active_config.include_raw_check_results:
        report["quality_check_results"] = {
            key: _json_safe(value)
            for key, value in check_results.items()
        }

    return _json_safe(report)


def build_audit_report(
    dataframe: pd.DataFrame,
    *,
    file_metadata: Mapping[str, Any] | Any | None = None,
    generated_at_utc: datetime | str | None = None,
    check_config: QualityCheckConfig | None = None,
    score_config: QualityScoreConfig | None = None,
    report_config: ReportBuilderConfig | None = None,
) -> dict[str, Any]:
    """
    Run checks, calculate the score, and build a structured audit report.
    """

    checks = run_all_quality_checks(
        dataframe,
        config=check_config,
    )
    score = calculate_quality_score(
        checks,
        config=score_config,
    )

    return build_audit_report_from_results(
        checks,
        score,
        file_metadata=file_metadata,
        generated_at_utc=generated_at_utc,
        config=report_config,
    )


def audit_report_to_json(
    report: Mapping[str, Any],
    *,
    indent: int = 2,
) -> str:
    """Serialize an audit report for download or API transport."""

    if not isinstance(report, Mapping):
        raise ReportBuilderError(
            "report must be a mapping.",
            code="invalid_report",
        )

    return json.dumps(
        _json_safe(report),
        indent=indent,
        ensure_ascii=False,
        sort_keys=False,
        allow_nan=False,
    )
