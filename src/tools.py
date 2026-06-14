from __future__ import annotations

import inspect
import json
import math
from copy import deepcopy
from typing import Any, Callable, Mapping, Sequence


DEFAULT_RESULT_LIMIT = 10
MAX_RESULT_LIMIT = 25

AUDIT_TOOL_NAMES = (
    "get_dataset_overview",
    "get_quality_summary",
    "get_missing_value_report",
    "get_duplicate_report",
    "get_column_quality_report",
    "get_priority_issues",
    "get_ml_readiness_report",
)

VALID_SEVERITIES = (
    "CRITICAL",
    "HIGH",
    "MEDIUM",
    "LOW",
)

VALID_CHECK_NAMES = (
    "missing_values",
    "duplicates",
    "constant_columns",
    "near_constant_columns",
    "potential_identifiers",
    "high_cardinality_categories",
    "numeric_outliers",
    "category_consistency",
    "data_type_warnings",
)

_REQUIRED_REPORT_KEYS = {
    "report_id",
    "generated_at_utc",
    "file_metadata",
    "summary",
    "overview",
    "component_scores",
    "issues",
    "prioritized_recommendations",
    "scoring_diagnostics",
    "limitations",
    "disclaimer",
}

_REQUIRED_SUMMARY_KEYS = {
    "overall_score",
    "score_band_status",
    "readiness_status",
    "status_adjusted_by_gates",
    "issue_count_total",
    "issue_counts",
    "highest_severity",
    "readiness_gates",
}

_REQUIRED_OVERVIEW_KEYS = {
    "row_count",
    "column_count",
    "total_cells",
    "missing_cells",
    "missing_percentage",
    "duplicate_rows",
    "duplicate_percentage",
    "numeric_columns",
    "categorical_columns",
    "datetime_columns",
    "boolean_columns",
    "other_columns",
}

_REQUIRED_ISSUE_KEYS = {
    "issue_id",
    "priority_rank",
    "check_name",
    "issue_type",
    "column",
    "severity",
    "status",
    "evidence",
    "recommendation",
    "limitation",
}

_SEVERITY_RANK = {
    "CRITICAL": 0,
    "HIGH": 1,
    "MEDIUM": 2,
    "LOW": 3,
}

_COLUMN_TYPE_FIELDS = (
    ("numeric", "numeric_columns"),
    ("categorical", "categorical_columns"),
    ("datetime", "datetime_columns"),
    ("boolean", "boolean_columns"),
    ("other", "other_columns"),
)

_ML_CATEGORY_BY_CHECK = {
    "missing_values": "MISSINGNESS",
    "duplicates": "ROW_INTEGRITY",
    "constant_columns": "FEATURE_VARIANCE",
    "near_constant_columns": "FEATURE_VARIANCE",
    "potential_identifiers": "IDENTIFIER_RISK",
    "high_cardinality_categories": "ENCODING_COMPLEXITY",
    "numeric_outliers": "DISTRIBUTION_RISK",
    "category_consistency": "CATEGORY_ENCODING",
    "data_type_warnings": "TYPE_CONVERSION",
}


class AuditToolError(ValueError):
    """Raised internally when a read-only audit tool rejects a request."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "audit_tool_error",
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.details = dict(details or {})


def _json_safe(value: Any) -> Any:
    """Recursively convert a value into strict JSON-compatible data."""

    if value is None or isinstance(value, (str, bool, int)):
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

    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except (TypeError, ValueError):
            pass

    return str(value)


def _strict_json_copy(value: Any) -> Any:
    """Create an independent JSON-safe copy and reject invalid payloads."""

    safe_value = _json_safe(value)

    try:
        payload = json.dumps(
            safe_value,
            allow_nan=False,
            ensure_ascii=False,
        )
    except (TypeError, ValueError) as exc:
        raise AuditToolError(
            "The audit report contains values that are not JSON-safe.",
            code="report_not_json_safe",
        ) from exc

    return json.loads(payload)


def _success_response(
    tool_name: str,
    *,
    message: str,
    data: Mapping[str, Any],
) -> dict[str, Any]:
    return _strict_json_copy(
        {
            "ok": True,
            "tool_name": tool_name,
            "message": message,
            "data": data,
        }
    )


def _error_response(
    tool_name: str,
    error: AuditToolError,
) -> dict[str, Any]:
    return _strict_json_copy(
        {
            "ok": False,
            "tool_name": tool_name,
            "message": str(error),
            "error": {
                "code": error.code,
                "details": error.details,
            },
            "data": None,
        }
    )


def _validate_limit(limit: int) -> int:
    if isinstance(limit, bool) or not isinstance(limit, int):
        raise AuditToolError(
            "limit must be an integer.",
            code="invalid_limit_type",
        )

    if not 1 <= limit <= MAX_RESULT_LIMIT:
        raise AuditToolError(
            (
                f"limit must be between 1 and "
                f"{MAX_RESULT_LIMIT}."
            ),
            code="limit_out_of_range",
            details={
                "minimum": 1,
                "maximum": MAX_RESULT_LIMIT,
            },
        )

    return limit


def _validate_percentage(
    value: float,
    *,
    parameter_name: str,
) -> float:
    if isinstance(value, bool) or not isinstance(
        value,
        (int, float),
    ):
        raise AuditToolError(
            f"{parameter_name} must be a number.",
            code="invalid_percentage_type",
        )

    numeric_value = float(value)

    if not math.isfinite(numeric_value):
        raise AuditToolError(
            f"{parameter_name} must be finite.",
            code="invalid_percentage_value",
        )

    if not 0.0 <= numeric_value <= 100.0:
        raise AuditToolError(
            f"{parameter_name} must be between 0 and 100.",
            code="percentage_out_of_range",
        )

    return numeric_value


def _normalize_severity(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AuditToolError(
            "minimum_severity must be a non-empty string.",
            code="invalid_severity_type",
        )

    severity = value.strip().upper()

    if severity not in _SEVERITY_RANK:
        raise AuditToolError(
            (
                f"Unknown severity {value!r}. Allowed values: "
                f"{', '.join(VALID_SEVERITIES)}."
            ),
            code="unknown_severity",
            details={
                "allowed_values": list(VALID_SEVERITIES),
            },
        )

    return severity


def _normalize_check_name(
    value: str | None,
) -> str | None:
    if value is None:
        return None

    if not isinstance(value, str) or not value.strip():
        raise AuditToolError(
            "check_name must be a non-empty string or null.",
            code="invalid_check_name_type",
        )

    normalized = (
        value.strip()
        .casefold()
        .replace("-", "_")
        .replace(" ", "_")
    )

    if normalized not in VALID_CHECK_NAMES:
        raise AuditToolError(
            (
                f"Unknown check_name {value!r}. Allowed values: "
                f"{', '.join(VALID_CHECK_NAMES)}."
            ),
            code="unknown_check_name",
            details={
                "allowed_values": list(VALID_CHECK_NAMES),
            },
        )

    return normalized


def _validate_report(
    audit_report: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(audit_report, Mapping):
        raise AuditToolError(
            "audit_report must be a mapping.",
            code="invalid_audit_report",
        )

    missing_keys = sorted(
        _REQUIRED_REPORT_KEYS - set(audit_report)
    )

    if missing_keys:
        raise AuditToolError(
            (
                "The structured audit report is missing required "
                f"fields: {', '.join(missing_keys)}."
            ),
            code="incomplete_audit_report",
            details={"missing_fields": missing_keys},
        )

    report = _strict_json_copy(audit_report)

    if not isinstance(report["summary"], dict):
        raise AuditToolError(
            "audit_report.summary must be an object.",
            code="invalid_summary",
        )

    missing_summary = sorted(
        _REQUIRED_SUMMARY_KEYS - set(report["summary"])
    )

    if missing_summary:
        raise AuditToolError(
            (
                "The audit summary is missing required fields: "
                f"{', '.join(missing_summary)}."
            ),
            code="incomplete_summary",
            details={"missing_fields": missing_summary},
        )

    if not isinstance(report["overview"], dict):
        raise AuditToolError(
            "audit_report.overview must be an object.",
            code="invalid_overview",
        )

    missing_overview = sorted(
        _REQUIRED_OVERVIEW_KEYS - set(report["overview"])
    )

    if missing_overview:
        raise AuditToolError(
            (
                "The dataset overview is missing required fields: "
                f"{', '.join(missing_overview)}."
            ),
            code="incomplete_overview",
            details={"missing_fields": missing_overview},
        )

    if not isinstance(report["issues"], list):
        raise AuditToolError(
            "audit_report.issues must be a list.",
            code="invalid_issue_table",
        )

    for index, issue in enumerate(report["issues"]):
        if not isinstance(issue, dict):
            raise AuditToolError(
                f"Issue at index {index} must be an object.",
                code="invalid_issue_record",
            )

        missing_issue_fields = sorted(
            _REQUIRED_ISSUE_KEYS - set(issue)
        )

        if missing_issue_fields:
            raise AuditToolError(
                (
                    f"Issue at index {index} is missing fields: "
                    f"{', '.join(missing_issue_fields)}."
                ),
                code="incomplete_issue_record",
                details={
                    "issue_index": index,
                    "missing_fields": missing_issue_fields,
                },
            )

        severity = str(issue["severity"]).upper()
        check_name = str(issue["check_name"])

        if severity not in _SEVERITY_RANK:
            raise AuditToolError(
                (
                    f"Issue {issue['issue_id']} has an unknown "
                    f"severity: {issue['severity']!r}."
                ),
                code="invalid_issue_severity",
            )

        if check_name not in VALID_CHECK_NAMES:
            raise AuditToolError(
                (
                    f"Issue {issue['issue_id']} has an unknown "
                    f"check name: {check_name!r}."
                ),
                code="invalid_issue_check_name",
            )

    if not isinstance(
        report["prioritized_recommendations"],
        list,
    ):
        raise AuditToolError(
            (
                "audit_report.prioritized_recommendations "
                "must be a list."
            ),
            code="invalid_recommendation_table",
        )

    return report


class AuditToolbox:
    """
    Read-only access layer over one structured DataSentry audit report.

    The toolbox stores an independent JSON copy of the report. It never
    receives or exposes the source pandas DataFrame.
    """

    def __init__(
        self,
        audit_report: Mapping[str, Any],
    ) -> None:
        self._report = _validate_report(audit_report)
        self._columns = self._collect_columns()
        self._column_lookup = self._build_column_lookup()

        self._validate_issue_columns()

    def __repr__(self) -> str:
        return (
            "AuditToolbox("
            f"report_id={self._report['report_id']!r}, "
            f"columns={len(self._columns)}, "
            f"issues={len(self._report['issues'])}"
            ")"
        )

    def _collect_columns(self) -> tuple[str, ...]:
        overview = self._report["overview"]
        columns: list[str] = []

        for _, field_name in _COLUMN_TYPE_FIELDS:
            raw_columns = overview[field_name]

            if not isinstance(raw_columns, list):
                raise AuditToolError(
                    f"overview.{field_name} must be a list.",
                    code="invalid_column_list",
                )

            for column in raw_columns:
                column_name = str(column)

                if column_name not in columns:
                    columns.append(column_name)

        if len(columns) != int(overview["column_count"]):
            raise AuditToolError(
                (
                    "Column lists do not match overview.column_count."
                ),
                code="column_count_mismatch",
                details={
                    "declared_column_count": (
                        overview["column_count"]
                    ),
                    "listed_column_count": len(columns),
                },
            )

        return tuple(columns)

    def _build_column_lookup(
        self,
    ) -> dict[str, list[str]]:
        lookup: dict[str, list[str]] = {}

        for column in self._columns:
            key = column.casefold()
            lookup.setdefault(key, []).append(column)

        return lookup

    def _validate_issue_columns(self) -> None:
        known_columns = set(self._columns)

        for issue in self._report["issues"]:
            column = issue.get("column")

            if column is not None and column not in known_columns:
                raise AuditToolError(
                    (
                        f"Issue {issue['issue_id']} references unknown "
                        f"column {column!r}."
                    ),
                    code="issue_unknown_column",
                )

    def _column_type(self, column_name: str) -> str:
        overview = self._report["overview"]

        for column_type, field_name in _COLUMN_TYPE_FIELDS:
            if column_name in overview[field_name]:
                return column_type

        raise AuditToolError(
            f"Column {column_name!r} has no registered type.",
            code="column_type_unavailable",
        )

    def _resolve_column_name(
        self,
        column_name: str,
    ) -> str:
        if not isinstance(column_name, str):
            raise AuditToolError(
                "column_name must be a string.",
                code="invalid_column_name_type",
            )

        stripped = column_name.strip()

        if not stripped:
            raise AuditToolError(
                "column_name cannot be empty.",
                code="empty_column_name",
            )

        if stripped in self._columns:
            return stripped

        matches = self._column_lookup.get(
            stripped.casefold(),
            [],
        )

        if len(matches) == 1:
            return matches[0]

        if len(matches) > 1:
            raise AuditToolError(
                (
                    f"Column name {column_name!r} is ambiguous. "
                    "Use the exact case-sensitive name."
                ),
                code="ambiguous_column_name",
                details={"matches": matches},
            )

        raise AuditToolError(
            (
                f"Column {column_name!r} is not available in the "
                "active audit report."
            ),
            code="unknown_column",
            details={
                "available_columns": list(self._columns),
            },
        )

    def _run(
        self,
        tool_name: str,
        operation: Callable[
            [],
            tuple[str, Mapping[str, Any]],
        ],
    ) -> dict[str, Any]:
        try:
            message, data = operation()
            return _success_response(
                tool_name,
                message=message,
                data=data,
            )
        except AuditToolError as error:
            return _error_response(
                tool_name,
                error,
            )

    def get_dataset_overview(self) -> dict[str, Any]:
        """
        Return dimensions, column groups, file metadata, and basic rates.
        """

        def operation() -> tuple[str, Mapping[str, Any]]:
            overview = self._report["overview"]
            metadata = self._report["file_metadata"]

            data = {
                "report_id": self._report["report_id"],
                "generated_at_utc": self._report[
                    "generated_at_utc"
                ],
                "file": {
                    "file_name": metadata.get("file_name"),
                    "file_size_bytes": metadata.get(
                        "file_size_bytes"
                    ),
                    "encoding": metadata.get("encoding"),
                    "delimiter": metadata.get("delimiter"),
                    "fingerprint_sha256": metadata.get(
                        "fingerprint_sha256"
                    ),
                    "loader_warnings": metadata.get(
                        "loader_warnings",
                        [],
                    ),
                },
                "dimensions": {
                    "row_count": overview["row_count"],
                    "column_count": overview["column_count"],
                    "total_cells": overview["total_cells"],
                },
                "column_type_counts": {
                    "numeric": len(
                        overview["numeric_columns"]
                    ),
                    "categorical": len(
                        overview["categorical_columns"]
                    ),
                    "datetime": len(
                        overview["datetime_columns"]
                    ),
                    "boolean": len(
                        overview["boolean_columns"]
                    ),
                    "other": len(
                        overview["other_columns"]
                    ),
                },
                "columns_by_type": {
                    "numeric": overview["numeric_columns"],
                    "categorical": (
                        overview["categorical_columns"]
                    ),
                    "datetime": overview["datetime_columns"],
                    "boolean": overview["boolean_columns"],
                    "other": overview["other_columns"],
                },
                "basic_quality": {
                    "missing_cells": (
                        overview["missing_cells"]
                    ),
                    "missing_percentage": (
                        overview["missing_percentage"]
                    ),
                    "duplicate_rows": (
                        overview["duplicate_rows"]
                    ),
                    "duplicate_percentage": (
                        overview["duplicate_percentage"]
                    ),
                },
                "memory_usage_bytes": overview.get(
                    "memory_usage_bytes"
                ),
                "memory_usage_mb": overview.get(
                    "memory_usage_mb"
                ),
            }

            message = (
                "Dataset overview returned for "
                f"{overview['row_count']} rows and "
                f"{overview['column_count']} columns."
            )

            return message, data

        return self._run(
            "get_dataset_overview",
            operation,
        )

    def get_quality_summary(self) -> dict[str, Any]:
        """
        Return the quality score, readiness status, components, and caveats.
        """

        def operation() -> tuple[str, Mapping[str, Any]]:
            summary = self._report["summary"]

            data = {
                "report_id": self._report["report_id"],
                "headline": summary["headline"],
                "overall_score": summary["overall_score"],
                "score_band_status": summary[
                    "score_band_status"
                ],
                "readiness_status": summary[
                    "readiness_status"
                ],
                "status_adjusted_by_gates": summary[
                    "status_adjusted_by_gates"
                ],
                "readiness_gates": summary[
                    "readiness_gates"
                ],
                "highest_severity": summary[
                    "highest_severity"
                ],
                "issue_count_total": summary[
                    "issue_count_total"
                ],
                "issue_counts": summary["issue_counts"],
                "component_scores": self._report[
                    "component_scores"
                ],
                "weighted_contributions": self._report.get(
                    "weighted_contributions",
                    {},
                ),
                "score_weights": self._report.get(
                    "score_weights",
                    {},
                ),
                "limitations": self._report[
                    "limitations"
                ],
                "disclaimer": self._report["disclaimer"],
            }

            message = (
                f"Quality score {summary['overall_score']:.2f}; "
                f"final readiness status is "
                f"{summary['readiness_status']}."
            )

            return message, data

        return self._run(
            "get_quality_summary",
            operation,
        )

    def get_missing_value_report(
        self,
        limit: int = DEFAULT_RESULT_LIMIT,
        min_missing_percentage: float = 0.0,
        column_name: str | None = None,
    ) -> dict[str, Any]:
        """
        Return missing-value issues, optionally filtered by column and rate.
        """

        def operation() -> tuple[str, Mapping[str, Any]]:
            validated_limit = _validate_limit(limit)
            minimum_percentage = _validate_percentage(
                min_missing_percentage,
                parameter_name="min_missing_percentage",
            )
            resolved_column = (
                self._resolve_column_name(column_name)
                if column_name is not None
                else None
            )

            items = [
                issue
                for issue in self._report["issues"]
                if issue["check_name"] == "missing_values"
                and (
                    resolved_column is None
                    or issue["column"] == resolved_column
                )
                and float(issue.get("percentage") or 0.0)
                >= minimum_percentage
            ]

            returned_items = items[:validated_limit]

            data = {
                "filters": {
                    "column_name": resolved_column,
                    "min_missing_percentage": (
                        minimum_percentage
                    ),
                    "limit": validated_limit,
                },
                "dataset_missing_cells": self._report[
                    "overview"
                ]["missing_cells"],
                "dataset_missing_percentage": self._report[
                    "overview"
                ]["missing_percentage"],
                "total_matching_count": len(items),
                "returned_count": len(returned_items),
                "truncated": len(items) > validated_limit,
                "items": returned_items,
            }

            if returned_items:
                message = (
                    f"Returned {len(returned_items)} of "
                    f"{len(items)} matching missing-value issues."
                )
            elif resolved_column is not None:
                message = (
                    "No missing-value issues matched the requested "
                    f"filters for column {resolved_column!r}."
                )
            elif minimum_percentage > 0:
                message = (
                    "No missing-value issues matched the requested "
                    "minimum percentage."
                )
            else:
                message = (
                    "No missing-value issues were found in the "
                    "current audit."
                )

            return message, data

        return self._run(
            "get_missing_value_report",
            operation,
        )

    def get_duplicate_report(self) -> dict[str, Any]:
        """Return the exact-duplicate assessment from the audit report."""

        def operation() -> tuple[str, Mapping[str, Any]]:
            duplicate_issues = [
                issue
                for issue in self._report["issues"]
                if issue["check_name"] == "duplicates"
            ]

            issue = (
                duplicate_issues[0]
                if duplicate_issues
                else None
            )
            overview = self._report["overview"]

            data = {
                "has_exact_duplicates": (
                    overview["duplicate_rows"] > 0
                ),
                "duplicate_rows": overview[
                    "duplicate_rows"
                ],
                "duplicate_percentage": overview[
                    "duplicate_percentage"
                ],
                "method": (
                    issue["method"]
                    if issue is not None
                    else (
                        "Exact full-row comparison using "
                        "pandas duplicated()."
                    )
                ),
                "issue": issue,
            }

            if issue is None:
                message = (
                    "No extra exact duplicate rows were found in "
                    "the current audit."
                )
            else:
                message = (
                    f"Found {overview['duplicate_rows']} extra "
                    "exact duplicate rows "
                    f"({overview['duplicate_percentage']:.2f}%)."
                )

            return message, data

        return self._run(
            "get_duplicate_report",
            operation,
        )

    def get_column_quality_report(
        self,
        column_name: str,
        limit: int = DEFAULT_RESULT_LIMIT,
    ) -> dict[str, Any]:
        """
        Return all available quality evidence for one audited column.
        """

        def operation() -> tuple[str, Mapping[str, Any]]:
            validated_limit = _validate_limit(limit)
            resolved_column = self._resolve_column_name(
                column_name
            )

            issues = [
                issue
                for issue in self._report["issues"]
                if issue.get("column") == resolved_column
            ]
            recommendations = [
                recommendation
                for recommendation in self._report[
                    "prioritized_recommendations"
                ]
                if recommendation.get("column")
                == resolved_column
            ]

            diagnostics = self._report[
                "scoring_diagnostics"
            ]

            severity_counts = {
                severity: sum(
                    issue["severity"] == severity
                    for issue in issues
                )
                for severity in VALID_SEVERITIES
            }

            data = {
                "column_name": resolved_column,
                "column_type": self._column_type(
                    resolved_column
                ),
                "is_clean_in_current_audit": not issues,
                "issue_count": len(issues),
                "severity_counts": severity_counts,
                "returned_issue_count": min(
                    len(issues),
                    validated_limit,
                ),
                "issues_truncated": (
                    len(issues) > validated_limit
                ),
                "issues": issues[:validated_limit],
                "recommendations": recommendations[
                    :validated_limit
                ],
                "scoring_diagnostics": {
                    "schema_penalty": diagnostics.get(
                        "schema_penalty_by_column",
                        {},
                    ).get(resolved_column),
                    "category_affected_percentage": (
                        diagnostics.get(
                            (
                                "category_affected_percentage_"
                                "by_column"
                            ),
                            {},
                        ).get(resolved_column)
                    ),
                    "outlier_percentage": diagnostics.get(
                        "outlier_percentage_by_column",
                        {},
                    ).get(resolved_column),
                },
            }

            if issues:
                message = (
                    f"Column {resolved_column!r} has "
                    f"{len(issues)} quality issue(s) in the "
                    "current audit."
                )
            else:
                message = (
                    f"No quality issues were found for column "
                    f"{resolved_column!r} by the current generic "
                    "checks."
                )

            return message, data

        return self._run(
            "get_column_quality_report",
            operation,
        )

    def get_priority_issues(
        self,
        limit: int = DEFAULT_RESULT_LIMIT,
        minimum_severity: str = "LOW",
        check_name: str | None = None,
        column_name: str | None = None,
    ) -> dict[str, Any]:
        """
        Return prioritized issues filtered by severity, check, or column.
        """

        def operation() -> tuple[str, Mapping[str, Any]]:
            validated_limit = _validate_limit(limit)
            severity = _normalize_severity(
                minimum_severity
            )
            normalized_check = _normalize_check_name(
                check_name
            )
            resolved_column = (
                self._resolve_column_name(column_name)
                if column_name is not None
                else None
            )
            maximum_rank = _SEVERITY_RANK[severity]

            items = [
                issue
                for issue in self._report["issues"]
                if _SEVERITY_RANK[issue["severity"]]
                <= maximum_rank
                and (
                    normalized_check is None
                    or issue["check_name"] == normalized_check
                )
                and (
                    resolved_column is None
                    or issue["column"] == resolved_column
                )
            ]
            returned_items = items[:validated_limit]

            data = {
                "filters": {
                    "minimum_severity": severity,
                    "check_name": normalized_check,
                    "column_name": resolved_column,
                    "limit": validated_limit,
                },
                "total_matching_count": len(items),
                "returned_count": len(returned_items),
                "truncated": len(items) > validated_limit,
                "items": returned_items,
            }

            if returned_items:
                message = (
                    f"Returned {len(returned_items)} of "
                    f"{len(items)} prioritized issue(s)."
                )
            else:
                message = (
                    "No quality issues matched the requested "
                    "priority filters."
                )

            return message, data

        return self._run(
            "get_priority_issues",
            operation,
        )

    def get_ml_readiness_report(
        self,
        limit: int = DEFAULT_RESULT_LIMIT,
    ) -> dict[str, Any]:
        """
        Return a generic feature-quality view for ML preparation.

        This tool does not assess a target, leakage, model performance,
        fairness, validation strategy, or deployment readiness.
        """

        def operation() -> tuple[str, Mapping[str, Any]]:
            validated_limit = _validate_limit(limit)
            issues = self._report["issues"]

            categorized_items = [
                {
                    **issue,
                    "ml_category": _ML_CATEGORY_BY_CHECK[
                        issue["check_name"]
                    ],
                }
                for issue in issues
                if issue["check_name"]
                in _ML_CATEGORY_BY_CHECK
            ]

            category_counts = {
                category: sum(
                    item["ml_category"] == category
                    for item in categorized_items
                )
                for category in dict.fromkeys(
                    _ML_CATEGORY_BY_CHECK.values()
                )
            }
            category_counts = {
                key: value
                for key, value in category_counts.items()
                if value > 0
            }

            high_or_critical = sum(
                item["severity"] in {"CRITICAL", "HIGH"}
                for item in categorized_items
            )

            data = {
                "assessment_scope": (
                    "GENERIC_FEATURE_QUALITY_ONLY"
                ),
                "overall_quality_score": self._report[
                    "summary"
                ]["overall_score"],
                "general_readiness_status": self._report[
                    "summary"
                ]["readiness_status"],
                "component_scores": self._report[
                    "component_scores"
                ],
                "feature_quality_issue_count": len(
                    categorized_items
                ),
                "high_or_critical_issue_count": (
                    high_or_critical
                ),
                "issue_counts_by_ml_category": (
                    category_counts
                ),
                "returned_priority_item_count": min(
                    len(categorized_items),
                    validated_limit,
                ),
                "priority_items_truncated": (
                    len(categorized_items)
                    > validated_limit
                ),
                "priority_items": categorized_items[
                    :validated_limit
                ],
                "capability_boundaries": {
                    "target_column_assessed": False,
                    "target_balance_assessed": False,
                    "data_leakage_assessed": False,
                    "train_validation_split_assessed": False,
                    "model_performance_assessed": False,
                    "fairness_assessed": False,
                    "deployment_readiness_assessed": False,
                },
                "interpretation": (
                    "This is a heuristic feature-quality view, "
                    "not a machine-learning readiness "
                    "certification."
                ),
            }

            if categorized_items:
                message = (
                    "Generic ML feature-quality review returned "
                    f"{min(len(categorized_items), validated_limit)} "
                    f"of {len(categorized_items)} prioritized "
                    "issue(s)."
                )
            else:
                message = (
                    "No generic feature-quality issues were found "
                    "by the current checks. Target, leakage, split, "
                    "performance, and fairness evidence are still "
                    "unavailable."
                )

            return message, data

        return self._run(
            "get_ml_readiness_report",
            operation,
        )

    def get_tool_registry(
        self,
    ) -> dict[str, Callable[..., dict[str, Any]]]:
        """Return the registered bound tool functions by stable name."""

        return {
            name: getattr(self, name)
            for name in AUDIT_TOOL_NAMES
        }

    def get_tool_functions(
        self,
    ) -> tuple[Callable[..., dict[str, Any]], ...]:
        """Return bound callables suitable for Gemini tool registration."""

        registry = self.get_tool_registry()

        return tuple(
            registry[name]
            for name in AUDIT_TOOL_NAMES
        )

    def dispatch_tool(
        self,
        tool_name: str,
        arguments: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Execute one registered tool by name.

        This method is intended for manual Gemini function-call dispatch.
        Invalid tool names and arguments are returned as JSON-safe errors.
        """

        if not isinstance(tool_name, str) or not tool_name.strip():
            return _error_response(
                "dispatch_tool",
                AuditToolError(
                    "tool_name must be a non-empty string.",
                    code="invalid_tool_name",
                ),
            )

        normalized_name = tool_name.strip()

        if normalized_name not in AUDIT_TOOL_NAMES:
            return _error_response(
                normalized_name,
                AuditToolError(
                    (
                        f"Unknown audit tool {tool_name!r}. "
                        f"Allowed tools: "
                        f"{', '.join(AUDIT_TOOL_NAMES)}."
                    ),
                    code="unknown_tool",
                    details={
                        "allowed_tools": list(
                            AUDIT_TOOL_NAMES
                        )
                    },
                ),
            )

        if arguments is None:
            normalized_arguments: dict[str, Any] = {}
        elif isinstance(arguments, Mapping):
            normalized_arguments = dict(arguments)
        else:
            return _error_response(
                normalized_name,
                AuditToolError(
                    "Tool arguments must be an object.",
                    code="invalid_tool_arguments",
                ),
            )

        function = self.get_tool_registry()[
            normalized_name
        ]

        try:
            inspect.signature(function).bind(
                **normalized_arguments
            )
        except TypeError as exc:
            return _error_response(
                normalized_name,
                AuditToolError(
                    f"Invalid arguments for {normalized_name}: {exc}",
                    code="tool_argument_binding_error",
                ),
            )

        try:
            result = function(**normalized_arguments)
        except Exception as exc:  # Defensive boundary for SDK dispatch.
            return _error_response(
                normalized_name,
                AuditToolError(
                    (
                        f"Tool execution failed safely: "
                        f"{type(exc).__name__}."
                    ),
                    code="tool_execution_error",
                ),
            )

        return _strict_json_copy(result)


def build_audit_toolbox(
    audit_report: Mapping[str, Any],
) -> AuditToolbox:
    """Build a validated read-only toolbox for one active audit report."""

    return AuditToolbox(audit_report)
