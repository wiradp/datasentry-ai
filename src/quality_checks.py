from __future__ import annotations

import re
from dataclasses import dataclass
from numbers import Number
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
    is_string_dtype,
)


class QualityCheckError(ValueError):
    """Raised when a DataFrame cannot be analyzed safely."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "quality_check_error",
    ) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class QualityCheckConfig:
    """Heuristic thresholds used by deterministic quality checks."""

    missing_low_max_pct: float = 5.0
    missing_medium_max_pct: float = 20.0
    missing_high_max_pct: float = 40.0

    duplicate_low_max_pct: float = 1.0
    duplicate_medium_max_pct: float = 5.0
    duplicate_high_max_pct: float = 10.0

    near_constant_min_dominance_pct: float = 95.0
    near_constant_high_min_dominance_pct: float = 99.0

    potential_identifier_min_unique_pct: float = 95.0
    potential_identifier_name_min_unique_pct: float = 80.0

    high_cardinality_min_unique_count: int = 20
    high_cardinality_min_unique_pct: float = 20.0
    high_cardinality_high_min_unique_pct: float = 80.0

    outlier_iqr_multiplier: float = 1.5
    outlier_low_max_pct: float = 1.0
    outlier_medium_max_pct: float = 5.0
    outlier_high_max_pct: float = 10.0

    category_inconsistency_low_max_pct: float = 5.0
    category_inconsistency_medium_max_pct: float = 20.0

    numeric_like_text_min_parse_pct: float = 90.0
    datetime_like_text_min_parse_pct: float = 80.0
    datetime_mixed_format_min_count: int = 2

    def __post_init__(self) -> None:
        ordered_threshold_groups = {
            "missing": (
                self.missing_low_max_pct,
                self.missing_medium_max_pct,
                self.missing_high_max_pct,
            ),
            "duplicate": (
                self.duplicate_low_max_pct,
                self.duplicate_medium_max_pct,
                self.duplicate_high_max_pct,
            ),
            "outlier": (
                self.outlier_low_max_pct,
                self.outlier_medium_max_pct,
                self.outlier_high_max_pct,
            ),
        }

        for name, values in ordered_threshold_groups.items():
            if not (
                0
                <= values[0]
                <= values[1]
                <= values[2]
                <= 100
            ):
                raise ValueError(
                    f"{name} thresholds must be ordered "
                    "between 0 and 100."
                )

        percentage_fields = {
            "near_constant_min_dominance_pct": (
                self.near_constant_min_dominance_pct
            ),
            "near_constant_high_min_dominance_pct": (
                self.near_constant_high_min_dominance_pct
            ),
            "potential_identifier_min_unique_pct": (
                self.potential_identifier_min_unique_pct
            ),
            "potential_identifier_name_min_unique_pct": (
                self.potential_identifier_name_min_unique_pct
            ),
            "high_cardinality_min_unique_pct": (
                self.high_cardinality_min_unique_pct
            ),
            "high_cardinality_high_min_unique_pct": (
                self.high_cardinality_high_min_unique_pct
            ),
            "category_inconsistency_low_max_pct": (
                self.category_inconsistency_low_max_pct
            ),
            "category_inconsistency_medium_max_pct": (
                self.category_inconsistency_medium_max_pct
            ),
            "numeric_like_text_min_parse_pct": (
                self.numeric_like_text_min_parse_pct
            ),
            "datetime_like_text_min_parse_pct": (
                self.datetime_like_text_min_parse_pct
            ),
        }

        for name, value in percentage_fields.items():
            if not 0 <= value <= 100:
                raise ValueError(
                    f"{name} must be between 0 and 100."
                )

        if (
            self.near_constant_high_min_dominance_pct
            < self.near_constant_min_dominance_pct
        ):
            raise ValueError(
                "near-constant high threshold cannot be lower "
                "than its minimum threshold."
            )

        if (
            self.high_cardinality_high_min_unique_pct
            < self.high_cardinality_min_unique_pct
        ):
            raise ValueError(
                "high-cardinality high threshold cannot be lower "
                "than its minimum threshold."
            )

        if self.high_cardinality_min_unique_count < 2:
            raise ValueError(
                "high_cardinality_min_unique_count must be at least 2."
            )

        if self.outlier_iqr_multiplier <= 0:
            raise ValueError(
                "outlier_iqr_multiplier must be greater than 0."
            )

        if (
            self.category_inconsistency_medium_max_pct
            < self.category_inconsistency_low_max_pct
        ):
            raise ValueError(
                "category inconsistency thresholds must be ordered."
            )

        if self.datetime_mixed_format_min_count < 2:
            raise ValueError(
                "datetime_mixed_format_min_count must be at least 2."
            )


def _validate_dataframe(dataframe: pd.DataFrame) -> None:
    """Validate the input contract for quality checks."""

    if not isinstance(dataframe, pd.DataFrame):
        raise QualityCheckError(
            "Input must be a pandas DataFrame.",
            code="invalid_dataframe_type",
        )

    if dataframe.shape[1] == 0:
        raise QualityCheckError(
            "DataFrame does not contain columns.",
            code="no_columns",
        )

    if dataframe.shape[0] == 0:
        raise QualityCheckError(
            "DataFrame does not contain data rows.",
            code="no_rows",
        )

    normalized_columns = pd.Index(
        [
            str(column).strip().casefold()
            for column in dataframe.columns
        ]
    )

    duplicate_mask = normalized_columns.duplicated(
        keep=False
    )

    if duplicate_mask.any():
        duplicate_columns = sorted(
            set(normalized_columns[duplicate_mask])
        )

        raise QualityCheckError(
            (
                "Duplicate column names detected: "
                f"{', '.join(duplicate_columns)}"
            ),
            code="duplicate_columns",
        )


def _percentage(
    numerator: int,
    denominator: int,
) -> float:
    """Return a percentage rounded to two decimal places."""

    if denominator == 0:
        return 0.0

    return round(
        numerator / denominator * 100.0,
        2,
    )


def _classify_columns(
    dataframe: pd.DataFrame,
) -> dict[str, list[str]]:
    """Group columns by broad analytical data type."""

    groups: dict[str, list[str]] = {
        "numeric": [],
        "categorical": [],
        "datetime": [],
        "boolean": [],
        "other": [],
    }

    for column, dtype in dataframe.dtypes.items():
        column_name = str(column)

        if is_bool_dtype(dtype):
            groups["boolean"].append(column_name)

        elif is_numeric_dtype(dtype):
            groups["numeric"].append(column_name)

        elif is_datetime64_any_dtype(dtype):
            groups["datetime"].append(column_name)

        elif (
            isinstance(dtype, pd.CategoricalDtype)
            or is_object_dtype(dtype)
            or is_string_dtype(dtype)
        ):
            groups["categorical"].append(column_name)

        else:
            groups["other"].append(column_name)

    return groups


def get_dataset_overview(
    dataframe: pd.DataFrame,
) -> dict[str, Any]:
    """Return a deterministic summary of the dataset."""

    _validate_dataframe(dataframe)

    row_count, column_count = dataframe.shape
    total_cells = row_count * column_count

    missing_cells = int(
        dataframe.isna().to_numpy().sum()
    )

    duplicate_rows = int(
        dataframe.duplicated(
            keep="first"
        ).sum()
    )

    column_groups = _classify_columns(dataframe)

    memory_usage_bytes = int(
        dataframe.memory_usage(
            index=True,
            deep=True,
        ).sum()
    )

    return {
        "row_count": int(row_count),
        "column_count": int(column_count),
        "total_cells": int(total_cells),
        "missing_cells": missing_cells,
        "missing_percentage": _percentage(
            missing_cells,
            total_cells,
        ),
        "duplicate_rows": duplicate_rows,
        "duplicate_percentage": _percentage(
            duplicate_rows,
            row_count,
        ),
        "numeric_column_count": len(
            column_groups["numeric"]
        ),
        "categorical_column_count": len(
            column_groups["categorical"]
        ),
        "datetime_column_count": len(
            column_groups["datetime"]
        ),
        "boolean_column_count": len(
            column_groups["boolean"]
        ),
        "other_column_count": len(
            column_groups["other"]
        ),
        "numeric_columns": column_groups["numeric"],
        "categorical_columns": (
            column_groups["categorical"]
        ),
        "datetime_columns": (
            column_groups["datetime"]
        ),
        "boolean_columns": column_groups["boolean"],
        "other_columns": column_groups["other"],
        "memory_usage_bytes": memory_usage_bytes,
        "memory_usage_mb": round(
            memory_usage_bytes / (1024**2),
            4,
        ),
    }


def _missing_severity(
    missing_percentage: float,
    config: QualityCheckConfig,
) -> str:
    """Map missing percentage to a heuristic severity."""

    if missing_percentage == 0:
        return "PASS"

    if (
        missing_percentage
        <= config.missing_low_max_pct
    ):
        return "LOW"

    if (
        missing_percentage
        <= config.missing_medium_max_pct
    ):
        return "MEDIUM"

    if (
        missing_percentage
        <= config.missing_high_max_pct
    ):
        return "HIGH"

    return "CRITICAL"


def _missing_recommendation(
    severity: str,
) -> str:
    """Return an action-oriented missing-value recommendation."""

    recommendations = {
        "PASS": (
            "No missing values detected."
        ),
        "LOW": (
            "Review the missing pattern and confirm "
            "whether treatment is necessary."
        ),
        "MEDIUM": (
            "Investigate the missing pattern and define "
            "an imputation or exclusion rule."
        ),
        "HIGH": (
            "Prioritize source validation and remediation "
            "before downstream analysis."
        ),
        "CRITICAL": (
            "Consider excluding or recovering the column "
            "unless it is business-critical."
        ),
    }

    return recommendations[severity]


def analyze_missing_values(
    dataframe: pd.DataFrame,
    *,
    config: QualityCheckConfig | None = None,
    include_complete: bool = True,
) -> pd.DataFrame:
    """
    Analyze missing values per column.

    By default, the report includes columns without missing
    values so the result is also useful as an audit trail.
    """

    _validate_dataframe(dataframe)

    active_config = config or QualityCheckConfig()
    row_count = int(dataframe.shape[0])

    records: list[dict[str, Any]] = []

    for column in dataframe.columns:
        series = dataframe[column]

        missing_count = int(
            series.isna().sum()
        )

        missing_percentage = _percentage(
            missing_count,
            row_count,
        )

        severity = _missing_severity(
            missing_percentage,
            active_config,
        )

        records.append(
            {
                "column": str(column),
                "dtype": str(series.dtype),
                "missing_count": missing_count,
                "missing_percentage": (
                    missing_percentage
                ),
                "non_missing_count": (
                    row_count - missing_count
                ),
                "severity": severity,
                "status": (
                    "PASS"
                    if missing_count == 0
                    else "ISSUE"
                ),
                "recommendation": (
                    _missing_recommendation(severity)
                ),
            }
        )

    report = pd.DataFrame(records)

    if not include_complete:
        report = report.loc[
            report["missing_count"] > 0
        ]

    if not report.empty:
        report = (
            report.sort_values(
                by=[
                    "missing_percentage",
                    "column",
                ],
                ascending=[False, True],
                kind="stable",
            )
            .reset_index(drop=True)
        )

    return report


def _duplicate_severity(
    duplicate_percentage: float,
    config: QualityCheckConfig,
) -> str:
    """Map exact duplicate percentage to a severity."""

    if duplicate_percentage == 0:
        return "PASS"

    if (
        duplicate_percentage
        <= config.duplicate_low_max_pct
    ):
        return "LOW"

    if (
        duplicate_percentage
        <= config.duplicate_medium_max_pct
    ):
        return "MEDIUM"

    if (
        duplicate_percentage
        <= config.duplicate_high_max_pct
    ):
        return "HIGH"

    return "CRITICAL"


def analyze_duplicates(
    dataframe: pd.DataFrame,
    *,
    config: QualityCheckConfig | None = None,
    max_example_indices: int = 10,
) -> dict[str, Any]:
    """Analyze exact duplicate rows without modifying data."""

    _validate_dataframe(dataframe)

    if max_example_indices < 0:
        raise ValueError(
            "max_example_indices cannot be negative."
        )

    active_config = config or QualityCheckConfig()
    row_count = int(dataframe.shape[0])

    extra_duplicate_mask = dataframe.duplicated(
        keep="first"
    )

    all_duplicate_mask = dataframe.duplicated(
        keep=False
    )

    duplicate_rows = int(
        extra_duplicate_mask.sum()
    )

    rows_in_duplicate_groups = int(
        all_duplicate_mask.sum()
    )

    duplicate_group_count = 0

    if rows_in_duplicate_groups:
        duplicate_group_count = int(
            dataframe.loc[
                all_duplicate_mask
            ]
            .drop_duplicates()
            .shape[0]
        )

    duplicate_percentage = _percentage(
        duplicate_rows,
        row_count,
    )

    severity = _duplicate_severity(
        duplicate_percentage,
        active_config,
    )

    example_indices = [
        str(index)
        for index in dataframe.index[
            extra_duplicate_mask
        ].tolist()[:max_example_indices]
    ]

    recommendation = (
        "No exact duplicate rows detected."
        if duplicate_rows == 0
        else (
            "Review duplicate groups before removing "
            "records; confirm whether repeated rows are "
            "legitimate events."
        )
    )

    return {
        "duplicate_rows": duplicate_rows,
        "duplicate_percentage": (
            duplicate_percentage
        ),
        "rows_in_duplicate_groups": (
            rows_in_duplicate_groups
        ),
        "duplicate_group_count": (
            duplicate_group_count
        ),
        "severity": severity,
        "status": (
            "PASS"
            if duplicate_rows == 0
            else "ISSUE"
        ),
        "example_duplicate_indices": (
            example_indices
        ),
        "recommendation": recommendation,
    }


def _safe_scalar(
    value: Any,
) -> str | int | float | bool | None:
    """Convert a scalar value into a serializable preview."""

    if hasattr(value, "item"):
        try:
            value = value.item()
        except (TypeError, ValueError):
            pass

    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass

    if isinstance(
        value,
        (str, Number, bool),
    ):
        result: Any = value

    else:
        result = str(value)

    if (
        isinstance(result, str)
        and len(result) > 100
    ):
        return f"{result[:97]}..."

    return result


def analyze_constant_columns(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Detect constant and entirely missing columns.

    A constant column has exactly one distinct non-null value.
    An all-missing column is reported separately as critical.
    """

    _validate_dataframe(dataframe)

    row_count = int(dataframe.shape[0])
    records: list[dict[str, Any]] = []

    for column in dataframe.columns:
        series = dataframe[column]

        missing_count = int(
            series.isna().sum()
        )

        unique_non_null = int(
            series.nunique(dropna=True)
        )

        if unique_non_null == 0:
            status = "ALL_MISSING"
            severity = "CRITICAL"
            constant_value = None
            recommendation = (
                "Recover the column from the source or "
                "exclude it from downstream analysis."
            )

        elif unique_non_null == 1:
            status = "CONSTANT"
            severity = "HIGH"

            constant_value = _safe_scalar(
                series.dropna().iloc[0]
            )

            recommendation = (
                "Review whether the column has business "
                "value; constant columns usually add no "
                "analytical variation."
            )

        else:
            continue

        records.append(
            {
                "column": str(column),
                "dtype": str(series.dtype),
                "status": status,
                "severity": severity,
                "unique_non_null": unique_non_null,
                "missing_count": missing_count,
                "missing_percentage": _percentage(
                    missing_count,
                    row_count,
                ),
                "constant_value": constant_value,
                "recommendation": recommendation,
            }
        )

    columns = [
        "column",
        "dtype",
        "status",
        "severity",
        "unique_non_null",
        "missing_count",
        "missing_percentage",
        "constant_value",
        "recommendation",
    ]

    report = pd.DataFrame(
        records,
        columns=columns,
    )

    if not report.empty:
        severity_rank = {
            "CRITICAL": 0,
            "HIGH": 1,
        }

        report = (
            report.assign(
                _severity_rank=report[
                    "severity"
                ].map(severity_rank)
            )
            .sort_values(
                by=[
                    "_severity_rank",
                    "column",
                ],
                kind="stable",
            )
            .drop(columns="_severity_rank")
            .reset_index(drop=True)
        )

    return report


def analyze_near_constant_columns(
    dataframe: pd.DataFrame,
    *,
    config: QualityCheckConfig | None = None,
) -> pd.DataFrame:
    """Detect columns dominated by one non-null value."""

    _validate_dataframe(dataframe)
    active_config = config or QualityCheckConfig()
    row_count = int(dataframe.shape[0])
    records: list[dict[str, Any]] = []

    for column in dataframe.columns:
        series = dataframe[column]
        non_null = series.dropna()
        non_null_count = int(non_null.shape[0])
        unique_non_null = int(non_null.nunique())

        # Constants and all-missing columns are handled separately.
        if non_null_count == 0 or unique_non_null <= 1:
            continue

        value_counts = non_null.value_counts(
            dropna=False
        )
        dominant_value = value_counts.index[0]
        dominant_count = int(value_counts.iloc[0])
        dominance_percentage = _percentage(
            dominant_count,
            non_null_count,
        )

        if (
            dominance_percentage
            < active_config.near_constant_min_dominance_pct
        ):
            continue

        severity = (
            "HIGH"
            if dominance_percentage
            >= active_config.near_constant_high_min_dominance_pct
            else "MEDIUM"
        )

        records.append(
            {
                "column": str(column),
                "dtype": str(series.dtype),
                "status": "NEAR_CONSTANT",
                "severity": severity,
                "non_null_count": non_null_count,
                "unique_non_null": unique_non_null,
                "dominant_value": _safe_scalar(
                    dominant_value
                ),
                "dominant_count": dominant_count,
                "dominance_percentage": (
                    dominance_percentage
                ),
                "missing_count": int(
                    series.isna().sum()
                ),
                "missing_percentage": _percentage(
                    int(series.isna().sum()),
                    row_count,
                ),
                "recommendation": (
                    "Confirm whether the rare values are valid. "
                    "Near-constant columns may add little signal "
                    "and can distort some analyses."
                ),
            }
        )

    columns = [
        "column",
        "dtype",
        "status",
        "severity",
        "non_null_count",
        "unique_non_null",
        "dominant_value",
        "dominant_count",
        "dominance_percentage",
        "missing_count",
        "missing_percentage",
        "recommendation",
    ]

    report = pd.DataFrame(records, columns=columns)

    if not report.empty:
        report = (
            report.sort_values(
                by=[
                    "dominance_percentage",
                    "column",
                ],
                ascending=[False, True],
                kind="stable",
            )
            .reset_index(drop=True)
        )

    return report


def _identifier_name_hint(column: Any) -> bool:
    """Return whether a column name resembles an identifier."""

    normalized = str(column).strip().casefold()

    if normalized.startswith("unnamed:"):
        return True

    tokens = {
        token
        for token in re.split(r"[^a-z0-9]+", normalized)
        if token
    }

    identifier_tokens = {
        "id",
        "uuid",
        "guid",
        "identifier",
        "key",
        "index",
        "code",
    }

    return bool(tokens & identifier_tokens)


def analyze_potential_identifiers(
    dataframe: pd.DataFrame,
    *,
    config: QualityCheckConfig | None = None,
) -> pd.DataFrame:
    """
    Detect columns that may be row identifiers.

    Numeric columns require an identifier-like name to avoid
    flagging ordinary continuous measurements solely because
    they contain many unique values.
    """

    _validate_dataframe(dataframe)
    active_config = config or QualityCheckConfig()
    records: list[dict[str, Any]] = []

    for column in dataframe.columns:
        series = dataframe[column]
        non_null = series.dropna()
        non_null_count = int(non_null.shape[0])

        if non_null_count == 0:
            continue

        unique_non_null = int(non_null.nunique())

        if unique_non_null <= 1:
            continue

        unique_percentage = _percentage(
            unique_non_null,
            non_null_count,
        )

        name_hint = _identifier_name_hint(column)
        categorical_like = bool(
            isinstance(series.dtype, pd.CategoricalDtype)
            or is_object_dtype(series.dtype)
            or is_string_dtype(series.dtype)
        )

        detected_by_name = bool(
            name_hint
            and unique_percentage
            >= active_config.potential_identifier_name_min_unique_pct
        )

        detected_by_uniqueness = bool(
            categorical_like
            and unique_percentage
            >= active_config.potential_identifier_min_unique_pct
        )

        if not (
            detected_by_name
            or detected_by_uniqueness
        ):
            continue

        if detected_by_name and detected_by_uniqueness:
            detection_reason = (
                "IDENTIFIER_NAME_AND_HIGH_UNIQUENESS"
            )
        elif detected_by_name:
            detection_reason = (
                "IDENTIFIER_NAME_PATTERN"
            )
        else:
            detection_reason = (
                "HIGH_UNIQUENESS_CATEGORICAL"
            )

        records.append(
            {
                "column": str(column),
                "dtype": str(series.dtype),
                "status": "POTENTIAL_IDENTIFIER",
                "severity": (
                    "HIGH" if name_hint else "MEDIUM"
                ),
                "non_null_count": non_null_count,
                "unique_non_null": unique_non_null,
                "unique_percentage": unique_percentage,
                "name_hint": name_hint,
                "detection_reason": detection_reason,
                "recommendation": (
                    "Verify the business meaning and exclude the "
                    "column from model features when it only "
                    "identifies records."
                ),
            }
        )

    columns = [
        "column",
        "dtype",
        "status",
        "severity",
        "non_null_count",
        "unique_non_null",
        "unique_percentage",
        "name_hint",
        "detection_reason",
        "recommendation",
    ]

    report = pd.DataFrame(records, columns=columns)

    if not report.empty:
        severity_rank = {
            "HIGH": 0,
            "MEDIUM": 1,
        }

        report = (
            report.assign(
                _severity_rank=report[
                    "severity"
                ].map(severity_rank)
            )
            .sort_values(
                by=[
                    "_severity_rank",
                    "unique_percentage",
                    "column",
                ],
                ascending=[True, False, True],
                kind="stable",
            )
            .drop(columns="_severity_rank")
            .reset_index(drop=True)
        )

    return report


def analyze_high_cardinality_categories(
    dataframe: pd.DataFrame,
    *,
    config: QualityCheckConfig | None = None,
) -> pd.DataFrame:
    """Detect categorical columns with many distinct values."""

    _validate_dataframe(dataframe)
    active_config = config or QualityCheckConfig()
    records: list[dict[str, Any]] = []

    for column in dataframe.columns:
        series = dataframe[column]
        categorical_like = bool(
            isinstance(series.dtype, pd.CategoricalDtype)
            or is_object_dtype(series.dtype)
            or is_string_dtype(series.dtype)
        )

        if not categorical_like or is_bool_dtype(series.dtype):
            continue

        non_null = series.dropna()
        non_null_count = int(non_null.shape[0])

        if non_null_count == 0:
            continue

        unique_non_null = int(non_null.nunique())
        unique_percentage = _percentage(
            unique_non_null,
            non_null_count,
        )

        if (
            unique_non_null
            < active_config.high_cardinality_min_unique_count
            or unique_percentage
            < active_config.high_cardinality_min_unique_pct
        ):
            continue

        severity = (
            "HIGH"
            if unique_percentage
            >= active_config.high_cardinality_high_min_unique_pct
            else "MEDIUM"
        )

        records.append(
            {
                "column": str(column),
                "dtype": str(series.dtype),
                "status": "HIGH_CARDINALITY",
                "severity": severity,
                "non_null_count": non_null_count,
                "unique_non_null": unique_non_null,
                "unique_percentage": unique_percentage,
                "potential_identifier_name": (
                    _identifier_name_hint(column)
                ),
                "recommendation": (
                    "Review whether the column should be treated "
                    "as an identifier, transformed, grouped, or "
                    "excluded before encoding."
                ),
            }
        )

    columns = [
        "column",
        "dtype",
        "status",
        "severity",
        "non_null_count",
        "unique_non_null",
        "unique_percentage",
        "potential_identifier_name",
        "recommendation",
    ]

    report = pd.DataFrame(records, columns=columns)

    if not report.empty:
        report = (
            report.sort_values(
                by=[
                    "unique_percentage",
                    "unique_non_null",
                    "column",
                ],
                ascending=[False, False, True],
                kind="stable",
            )
            .reset_index(drop=True)
        )

    return report


def _outlier_severity(
    outlier_percentage: float,
    config: QualityCheckConfig,
) -> str:
    """Map numeric outlier percentage to a severity."""

    if outlier_percentage == 0:
        return "PASS"

    if outlier_percentage <= config.outlier_low_max_pct:
        return "LOW"

    if outlier_percentage <= config.outlier_medium_max_pct:
        return "MEDIUM"

    if outlier_percentage <= config.outlier_high_max_pct:
        return "HIGH"

    return "CRITICAL"


def analyze_numeric_outliers(
    dataframe: pd.DataFrame,
    *,
    config: QualityCheckConfig | None = None,
    include_clean: bool = False,
    max_example_values: int = 5,
) -> pd.DataFrame:
    """Detect numeric outliers using the IQR rule."""

    _validate_dataframe(dataframe)

    if max_example_values < 0:
        raise ValueError(
            "max_example_values cannot be negative."
        )

    active_config = config or QualityCheckConfig()
    records: list[dict[str, Any]] = []

    for column in dataframe.columns:
        series = dataframe[column]

        if (
            not is_numeric_dtype(series.dtype)
            or is_bool_dtype(series.dtype)
        ):
            continue

        numeric_series = pd.to_numeric(
            series,
            errors="coerce",
        )

        present_mask = numeric_series.notna()
        non_null_count = int(present_mask.sum())

        if non_null_count == 0:
            continue

        finite_mask = present_mask & np.isfinite(
            numeric_series
        )
        infinite_mask = present_mask & ~np.isfinite(
            numeric_series
        )
        finite_values = numeric_series.loc[
            finite_mask
        ]

        q1: float | None = None
        q3: float | None = None
        iqr: float | None = None
        lower_bound: float | None = None
        upper_bound: float | None = None

        outlier_mask = infinite_mask.copy()

        if not finite_values.empty:
            q1 = float(finite_values.quantile(0.25))
            q3 = float(finite_values.quantile(0.75))
            iqr = float(q3 - q1)
            lower_bound = float(
                q1 - active_config.outlier_iqr_multiplier * iqr
            )
            upper_bound = float(
                q3 + active_config.outlier_iqr_multiplier * iqr
            )

            outlier_mask = outlier_mask | (
                finite_mask
                & (
                    (numeric_series < lower_bound)
                    | (numeric_series > upper_bound)
                )
            )

        outlier_count = int(outlier_mask.sum())
        outlier_percentage = _percentage(
            outlier_count,
            non_null_count,
        )

        if outlier_count == 0 and not include_clean:
            continue

        outlier_values = numeric_series.loc[
            outlier_mask
        ]

        example_values = [
            _safe_scalar(value)
            for value in outlier_values.head(
                max_example_values
            ).tolist()
        ]

        severity = _outlier_severity(
            outlier_percentage,
            active_config,
        )

        records.append(
            {
                "column": str(column),
                "dtype": str(series.dtype),
                "status": (
                    "PASS"
                    if outlier_count == 0
                    else "OUTLIERS_DETECTED"
                ),
                "severity": severity,
                "method": "IQR",
                "iqr_multiplier": (
                    active_config.outlier_iqr_multiplier
                ),
                "non_null_count": non_null_count,
                "finite_count": int(finite_mask.sum()),
                "infinite_count": int(infinite_mask.sum()),
                "q1": (
                    round(q1, 4)
                    if q1 is not None
                    else None
                ),
                "q3": (
                    round(q3, 4)
                    if q3 is not None
                    else None
                ),
                "iqr": (
                    round(iqr, 4)
                    if iqr is not None
                    else None
                ),
                "lower_bound": (
                    round(lower_bound, 4)
                    if lower_bound is not None
                    else None
                ),
                "upper_bound": (
                    round(upper_bound, 4)
                    if upper_bound is not None
                    else None
                ),
                "outlier_count": outlier_count,
                "outlier_percentage": outlier_percentage,
                "minimum_outlier": (
                    _safe_scalar(outlier_values.min())
                    if outlier_count
                    else None
                ),
                "maximum_outlier": (
                    _safe_scalar(outlier_values.max())
                    if outlier_count
                    else None
                ),
                "example_outlier_values": example_values,
                "recommendation": (
                    "Inspect source records and domain rules. "
                    "An IQR outlier is not automatically an error."
                    if outlier_count
                    else "No IQR outliers detected."
                ),
            }
        )

    columns = [
        "column",
        "dtype",
        "status",
        "severity",
        "method",
        "iqr_multiplier",
        "non_null_count",
        "finite_count",
        "infinite_count",
        "q1",
        "q3",
        "iqr",
        "lower_bound",
        "upper_bound",
        "outlier_count",
        "outlier_percentage",
        "minimum_outlier",
        "maximum_outlier",
        "example_outlier_values",
        "recommendation",
    ]

    report = pd.DataFrame(records, columns=columns)

    if not report.empty:
        severity_rank = {
            "CRITICAL": 0,
            "HIGH": 1,
            "MEDIUM": 2,
            "LOW": 3,
            "PASS": 4,
        }

        report = (
            report.assign(
                _severity_rank=report[
                    "severity"
                ].map(severity_rank)
            )
            .sort_values(
                by=[
                    "_severity_rank",
                    "outlier_percentage",
                    "column",
                ],
                ascending=[True, False, True],
                kind="stable",
            )
            .drop(columns="_severity_rank")
            .reset_index(drop=True)
        )

    return report



def _normalize_category_value(value: Any) -> str:
    """Normalize text for comparison without changing source data."""

    collapsed = re.sub(
        r"\s+",
        " ",
        str(value).strip(),
    )

    return collapsed.casefold()


def _collapse_whitespace(value: Any) -> str:
    """Trim and collapse repeated whitespace for diagnostics."""

    return re.sub(
        r"\s+",
        " ",
        str(value).strip(),
    )


def _category_inconsistency_severity(
    affected_percentage: float,
    config: QualityCheckConfig,
) -> str:
    """Map affected category percentage to a heuristic severity."""

    if (
        affected_percentage
        <= config.category_inconsistency_low_max_pct
    ):
        return "LOW"

    if (
        affected_percentage
        <= config.category_inconsistency_medium_max_pct
    ):
        return "MEDIUM"

    return "HIGH"


def analyze_category_consistency(
    dataframe: pd.DataFrame,
    *,
    config: QualityCheckConfig | None = None,
    max_example_variants: int = 10,
) -> pd.DataFrame:
    """
    Detect text categories that differ only by case or whitespace.

    The check is read-only. It reports candidate canonical groups but
    does not replace any source values.
    """

    _validate_dataframe(dataframe)

    if max_example_variants < 2:
        raise ValueError(
            "max_example_variants must be at least 2."
        )

    active_config = config or QualityCheckConfig()
    records: list[dict[str, Any]] = []

    for column in dataframe.columns:
        series = dataframe[column]

        if not (
            isinstance(series.dtype, pd.CategoricalDtype)
            or is_object_dtype(series.dtype)
            or is_string_dtype(series.dtype)
        ):
            continue

        non_null = series.dropna()
        non_null_count = int(non_null.shape[0])

        if non_null_count == 0:
            continue

        grouped_values: dict[str, dict[str, int]] = {}

        for value in non_null.tolist():
            raw_value = str(value)
            normalized_value = _normalize_category_value(
                raw_value
            )

            if not normalized_value:
                continue

            variant_counts = grouped_values.setdefault(
                normalized_value,
                {},
            )

            variant_counts[raw_value] = (
                variant_counts.get(raw_value, 0) + 1
            )

        for normalized_value, variant_counts in (
            grouped_values.items()
        ):
            if len(variant_counts) < 2:
                continue

            variants_sorted = sorted(
                variant_counts,
                key=lambda value: (
                    -variant_counts[value],
                    value.casefold(),
                    value,
                ),
            )

            affected_count = int(
                sum(variant_counts.values())
            )
            affected_percentage = _percentage(
                affected_count,
                non_null_count,
            )

            collapsed_variants = {
                _collapse_whitespace(value)
                for value in variant_counts
            }

            whitespace_variant_detected = any(
                value != _collapse_whitespace(value)
                for value in variant_counts
            )

            case_variant_detected = (
                len(collapsed_variants) > 1
                and len(
                    {
                        value.casefold()
                        for value in collapsed_variants
                    }
                )
                == 1
            )

            severity = _category_inconsistency_severity(
                affected_percentage,
                active_config,
            )

            records.append(
                {
                    "column": str(column),
                    "dtype": str(series.dtype),
                    "status": "CATEGORY_INCONSISTENCY",
                    "severity": severity,
                    "normalized_value": normalized_value,
                    "variant_count": len(variant_counts),
                    "variants": [
                        _safe_scalar(value)
                        for value in variants_sorted[
                            :max_example_variants
                        ]
                    ],
                    "variant_frequencies": {
                        value: int(variant_counts[value])
                        for value in variants_sorted[
                            :max_example_variants
                        ]
                    },
                    "affected_count": affected_count,
                    "affected_percentage": (
                        affected_percentage
                    ),
                    "non_null_count": non_null_count,
                    "whitespace_variant_detected": (
                        whitespace_variant_detected
                    ),
                    "case_variant_detected": (
                        case_variant_detected
                    ),
                    "recommendation": (
                        "Confirm the intended canonical label, then "
                        "standardize case and surrounding whitespace "
                        "before grouping or encoding."
                    ),
                }
            )

    columns = [
        "column",
        "dtype",
        "status",
        "severity",
        "normalized_value",
        "variant_count",
        "variants",
        "variant_frequencies",
        "affected_count",
        "affected_percentage",
        "non_null_count",
        "whitespace_variant_detected",
        "case_variant_detected",
        "recommendation",
    ]

    report = pd.DataFrame(records, columns=columns)

    if not report.empty:
        severity_rank = {
            "HIGH": 0,
            "MEDIUM": 1,
            "LOW": 2,
        }

        report = (
            report.assign(
                _severity_rank=report["severity"].map(
                    severity_rank
                )
            )
            .sort_values(
                by=[
                    "_severity_rank",
                    "affected_percentage",
                    "column",
                    "normalized_value",
                ],
                ascending=[True, False, True, True],
                kind="stable",
            )
            .drop(columns="_severity_rank")
            .reset_index(drop=True)
        )

    return report


_DATE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "ISO_DATETIME",
        re.compile(
            r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}"
        ),
    ),
    (
        "YYYY-MM-DD",
        re.compile(r"^\d{4}-\d{2}-\d{2}$"),
    ),
    (
        "YYYY/MM/DD",
        re.compile(r"^\d{4}/\d{2}/\d{2}$"),
    ),
    (
        "SLASH_DATE",
        re.compile(r"^\d{1,2}/\d{1,2}/\d{4}$"),
    ),
    (
        "MON_DD_YYYY",
        re.compile(
            r"^[A-Za-z]{3,9}\s+\d{1,2},\s+\d{4}$"
        ),
    ),
    (
        "DD_MON_YYYY",
        re.compile(
            r"^\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4}$"
        ),
    ),
)


def _date_format_label(value: str) -> str | None:
    """Return a conservative date-format label for a text value."""

    stripped = value.strip()

    for label, pattern in _DATE_PATTERNS:
        if pattern.fullmatch(stripped):
            return label

    return None


def _looks_like_unnamed_column(column: Any) -> bool:
    """Return whether a column name resembles an exported index."""

    normalized = str(column).strip().casefold()

    return bool(
        re.match(r"^unnamed(?::|\s|$)", normalized)
    )


def _looks_like_datetime_name(column: Any) -> bool:
    """Return whether a column name suggests date/time semantics."""

    normalized = re.sub(
        r"[^a-z0-9]+",
        "_",
        str(column).strip().casefold(),
    ).strip("_")

    tokens = set(normalized.split("_"))

    return bool(
        tokens
        & {
            "date",
            "datetime",
            "time",
            "timestamp",
            "created",
            "updated",
            "dob",
        }
    )


def analyze_data_type_warnings(
    dataframe: pd.DataFrame,
    *,
    config: QualityCheckConfig | None = None,
    max_example_values: int = 5,
) -> pd.DataFrame:
    """
    Detect likely schema issues without coercing source columns.

    Current warnings cover unnamed index-like columns, numeric-looking
    text, datetime-looking text, mixed date formats, and invalid values
    within otherwise parseable text columns.
    """

    _validate_dataframe(dataframe)

    if max_example_values < 0:
        raise ValueError(
            "max_example_values cannot be negative."
        )

    active_config = config or QualityCheckConfig()
    records: list[dict[str, Any]] = []

    for column in dataframe.columns:
        series = dataframe[column]
        column_name = str(column)
        non_null_count = int(series.notna().sum())

        if _looks_like_unnamed_column(column_name):
            records.append(
                {
                    "column": column_name,
                    "dtype": str(series.dtype),
                    "status": "UNNAMED_COLUMN",
                    "severity": "MEDIUM",
                    "inferred_type": "INDEX_ARTIFACT",
                    "non_null_count": non_null_count,
                    "parse_success_count": None,
                    "parse_success_percentage": None,
                    "invalid_count": None,
                    "invalid_percentage": None,
                    "detected_format_count": None,
                    "detected_formats": [],
                    "example_invalid_values": [],
                    "recommendation": (
                        "Confirm whether this is an exported DataFrame "
                        "index. Exclude it only after verifying that it "
                        "has no business meaning."
                    ),
                }
            )

        if not (
            isinstance(series.dtype, pd.CategoricalDtype)
            or is_object_dtype(series.dtype)
            or is_string_dtype(series.dtype)
        ):
            continue

        text_values = (
            series.dropna()
            .astype(str)
            .str.strip()
        )
        text_values = text_values.loc[
            text_values.ne("")
        ]
        text_count = int(text_values.shape[0])

        if text_count == 0:
            continue

        numeric_values = pd.to_numeric(
            text_values,
            errors="coerce",
        )
        numeric_success_count = int(
            numeric_values.notna().sum()
        )
        numeric_success_percentage = _percentage(
            numeric_success_count,
            text_count,
        )

        if (
            numeric_success_count > 0
            and numeric_success_percentage
            >= active_config.numeric_like_text_min_parse_pct
        ):
            numeric_invalid_mask = numeric_values.isna()
            numeric_invalid_values = (
                text_values.loc[numeric_invalid_mask]
                .drop_duplicates()
                .head(max_example_values)
                .tolist()
            )
            numeric_invalid_count = int(
                numeric_invalid_mask.sum()
            )

            records.append(
                {
                    "column": column_name,
                    "dtype": str(series.dtype),
                    "status": "NUMERIC_LIKE_TEXT",
                    "severity": (
                        "MEDIUM"
                        if numeric_invalid_count
                        else "LOW"
                    ),
                    "inferred_type": "NUMERIC",
                    "non_null_count": text_count,
                    "parse_success_count": (
                        numeric_success_count
                    ),
                    "parse_success_percentage": (
                        numeric_success_percentage
                    ),
                    "invalid_count": numeric_invalid_count,
                    "invalid_percentage": _percentage(
                        numeric_invalid_count,
                        text_count,
                    ),
                    "detected_format_count": None,
                    "detected_formats": [],
                    "example_invalid_values": [
                        _safe_scalar(value)
                        for value in numeric_invalid_values
                    ],
                    "recommendation": (
                        "Validate non-numeric tokens, then convert the "
                        "column to a numeric dtype before numerical "
                        "analysis."
                    ),
                }
            )

        format_labels = text_values.map(
            _date_format_label
        )
        date_success_mask = format_labels.notna()
        date_success_count = int(
            date_success_mask.sum()
        )
        date_success_percentage = _percentage(
            date_success_count,
            text_count,
        )
        detected_formats = sorted(
            {
                str(label)
                for label in format_labels.dropna().tolist()
            }
        )

        datetime_candidate = (
            _looks_like_datetime_name(column_name)
            or date_success_percentage
            >= active_config.datetime_like_text_min_parse_pct
        )

        if (
            datetime_candidate
            and date_success_count > 0
            and date_success_percentage
            >= active_config.datetime_like_text_min_parse_pct
        ):
            date_invalid_mask = ~date_success_mask
            date_invalid_values = (
                text_values.loc[date_invalid_mask]
                .drop_duplicates()
                .head(max_example_values)
                .tolist()
            )
            date_invalid_count = int(
                date_invalid_mask.sum()
            )
            mixed_formats = (
                len(detected_formats)
                >= active_config.datetime_mixed_format_min_count
            )

            if mixed_formats:
                status = "MIXED_DATETIME_FORMATS"
                severity = "HIGH"
                recommendation = (
                    "Standardize the accepted date format and parse "
                    "the column explicitly. Review invalid values "
                    "before conversion."
                )
            else:
                status = "DATETIME_LIKE_TEXT"
                severity = (
                    "MEDIUM"
                    if date_invalid_count
                    else "LOW"
                )
                recommendation = (
                    "Parse the column to a datetime dtype using an "
                    "explicit format after reviewing invalid values."
                )

            records.append(
                {
                    "column": column_name,
                    "dtype": str(series.dtype),
                    "status": status,
                    "severity": severity,
                    "inferred_type": "DATETIME",
                    "non_null_count": text_count,
                    "parse_success_count": date_success_count,
                    "parse_success_percentage": (
                        date_success_percentage
                    ),
                    "invalid_count": date_invalid_count,
                    "invalid_percentage": _percentage(
                        date_invalid_count,
                        text_count,
                    ),
                    "detected_format_count": len(
                        detected_formats
                    ),
                    "detected_formats": detected_formats,
                    "example_invalid_values": [
                        _safe_scalar(value)
                        for value in date_invalid_values
                    ],
                    "recommendation": recommendation,
                }
            )

    columns = [
        "column",
        "dtype",
        "status",
        "severity",
        "inferred_type",
        "non_null_count",
        "parse_success_count",
        "parse_success_percentage",
        "invalid_count",
        "invalid_percentage",
        "detected_format_count",
        "detected_formats",
        "example_invalid_values",
        "recommendation",
    ]

    report = pd.DataFrame(records, columns=columns)

    if not report.empty:
        severity_rank = {
            "CRITICAL": 0,
            "HIGH": 1,
            "MEDIUM": 2,
            "LOW": 3,
        }

        report = (
            report.assign(
                _severity_rank=report["severity"].map(
                    severity_rank
                )
            )
            .sort_values(
                by=[
                    "_severity_rank",
                    "column",
                    "status",
                ],
                kind="stable",
            )
            .drop(columns="_severity_rank")
            .reset_index(drop=True)
        )

    return report

def run_basic_quality_checks(
    dataframe: pd.DataFrame,
    *,
    config: QualityCheckConfig | None = None,
) -> dict[str, Any]:
    """Run the first four deterministic quality checks."""

    active_config = config or QualityCheckConfig()

    return {
        "overview": get_dataset_overview(dataframe),
        "missing_values": analyze_missing_values(
            dataframe,
            config=active_config,
        ),
        "duplicates": analyze_duplicates(
            dataframe,
            config=active_config,
        ),
        "constant_columns": (
            analyze_constant_columns(dataframe)
        ),
    }


def run_advanced_quality_checks(
    dataframe: pd.DataFrame,
    *,
    config: QualityCheckConfig | None = None,
) -> dict[str, pd.DataFrame]:
    """Run advanced deterministic and schema quality checks."""

    active_config = config or QualityCheckConfig()

    return {
        "near_constant_columns": (
            analyze_near_constant_columns(
                dataframe,
                config=active_config,
            )
        ),
        "potential_identifiers": (
            analyze_potential_identifiers(
                dataframe,
                config=active_config,
            )
        ),
        "high_cardinality_categories": (
            analyze_high_cardinality_categories(
                dataframe,
                config=active_config,
            )
        ),
        "numeric_outliers": analyze_numeric_outliers(
            dataframe,
            config=active_config,
        ),
        "category_consistency": (
            analyze_category_consistency(
                dataframe,
                config=active_config,
            )
        ),
        "data_type_warnings": (
            analyze_data_type_warnings(
                dataframe,
                config=active_config,
            )
        ),
    }


def run_all_quality_checks(
    dataframe: pd.DataFrame,
    *,
    config: QualityCheckConfig | None = None,
) -> dict[str, Any]:
    """Run all currently implemented quality checks."""

    active_config = config or QualityCheckConfig()

    return {
        **run_basic_quality_checks(
            dataframe,
            config=active_config,
        ),
        **run_advanced_quality_checks(
            dataframe,
            config=active_config,
        ),
    }
