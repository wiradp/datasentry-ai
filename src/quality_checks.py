
from __future__ import annotations

from dataclasses import dataclass
from numbers import Number
from typing import Any

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
    """Heuristic thresholds used by the initial quality checks."""

    missing_low_max_pct: float = 5.0
    missing_medium_max_pct: float = 20.0
    missing_high_max_pct: float = 40.0

    duplicate_low_max_pct: float = 1.0
    duplicate_medium_max_pct: float = 5.0
    duplicate_high_max_pct: float = 10.0

    def __post_init__(self) -> None:
        threshold_groups = {
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
        }

        for name, values in threshold_groups.items():
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
