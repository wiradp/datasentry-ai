
from pathlib import Path

import pandas as pd
import pytest

from src.quality_checks import (
    QualityCheckError,
    analyze_constant_columns,
    analyze_duplicates,
    analyze_missing_values,
    get_dataset_overview,
    run_basic_quality_checks,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SAMPLE_PATH = (
    PROJECT_ROOT
    / "data"
    / "sample_dirty_customers.csv"
)


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_PATH)


def test_dataset_overview_matches_sample(
    sample_dataframe: pd.DataFrame,
) -> None:
    overview = get_dataset_overview(
        sample_dataframe
    )

    assert overview["row_count"] == 208
    assert overview["column_count"] == 13
    assert overview["total_cells"] == 2704
    assert overview["missing_cells"] == 66
    assert overview["missing_percentage"] == 2.44
    assert overview["duplicate_rows"] == 8
    assert overview["duplicate_percentage"] == 3.85
    assert overview["numeric_column_count"] == 4
    assert overview["categorical_column_count"] == 9
    assert overview["datetime_column_count"] == 0
    assert overview["memory_usage_bytes"] > 0


def test_missing_value_report_matches_sample(
    sample_dataframe: pd.DataFrame,
) -> None:
    report = analyze_missing_values(
        sample_dataframe,
        include_complete=False,
    )

    assert len(report) == 6
    assert report.iloc[0]["column"] == (
        "annual_income"
    )
    assert report.iloc[0]["missing_count"] == 25
    assert report.iloc[0]["severity"] == "MEDIUM"

    age_row = report.loc[
        report["column"] == "age"
    ].iloc[0]

    assert age_row["missing_count"] == 17
    assert age_row["missing_percentage"] == 8.17
    assert age_row["severity"] == "MEDIUM"


def test_missing_value_severity_boundaries() -> None:
    dataframe = pd.DataFrame(
        {
            "complete": [1] * 100,
            "low": [1] * 95 + [None] * 5,
            "medium": [1] * 80 + [None] * 20,
            "high": [1] * 60 + [None] * 40,
            "critical": [1] * 59 + [None] * 41,
        }
    )

    report = analyze_missing_values(dataframe)
    severity_by_column = dict(
        zip(
            report["column"],
            report["severity"],
            strict=True,
        )
    )

    assert severity_by_column == {
        "complete": "PASS",
        "low": "LOW",
        "medium": "MEDIUM",
        "high": "HIGH",
        "critical": "CRITICAL",
    }


def test_duplicate_report_matches_sample(
    sample_dataframe: pd.DataFrame,
) -> None:
    report = analyze_duplicates(
        sample_dataframe
    )

    assert report["duplicate_rows"] == 8
    assert report["duplicate_percentage"] == 3.85
    assert report["rows_in_duplicate_groups"] == 16
    assert report["duplicate_group_count"] == 8
    assert report["severity"] == "MEDIUM"
    assert report["status"] == "ISSUE"
    assert len(
        report["example_duplicate_indices"]
    ) == 8


def test_duplicate_report_passes_clean_data() -> None:
    dataframe = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "value": ["a", "b", "c"],
        }
    )

    report = analyze_duplicates(dataframe)

    assert report["duplicate_rows"] == 0
    assert report["severity"] == "PASS"
    assert report["status"] == "PASS"


def test_constant_column_report_matches_sample(
    sample_dataframe: pd.DataFrame,
) -> None:
    report = analyze_constant_columns(
        sample_dataframe
    )

    assert report["column"].tolist() == [
        "country"
    ]
    assert report.iloc[0]["status"] == "CONSTANT"
    assert report.iloc[0]["severity"] == "HIGH"
    assert (
        report.iloc[0]["constant_value"]
        == "Indonesia"
    )


def test_constant_and_all_missing_columns() -> None:
    dataframe = pd.DataFrame(
        {
            "all_missing": [None, None, None],
            "constant": ["x", "x", "x"],
            "variable": [1, 2, 3],
        }
    )

    report = analyze_constant_columns(dataframe)

    result = report.set_index("column")

    assert (
        result.loc["all_missing", "status"]
        == "ALL_MISSING"
    )
    assert (
        result.loc["all_missing", "severity"]
        == "CRITICAL"
    )
    assert (
        result.loc["constant", "status"]
        == "CONSTANT"
    )
    assert "variable" not in result.index


def test_basic_quality_check_bundle(
    sample_dataframe: pd.DataFrame,
) -> None:
    results = run_basic_quality_checks(
        sample_dataframe
    )

    assert set(results) == {
        "overview",
        "missing_values",
        "duplicates",
        "constant_columns",
    }


def test_rejects_empty_dataframe() -> None:
    dataframe = pd.DataFrame(
        columns=["a", "b"]
    )

    with pytest.raises(
        QualityCheckError,
        match="data rows",
    ):
        get_dataset_overview(dataframe)
