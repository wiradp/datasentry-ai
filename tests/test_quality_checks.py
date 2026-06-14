from pathlib import Path

import pandas as pd
import pytest

from src.quality_checks import (
    QualityCheckConfig,
    QualityCheckError,
    analyze_category_consistency,
    analyze_constant_columns,
    analyze_data_type_warnings,
    analyze_duplicates,
    analyze_high_cardinality_categories,
    analyze_missing_values,
    analyze_near_constant_columns,
    analyze_numeric_outliers,
    analyze_potential_identifiers,
    get_dataset_overview,
    run_advanced_quality_checks,
    run_all_quality_checks,
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


def test_near_constant_report_matches_sample(
    sample_dataframe: pd.DataFrame,
) -> None:
    report = analyze_near_constant_columns(
        sample_dataframe
    )

    assert report["column"].tolist() == [
        "account_status"
    ]
    assert report.iloc[0]["dominant_value"] == "Active"
    assert report.iloc[0]["dominant_count"] == 205
    assert report.iloc[0]["dominance_percentage"] == 98.56
    assert report.iloc[0]["severity"] == "MEDIUM"


def test_near_constant_excludes_constant_columns() -> None:
    dataframe = pd.DataFrame(
        {
            "constant": ["x"] * 100,
            "near_constant": ["x"] * 99 + ["y"],
            "variable": ["x"] * 50 + ["y"] * 50,
        }
    )

    report = analyze_near_constant_columns(dataframe)

    assert report["column"].tolist() == [
        "near_constant"
    ]
    assert report.iloc[0]["severity"] == "HIGH"


def test_potential_identifier_report_matches_sample(
    sample_dataframe: pd.DataFrame,
) -> None:
    report = analyze_potential_identifiers(
        sample_dataframe
    )

    assert set(report["column"]) == {
        "Unnamed: 0",
        "customer_id",
        "referral_code",
    }

    result = report.set_index("column")

    assert result.loc[
        "customer_id", "unique_percentage"
    ] == 96.15
    assert bool(result.loc["customer_id", "name_hint"])
    assert result.loc[
        "referral_code", "unique_percentage"
    ] == 95.85


def test_high_uniqueness_numeric_measure_is_not_identifier() -> None:
    dataframe = pd.DataFrame(
        {
            "measurement": list(range(100)),
            "record_id": list(range(100)),
        }
    )

    report = analyze_potential_identifiers(dataframe)

    assert report["column"].tolist() == [
        "record_id"
    ]


def test_high_cardinality_report_matches_sample(
    sample_dataframe: pd.DataFrame,
) -> None:
    report = analyze_high_cardinality_categories(
        sample_dataframe
    )

    assert set(report["column"]) == {
        "customer_id",
        "signup_date",
        "referral_code",
    }

    result = report.set_index("column")

    assert result.loc[
        "customer_id", "unique_non_null"
    ] == 200
    assert result.loc[
        "signup_date", "unique_percentage"
    ] == 90.29
    assert result.loc[
        "referral_code", "unique_non_null"
    ] == 185


def test_high_cardinality_thresholds_are_configurable() -> None:
    dataframe = pd.DataFrame(
        {
            "category": [
                f"group-{index % 10}"
                for index in range(100)
            ]
        }
    )

    default_report = (
        analyze_high_cardinality_categories(
            dataframe
        )
    )
    assert default_report.empty

    custom_report = (
        analyze_high_cardinality_categories(
            dataframe,
            config=QualityCheckConfig(
                high_cardinality_min_unique_count=10,
                high_cardinality_min_unique_pct=10.0,
            ),
        )
    )
    assert custom_report["column"].tolist() == [
        "category"
    ]


def test_numeric_outlier_report_matches_sample(
    sample_dataframe: pd.DataFrame,
) -> None:
    report = analyze_numeric_outliers(
        sample_dataframe
    )

    assert set(report["column"]) == {
        "age",
        "annual_income",
        "purchase_amount",
    }

    result = report.set_index("column")

    assert result.loc["age", "outlier_count"] == 1
    assert result.loc["age", "maximum_outlier"] == 150.0
    assert result.loc[
        "annual_income", "maximum_outlier"
    ] == 5_000_000.0
    assert result.loc[
        "purchase_amount", "maximum_outlier"
    ] == 250_000.0
    assert result.loc[
        "annual_income", "method"
    ] == "IQR"


def test_numeric_outlier_check_can_include_clean_columns() -> None:
    dataframe = pd.DataFrame(
        {
            "clean": [1, 2, 3, 4, 5],
            "label": ["a", "b", "c", "d", "e"],
        }
    )

    issue_only = analyze_numeric_outliers(dataframe)
    assert issue_only.empty

    full_report = analyze_numeric_outliers(
        dataframe,
        include_clean=True,
    )
    assert full_report["column"].tolist() == ["clean"]
    assert full_report.iloc[0]["severity"] == "PASS"


def test_advanced_quality_check_bundle(
    sample_dataframe: pd.DataFrame,
) -> None:
    results = run_advanced_quality_checks(
        sample_dataframe
    )

    assert set(results) == {
        "near_constant_columns",
        "potential_identifiers",
        "high_cardinality_categories",
        "numeric_outliers",
        "category_consistency",
        "data_type_warnings",
    }


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


def test_all_quality_check_bundle(
    sample_dataframe: pd.DataFrame,
) -> None:
    results = run_all_quality_checks(
        sample_dataframe
    )

    assert set(results) == {
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


def test_rejects_empty_dataframe() -> None:
    dataframe = pd.DataFrame(
        columns=["a", "b"]
    )

    with pytest.raises(
        QualityCheckError,
        match="data rows",
    ):
        get_dataset_overview(dataframe)



def test_category_consistency_matches_sample(
    sample_dataframe: pd.DataFrame,
) -> None:
    report = analyze_category_consistency(
        sample_dataframe
    )

    assert len(report) == 6
    assert set(report["column"]) == {
        "city",
        "membership_type",
    }

    jakarta = report.loc[
        (report["column"] == "city")
        & (report["normalized_value"] == "jakarta")
    ].iloc[0]

    assert jakarta["variant_count"] == 4
    assert jakarta["affected_count"] == 72
    assert jakarta["affected_percentage"] == 34.62
    assert bool(jakarta["whitespace_variant_detected"])
    assert bool(jakarta["case_variant_detected"])
    assert set(jakarta["variants"]) == {
        "Jakarta",
        "jakarta",
        " Jakarta ",
        "JAKARTA",
    }

    gold = report.loc[
        (report["column"] == "membership_type")
        & (report["normalized_value"] == "gold")
    ].iloc[0]

    assert gold["variant_count"] == 2
    assert gold["affected_count"] == 66
    assert gold["severity"] == "HIGH"


def test_category_consistency_ignores_clean_categories() -> None:
    dataframe = pd.DataFrame(
        {
            "category": ["A", "B", "C", "A"],
            "value": [1, 2, 3, 4],
        }
    )

    report = analyze_category_consistency(dataframe)

    assert report.empty


def test_category_consistency_detects_whitespace_only() -> None:
    dataframe = pd.DataFrame(
        {
            "category": ["Gold", " Gold ", "Gold"],
        }
    )

    report = analyze_category_consistency(dataframe)

    assert len(report) == 1
    assert bool(
        report.iloc[0]["whitespace_variant_detected"]
    )
    assert not bool(
        report.iloc[0]["case_variant_detected"]
    )


def test_data_type_warnings_match_sample(
    sample_dataframe: pd.DataFrame,
) -> None:
    report = analyze_data_type_warnings(
        sample_dataframe
    )

    result = report.set_index(["column", "status"])

    assert set(result.index) == {
        ("Unnamed: 0", "UNNAMED_COLUMN"),
        ("monthly_visits", "NUMERIC_LIKE_TEXT"),
        ("signup_date", "MIXED_DATETIME_FORMATS"),
    }

    monthly = result.loc[
        ("monthly_visits", "NUMERIC_LIKE_TEXT")
    ]
    assert monthly["parse_success_count"] == 201
    assert monthly["parse_success_percentage"] == 98.05
    assert monthly["invalid_count"] == 4
    assert monthly["example_invalid_values"] == [
        "unknown"
    ]

    signup = result.loc[
        ("signup_date", "MIXED_DATETIME_FORMATS")
    ]
    assert signup["parse_success_count"] == 204
    assert signup["parse_success_percentage"] == 99.03
    assert signup["invalid_count"] == 2
    assert signup["detected_format_count"] == 3
    assert set(signup["detected_formats"]) == {
        "YYYY-MM-DD",
        "SLASH_DATE",
        "MON_DD_YYYY",
    }
    assert signup["example_invalid_values"] == [
        "not_available"
    ]


def test_data_type_warnings_ignore_native_types() -> None:
    dataframe = pd.DataFrame(
        {
            "value": [1, 2, 3],
            "event_date": pd.to_datetime(
                ["2026-01-01", "2026-01-02", "2026-01-03"]
            ),
            "category": ["a", "b", "c"],
        }
    )

    report = analyze_data_type_warnings(dataframe)

    assert report.empty


def test_numeric_like_threshold_is_configurable() -> None:
    dataframe = pd.DataFrame(
        {
            "mixed": ["1", "2", "3", "bad", "bad"],
        }
    )

    default_report = analyze_data_type_warnings(dataframe)
    assert default_report.empty

    custom_report = analyze_data_type_warnings(
        dataframe,
        config=QualityCheckConfig(
            numeric_like_text_min_parse_pct=60.0,
        ),
    )

    assert custom_report["status"].tolist() == [
        "NUMERIC_LIKE_TEXT"
    ]
