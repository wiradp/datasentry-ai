from copy import deepcopy
from pathlib import Path

import pandas as pd
import pytest

from src.quality_checks import run_all_quality_checks
from src.quality_score import (
    QualityScoreConfig,
    QualityScoreError,
    calculate_dataframe_quality_score,
    calculate_quality_score,
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
                float(100 + (index * 50))
                for index in range(20)
            ],
        }
    )


def test_default_weights_sum_to_one() -> None:
    config = QualityScoreConfig()

    assert sum(config.weights.values()) == pytest.approx(1.0)


def test_rejects_invalid_weight_sum() -> None:
    with pytest.raises(
        ValueError,
        match="sum to 1.0",
    ):
        QualityScoreConfig(
            completeness_weight=0.50,
        )


def test_rejects_unordered_readiness_thresholds() -> None:
    with pytest.raises(
        ValueError,
        match="Readiness thresholds",
    ):
        QualityScoreConfig(
            ready_min_score=60.0,
            needs_cleaning_min_score=70.0,
        )


def test_clean_dataframe_receives_perfect_score(
    clean_dataframe: pd.DataFrame,
) -> None:
    result = calculate_dataframe_quality_score(
        clean_dataframe
    )

    assert result["overall_score"] == 100.0
    assert result["readiness_status"] == (
        "READY_WITH_MINOR_REVIEW"
    )
    assert result["issue_count_total"] == 0
    assert all(
        score == 100.0
        for score in result["component_scores"].values()
    )


def test_sample_dataset_score_is_deterministic(
    sample_dataframe: pd.DataFrame,
) -> None:
    first = calculate_dataframe_quality_score(
        sample_dataframe
    )
    second = calculate_dataframe_quality_score(
        sample_dataframe
    )

    assert first == second
    assert 0.0 <= first["overall_score"] <= 100.0
    assert first["score_band_status"] == (
        "READY_WITH_MINOR_REVIEW"
    )
    assert first["readiness_status"] == "NEEDS_CLEANING"
    assert first["status_adjusted_by_gates"] is True
    assert first["issue_counts"]["HIGH"] > 0


def test_weighted_contributions_reconstruct_score(
    sample_dataframe: pd.DataFrame,
) -> None:
    result = calculate_dataframe_quality_score(
        sample_dataframe
    )

    assert sum(
        result["weighted_contributions"].values()
    ) == pytest.approx(
        result["overall_score"],
        abs=0.01,
    )


def test_component_scores_stay_in_valid_range(
    sample_dataframe: pd.DataFrame,
) -> None:
    result = calculate_dataframe_quality_score(
        sample_dataframe
    )

    for score in result["component_scores"].values():
        assert 0.0 <= score <= 100.0


def test_score_accepts_precomputed_checks(
    sample_dataframe: pd.DataFrame,
) -> None:
    checks = run_all_quality_checks(sample_dataframe)

    from_checks = calculate_quality_score(checks)
    from_dataframe = calculate_dataframe_quality_score(
        sample_dataframe
    )

    assert from_checks == from_dataframe


def test_scoring_does_not_mutate_check_results(
    sample_dataframe: pd.DataFrame,
) -> None:
    checks = run_all_quality_checks(sample_dataframe)
    original = deepcopy(checks)

    calculate_quality_score(checks)

    assert checks["overview"] == original["overview"]
    assert checks["duplicates"] == original["duplicates"]

    for key, report in checks.items():
        if isinstance(report, pd.DataFrame):
            pd.testing.assert_frame_equal(
                report,
                original[key],
            )


def test_missing_check_result_is_rejected(
    sample_dataframe: pd.DataFrame,
) -> None:
    checks = run_all_quality_checks(sample_dataframe)
    checks.pop("data_type_warnings")

    with pytest.raises(
        QualityScoreError,
        match="Missing quality-check results",
    ):
        calculate_quality_score(checks)


def test_critical_issue_forces_not_ready(
    clean_dataframe: pd.DataFrame,
) -> None:
    checks = run_all_quality_checks(clean_dataframe)

    checks["constant_columns"] = pd.DataFrame(
        [
            {
                "column": "broken_column",
                "severity": "CRITICAL",
            }
        ]
    )

    result = calculate_quality_score(checks)

    assert result["readiness_status"] == "NOT_READY"
    assert result["status_adjusted_by_gates"] is True
    assert result["issue_counts"]["CRITICAL"] == 1


def test_high_gate_can_be_disabled(
    sample_dataframe: pd.DataFrame,
) -> None:
    result = calculate_dataframe_quality_score(
        sample_dataframe,
        score_config=QualityScoreConfig(
            apply_readiness_gates=False
        ),
    )

    assert result["readiness_status"] == (
        result["score_band_status"]
    )
    assert result["readiness_gates"] == []


def test_numeric_only_dataset_has_full_category_score() -> None:
    dataframe = pd.DataFrame(
        {
            "feature_a": [1, 2, 3, 4],
            "feature_b": [10.0, 20.0, 30.0, 40.0],
        }
    )

    result = calculate_dataframe_quality_score(dataframe)

    assert result["component_scores"][
        "categorical_consistency"
    ] == 100.0


def test_categorical_only_dataset_has_full_outlier_score() -> None:
    dataframe = pd.DataFrame(
        {
            "city": ["A", "B", "C", "D"],
            "segment": ["one", "two", "three", "four"],
        }
    )

    result = calculate_dataframe_quality_score(dataframe)

    assert result["component_scores"][
        "outlier_risk"
    ] == 100.0


def test_result_exposes_heuristic_disclaimer(
    clean_dataframe: pd.DataFrame,
) -> None:
    result = calculate_dataframe_quality_score(
        clean_dataframe
    )

    assert result["methodology"]["type"] == "HEURISTIC"
    assert "heuristic" in result["disclaimer"].lower()
    assert "human review" in result["disclaimer"].lower()
