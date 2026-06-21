# =============================================================
# TESTS — RESPONSE FORMATTER
# =============================================================

from src.response_formatter import cleanup_beginner_response, format_copilot_response


def test_cleanup_beginner_response_replaces_internal_status_values() -> None:
    raw = (
        "Score Band Status: READY_WITH_MINOR_REVIEW\n"
        "Final Readiness Status: NEEDS_CLEANING\n"
        "Readiness Gates: High-severity issues prevent readiness."
    )

    cleaned = cleanup_beginner_response(raw)

    assert "READY_WITH_MINOR_REVIEW" not in cleaned
    assert "NEEDS_CLEANING" not in cleaned
    assert "Score Band Status" not in cleaned
    assert "Final Readiness Status" not in cleaned
    assert "Readiness Gates" not in cleaned
    assert "High-severity issues" not in cleaned

    assert "Ready with Minor Review" in cleaned
    assert "Needs Cleaning" in cleaned
    assert "Score Category" in cleaned
    assert "Final Status" in cleaned
    assert "Main Reason" in cleaned
    assert "Serious Issues" in cleaned


def test_cleanup_beginner_response_replaces_lowercase_audit_terms() -> None:
    raw = (
        "The dataset has 14 high-severity issues. "
        "It is not ready for downstream analysis without targeted cleaning."
    )

    cleaned = cleanup_beginner_response(raw)

    assert "high-severity issues" not in cleaned
    assert "downstream analysis" not in cleaned
    assert "targeted cleaning" not in cleaned

    assert "14 serious issues" in cleaned
    assert "analysis" in cleaned
    assert "focused cleaning" in cleaned


def test_format_copilot_response_only_cleans_beginner_style() -> None:
    raw = "Final Readiness Status: NEEDS_CLEANING due to high-severity issues."

    beginner = format_copilot_response(raw, explanation_style="beginner-friendly")
    business = format_copilot_response(raw, explanation_style="business-friendly")

    assert "NEEDS_CLEANING" not in beginner
    assert "high-severity issues" not in beginner
    assert "Needs Cleaning" in beginner
    assert "serious issues" in beginner

    assert business == raw