# =============================================================
# RESPONSE FORMATTER
# =============================================================
"""Response formatting helpers for user-facing Copilot answers."""

from __future__ import annotations

import re
from typing import Any


_BEGINNER_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    (r"\bREADY_WITH_MINOR_REVIEW\b", "Ready with Minor Review"),
    (r"\bNEEDS_CLEANING\b", "Needs Cleaning"),
    (r"\bscore band status\b", "score category"),
    (r"\bScore Band Status\b", "Score Category"),
    (r"\bfinal readiness status\b", "final status"),
    (r"\bFinal Readiness Status\b", "Final Status"),
    (r"\breadiness status\b", "status"),
    (r"\bReadiness Status\b", "Status"),
    (r"\breadiness gates\b", "main reason"),
    (r"\bReadiness Gates\b", "Main Reason"),
    (r"\bhigh-severity issues\b", "serious issues"),
    (r"\bHigh-Severity Issues\b", "Serious Issues"),
    (r"\bhigh severity issues\b", "serious issues"),
    (r"\bHigh Severity Issues\b", "Serious Issues"),
    (r"\bhigh-severity issue\b", "serious issue"),
    (r"\bHigh-Severity Issue\b", "Serious Issue"),
    (r"\bHigh-severity issues\b", "Serious Issues"),
    (r"\bHigh-severity issue\b", "Serious Issue"),
    (r"\bhighest severity issue\b", "most serious issue"),
    (r"\bHighest Severity Issue\b", "Most Serious Issue"),
    (r"\bdownstream analysis\b", "analysis"),
    (r"\bDownstream Analysis\b", "Analysis"),
    (r"\btargeted cleaning\b", "focused cleaning"),
    (r"\bTargeted Cleaning\b", "Focused Cleaning"),
)


def _normalize_style(explanation_style: Any) -> str:
    """Normalize style values from strings or enums into a simple comparable form."""
    raw_value = getattr(explanation_style, "value", explanation_style)
    return str(raw_value).strip().lower().replace("_", "-").replace(" ", "-")


def cleanup_beginner_response(text: str) -> str:
    """Convert technical audit wording into more beginner-friendly wording.

    This function only changes presentation wording. It must not change the
    underlying audit facts, scores, counts, or recommendations.
    """
    cleaned = text

    for pattern, replacement in _BEGINNER_REPLACEMENTS:
        cleaned = re.sub(pattern, replacement, cleaned)

    return cleaned


def format_copilot_response(text: str, explanation_style: Any) -> str:
    """Apply presentation formatting based on the selected explanation style."""
    normalized_style = _normalize_style(explanation_style)

    if normalized_style in {"beginner", "beginner-friendly"}:
        return cleanup_beginner_response(text)

    return text