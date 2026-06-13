from io import BytesIO
from pathlib import Path

import pytest

from src.data_loader import (
    CSVLoadConfig,
    CSVValidationError,
    load_csv,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SAMPLE_PATH = (
    PROJECT_ROOT
    / "data"
    / "sample_dirty_customers.csv"
)


def test_load_sample_dirty_customers() -> None:
    result = load_csv(SAMPLE_PATH)

    assert result.row_count == 208
    assert result.column_count == 13
    assert result.file_name == "sample_dirty_customers.csv"
    assert result.delimiter == ","
    assert len(result.fingerprint_sha256) == 64
    assert result.dataframe.duplicated().sum() == 8


def test_detects_unnamed_column_warning() -> None:
    result = load_csv(SAMPLE_PATH)

    assert any(
        "Unnamed" in warning
        for warning in result.warnings
    )


def test_rejects_non_csv_extension() -> None:
    source = BytesIO(
        b"name,age\nAlice,30\n"
    )

    with pytest.raises(
        CSVValidationError,
        match="ekstensi .csv",
    ):
        load_csv(
            source,
            file_name="customers.txt",
        )


def test_rejects_empty_file() -> None:
    source = BytesIO(b"")

    with pytest.raises(
        CSVValidationError,
        match="kosong",
    ):
        load_csv(
            source,
            file_name="empty.csv",
        )


def test_rejects_duplicate_columns() -> None:
    source = BytesIO(
        b"name,name\nAlice,30\n"
    )

    with pytest.raises(
        CSVValidationError,
        match="kolom duplikat",
    ):
        load_csv(
            source,
            file_name="duplicate_columns.csv",
        )


def test_rejects_too_many_rows() -> None:
    source = BytesIO(
        b"name,age\nAlice,30\nBob,25\nCharlie,40\n"
    )

    config = CSVLoadConfig(
        max_rows=2,
    )

    with pytest.raises(
        CSVValidationError,
        match="Jumlah baris",
    ):
        load_csv(
            source,
            file_name="too_many_rows.csv",
            config=config,
        )


def test_supports_semicolon_delimiter() -> None:
    source = BytesIO(
        b"name;age\nAlice;30\nBob;25\n"
    )

    result = load_csv(
        source,
        file_name="semicolon.csv",
    )

    assert result.delimiter == ";"
    assert result.row_count == 2
    assert result.column_count == 2