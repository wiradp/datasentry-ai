from __future__ import annotations

import csv
import hashlib
import io
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


class CSVValidationError(ValueError):
    """Error yang muncul ketika file CSV gagal divalidasi."""

    def __init__(self, message: str, *, code: str = "invalid_csv") -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class CSVLoadConfig:
    """Batas validasi file CSV."""

    max_file_size_mb: int = 20
    max_rows: int = 100_000
    max_columns: int = 500

    allowed_delimiters: tuple[str, ...] = (
        ",",
        ";",
        "\t",
        "|",
    )

    supported_encodings: tuple[str, ...] = (
        "utf-8-sig",
        "cp1252",
        "latin-1",
    )

    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024


@dataclass
class CSVLoadResult:
    """Hasil pemuatan CSV yang telah tervalidasi."""

    dataframe: pd.DataFrame
    file_name: str
    file_size_bytes: int
    encoding: str
    delimiter: str
    fingerprint_sha256: str
    warnings: list[str] = field(default_factory=list)

    @property
    def row_count(self) -> int:
        return int(self.dataframe.shape[0])

    @property
    def column_count(self) -> int:
        return int(self.dataframe.shape[1])


def _resolve_file_name(
    source: Any,
    file_name: str | None,
) -> str:
    """Mendapatkan nama file tanpa path directory."""

    if file_name:
        return Path(file_name).name

    source_name = getattr(source, "name", None)

    if source_name:
        return Path(str(source_name)).name

    if isinstance(source, (str, Path)):
        return Path(source).name

    return "uploaded.csv"


def _read_source_bytes(source: Any) -> bytes:
    """
    Membaca source menjadi bytes.

    Source dapat berupa:
    - path string;
    - pathlib.Path;
    - bytes;
    - bytearray;
    - Streamlit UploadedFile;
    - file-like object.
    """

    if isinstance(source, bytes):
        return source

    if isinstance(source, bytearray):
        return bytes(source)

    if isinstance(source, (str, Path)):
        path = Path(source)

        if not path.exists():
            raise CSVValidationError(
                f"File tidak ditemukan: {path}",
                code="file_not_found",
            )

        if not path.is_file():
            raise CSVValidationError(
                f"Path bukan sebuah file: {path}",
                code="not_a_file",
            )

        return path.read_bytes()

    getvalue = getattr(source, "getvalue", None)

    if callable(getvalue):
        data = getvalue()

        if isinstance(data, str):
            return data.encode("utf-8")

        if isinstance(data, (bytes, bytearray)):
            return bytes(data)

    read = getattr(source, "read", None)

    if callable(read):
        original_position = None

        if hasattr(source, "tell"):
            try:
                original_position = source.tell()
            except (OSError, ValueError):
                original_position = None

        data = read()

        if original_position is not None and hasattr(source, "seek"):
            try:
                source.seek(original_position)
            except (OSError, ValueError):
                pass

        if isinstance(data, str):
            return data.encode("utf-8")

        if isinstance(data, (bytes, bytearray)):
            return bytes(data)

    raise CSVValidationError(
        "Sumber file tidak didukung.",
        code="unsupported_source",
    )


def _decode_content(
    raw_bytes: bytes,
    encodings: tuple[str, ...],
) -> tuple[str, str]:
    """Mencoba membaca file menggunakan encoding yang didukung."""

    for encoding in encodings:
        try:
            text = raw_bytes.decode(encoding)
            return text, encoding

        except UnicodeDecodeError:
            continue

    raise CSVValidationError(
        "Encoding file tidak didukung atau isi file rusak.",
        code="unsupported_encoding",
    )


def _detect_delimiter(
    text: str,
    allowed_delimiters: tuple[str, ...],
) -> str:
    """Mendeteksi separator CSV dari beberapa baris awal."""

    non_empty_lines = [
        line
        for line in text.splitlines()
        if line.strip()
    ]

    if not non_empty_lines:
        raise CSVValidationError(
            "File CSV tidak memiliki isi yang dapat dibaca.",
            code="empty_file",
        )

    sample = "\n".join(non_empty_lines[:20])

    try:
        dialect = csv.Sniffer().sniff(
            sample,
            delimiters="".join(allowed_delimiters),
        )

        return dialect.delimiter

    except csv.Error:
        header_line = non_empty_lines[0]

        delimiter_counts = {
            delimiter: header_line.count(delimiter)
            for delimiter in allowed_delimiters
        }

        best_delimiter = max(
            delimiter_counts,
            key=delimiter_counts.get,
        )

        if delimiter_counts[best_delimiter] == 0:
            return ","

        return best_delimiter


def _read_raw_header(
    text: str,
    delimiter: str,
) -> list[str]:
    """Membaca header sebelum Pandas mengubah nama kolom duplikat."""

    non_empty_lines = [
        line
        for line in text.splitlines()
        if line.strip()
    ]

    if not non_empty_lines:
        raise CSVValidationError(
            "File CSV tidak memiliki header.",
            code="missing_header",
        )

    try:
        return next(
            csv.reader(
                [non_empty_lines[0]],
                delimiter=delimiter,
            )
        )

    except (csv.Error, StopIteration) as exc:
        raise CSVValidationError(
            "Header CSV tidak dapat dibaca.",
            code="invalid_header",
        ) from exc


def _validate_header(header: list[str]) -> list[str]:
    """Memeriksa header kosong dan nama kolom duplikat."""

    if not header:
        raise CSVValidationError(
            "File CSV tidak memiliki kolom.",
            code="missing_columns",
        )

    warnings: list[str] = []

    normalized_names = [
        column.strip().casefold()
        for column in header
    ]

    duplicate_names = [
        name
        for name, count in Counter(normalized_names).items()
        if name and count > 1
    ]

    if duplicate_names:
        duplicate_text = ", ".join(
            sorted(duplicate_names)
        )

        raise CSVValidationError(
            f"Nama kolom duplikat terdeteksi: {duplicate_text}",
            code="duplicate_columns",
        )

    blank_headers = sum(
        not column.strip()
        for column in header
    )

    if blank_headers:
        warnings.append(
            f"{blank_headers} nama kolom kosong terdeteksi."
        )

    return warnings


def load_csv(
    source: Any,
    *,
    file_name: str | None = None,
    config: CSVLoadConfig | None = None,
) -> CSVLoadResult:
    """
    Membaca dan memvalidasi file CSV.

    Loader tidak membersihkan atau mengubah nilai dataset.
    """

    active_config = config or CSVLoadConfig()

    resolved_name = _resolve_file_name(
        source,
        file_name,
    )

    if Path(resolved_name).suffix.casefold() != ".csv":
        raise CSVValidationError(
            "Hanya file dengan ekstensi .csv yang didukung.",
            code="invalid_extension",
        )

    raw_bytes = _read_source_bytes(source)
    file_size_bytes = len(raw_bytes)

    if file_size_bytes == 0:
        raise CSVValidationError(
            "File CSV kosong.",
            code="empty_file",
        )

    if file_size_bytes > active_config.max_file_size_bytes:
        size_mb = file_size_bytes / (1024 * 1024)

        raise CSVValidationError(
            (
                f"Ukuran file {size_mb:.2f} MB melebihi "
                f"batas {active_config.max_file_size_mb} MB."
            ),
            code="file_too_large",
        )

    if b"\x00" in raw_bytes:
        raise CSVValidationError(
            (
                "File mengandung null byte dan kemungkinan "
                "bukan file CSV teks yang valid."
            ),
            code="binary_content",
        )

    text, encoding = _decode_content(
        raw_bytes,
        active_config.supported_encodings,
    )

    if not text.strip():
        raise CSVValidationError(
            "File CSV tidak memiliki isi yang dapat dibaca.",
            code="empty_file",
        )

    delimiter = _detect_delimiter(
        text,
        active_config.allowed_delimiters,
    )

    raw_header = _read_raw_header(
        text,
        delimiter,
    )

    warnings = _validate_header(raw_header)

    try:
        dataframe = pd.read_csv(
            io.StringIO(text),
            sep=delimiter,
            nrows=active_config.max_rows + 1,
            on_bad_lines="error",
            low_memory=False,
        )

    except pd.errors.EmptyDataError as exc:
        raise CSVValidationError(
            "File CSV tidak memiliki data.",
            code="empty_data",
        ) from exc

    except pd.errors.ParserError as exc:
        raise CSVValidationError(
            f"Struktur CSV tidak valid: {exc}",
            code="parser_error",
        ) from exc

    except (TypeError, ValueError) as exc:
        raise CSVValidationError(
            f"CSV gagal dibaca: {exc}",
            code="read_error",
        ) from exc

    row_count, column_count = dataframe.shape

    if column_count == 0:
        raise CSVValidationError(
            "CSV tidak memiliki kolom yang dapat digunakan.",
            code="missing_columns",
        )

    if column_count > active_config.max_columns:
        raise CSVValidationError(
            (
                f"Jumlah kolom {column_count} melebihi "
                f"batas {active_config.max_columns}."
            ),
            code="too_many_columns",
        )

    if row_count > active_config.max_rows:
        raise CSVValidationError(
            (
                f"Jumlah baris melebihi batas "
                f"{active_config.max_rows:,}."
            ),
            code="too_many_rows",
        )

    if dataframe.empty:
        raise CSVValidationError(
            "CSV hanya memiliki header tanpa baris data.",
            code="no_data_rows",
        )

    if dataframe.dropna(how="all").empty:
        raise CSVValidationError(
            "Semua baris dalam CSV kosong.",
            code="all_rows_empty",
        )

    if encoding not in {"utf-8", "utf-8-sig"}:
        warnings.append(
            (
                "File dibaca menggunakan encoding fallback "
                f"'{encoding}'."
            )
        )

    unnamed_columns = [
        str(column)
        for column in dataframe.columns
        if str(column)
        .strip()
        .casefold()
        .startswith("unnamed:")
    ]

    if unnamed_columns:
        warnings.append(
            (
                f"{len(unnamed_columns)} kolom "
                "'Unnamed' terdeteksi."
            )
        )

    if column_count == 1:
        warnings.append(
            (
                "CSV hanya memiliki satu kolom. "
                "Periksa delimiter jika ini tidak disengaja."
            )
        )

    all_null_columns = [
        str(column)
        for column in dataframe.columns
        if dataframe[column].isna().all()
    ]

    if all_null_columns:
        warnings.append(
            (
                f"{len(all_null_columns)} kolom "
                "seluruhnya kosong terdeteksi."
            )
        )

    fingerprint = hashlib.sha256(
        raw_bytes
    ).hexdigest()

    return CSVLoadResult(
        dataframe=dataframe,
        file_name=resolved_name,
        file_size_bytes=file_size_bytes,
        encoding=encoding,
        delimiter=delimiter,
        fingerprint_sha256=fingerprint,
        warnings=warnings,
    )