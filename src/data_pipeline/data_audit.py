from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable


AIR_QUALITY_EXPECTED_COLUMNS = [
    "Date",
    "Time",
    "CO(GT)",
    "PT08.S1(CO)",
    "NMHC(GT)",
    "C6H6(GT)",
    "PT08.S2(NMHC)",
    "NOx(GT)",
    "PT08.S3(NOx)",
    "NO2(GT)",
    "PT08.S4(NO2)",
    "PT08.S5(O3)",
    "T",
    "RH",
    "AH",
]

BEIJING_EXPECTED_COLUMNS = [
    "No",
    "year",
    "month",
    "day",
    "hour",
    "PM2.5",
    "PM10",
    "SO2",
    "NO2",
    "CO",
    "O3",
    "TEMP",
    "PRES",
    "DEWP",
    "RAIN",
    "wd",
    "WSPM",
    "station",
]

AIR_QUALITY_RANGE_RULES = {
    "CO(GT)": (0.0, 50.0),
    "PT08.S1(CO)": (0.0, 4000.0),
    "NMHC(GT)": (0.0, 2000.0),
    "C6H6(GT)": (0.0, 100.0),
    "PT08.S2(NMHC)": (0.0, 3000.0),
    "NOx(GT)": (0.0, 2000.0),
    "PT08.S3(NOx)": (0.0, 4000.0),
    "NO2(GT)": (0.0, 1000.0),
    "PT08.S4(NO2)": (0.0, 4000.0),
    "PT08.S5(O3)": (0.0, 4000.0),
    "T": (-40.0, 60.0),
    "RH": (0.0, 100.0),
    "AH": (0.0, 5.0),
}

BEIJING_RANGE_RULES = {
    "PM2.5": (0.0, 2000.0),
    "PM10": (0.0, 2500.0),
    "SO2": (0.0, 1000.0),
    "NO2": (0.0, 1000.0),
    "CO": (0.0, 20000.0),
    "O3": (0.0, 1200.0),
    "TEMP": (-50.0, 60.0),
    "PRES": (850.0, 1100.0),
    "DEWP": (-60.0, 50.0),
    "RAIN": (0.0, 500.0),
    "WSPM": (0.0, 60.0),
}

AIR_QUALITY_KEY_COLUMNS = ["CO(GT)", "NMHC(GT)", "C6H6(GT)", "NOx(GT)", "NO2(GT)"]
BEIJING_KEY_COLUMNS = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]

KNOWN_HIGH_MISSING_COLUMNS = {
    "UCI Air Quality": {"NMHC(GT)"},
    "UCI Beijing Multi-Site Air Quality": set(),
}


@dataclass
class NumericSummary:
    min_value: float | None = None
    max_value: float | None = None
    valid_count: int = 0
    out_of_range_count: int = 0

    def update(self, value: float, min_allowed: float, max_allowed: float) -> None:
        if self.min_value is None or value < self.min_value:
            self.min_value = value
        if self.max_value is None or value > self.max_value:
            self.max_value = value
        self.valid_count += 1
        if value < min_allowed or value > max_allowed:
            self.out_of_range_count += 1


@dataclass
class DatasetAuditResult:
    name: str
    total_rows: int = 0
    columns: list[str] = field(default_factory=list)
    surprise_columns: list[str] = field(default_factory=list)
    missing_counts: Counter[str] = field(default_factory=Counter)
    numeric_summaries: dict[str, NumericSummary] = field(default_factory=dict)
    duplicate_timestamps: int = 0
    invalid_timestamps: int = 0
    out_of_order_transitions: int = 0
    continuity_gaps: int = 0
    missing_hours: int = 0
    max_missing_run: dict[str, int] = field(default_factory=dict)
    all_key_missing_rows: int = 0
    station_issues: list[str] = field(default_factory=list)


def _clean_token(value: str) -> str:
    return value.strip().strip('"').strip("'")


def _is_missing(value: str, missing_tokens: set[str]) -> bool:
    return _clean_token(value) in missing_tokens


def _parse_float(value: str, decimal_comma: bool = False) -> float | None:
    cleaned = _clean_token(value)
    if cleaned == "":
        return None
    if decimal_comma:
        cleaned = cleaned.replace(",", ".")
    try:
        return float(cleaned)
    except ValueError:
        return None


def _parse_float_with_missing(
    value: str,
    missing_tokens: set[str],
    decimal_comma: bool = False,
) -> float | None:
    cleaned = _clean_token(value)
    if cleaned in missing_tokens:
        return None
    if decimal_comma:
        cleaned = cleaned.replace(",", ".")
    try:
        return float(cleaned)
    except ValueError:
        return None


def _compute_continuity_metrics(timestamps: Iterable[datetime]) -> tuple[int, int]:
    unique_timestamps = sorted(set(timestamps))
    if len(unique_timestamps) < 2:
        return 0, 0

    one_hour = timedelta(hours=1)
    continuity_gaps = 0
    missing_hours = 0

    for previous, current in zip(unique_timestamps, unique_timestamps[1:]):
        delta = current - previous
        if delta != one_hour:
            continuity_gaps += 1
            if delta > one_hour:
                missing_hours += max(0, int(delta.total_seconds() // 3600) - 1)

    return continuity_gaps, missing_hours


def _initialize_missing_runs(key_columns: list[str]) -> tuple[dict[str, int], dict[str, int]]:
    current_runs = {column: 0 for column in key_columns}
    max_runs = {column: 0 for column in key_columns}
    return current_runs, max_runs


def _update_missing_runs(
    row_values: dict[str, str],
    key_columns: list[str],
    missing_tokens: set[str],
    current_runs: dict[str, int],
    max_runs: dict[str, int],
) -> bool:
    all_missing = True

    for column in key_columns:
        value = row_values.get(column, "")
        if _is_missing(value, missing_tokens):
            current_runs[column] += 1
            if current_runs[column] > max_runs[column]:
                max_runs[column] = current_runs[column]
        else:
            current_runs[column] = 0
            all_missing = False

    return all_missing


def _update_numeric_summaries(
    row_values: dict[str, str],
    range_rules: dict[str, tuple[float, float]],
    numeric_summaries: dict[str, NumericSummary],
    missing_tokens: set[str],
    decimal_comma: bool,
) -> None:
    for column, (min_allowed, max_allowed) in range_rules.items():
        numeric_value = _parse_float_with_missing(
            row_values.get(column, ""),
            missing_tokens=missing_tokens,
            decimal_comma=decimal_comma,
        )
        if numeric_value is None:
            continue
        summary = numeric_summaries.setdefault(column, NumericSummary())
        summary.update(numeric_value, min_allowed=min_allowed, max_allowed=max_allowed)


def _build_missing_summary_lines(
    result: DatasetAuditResult,
    top_k: int,
) -> list[str]:
    summary_lines: list[str] = []
    if result.total_rows == 0:
        return ["- No rows available for missing-value summary."]

    sorted_columns = sorted(
        result.columns,
        key=lambda column: (
            result.missing_counts.get(column, 0) / result.total_rows,
            result.missing_counts.get(column, 0),
            column,
        ),
        reverse=True,
    )

    for column in sorted_columns[:top_k]:
        missing_count = result.missing_counts.get(column, 0)
        missing_ratio = (missing_count / result.total_rows) * 100
        summary_lines.append(
            f"- {column}: {missing_count} missing ({missing_ratio:.2f}%)"
        )

    return summary_lines


def _build_range_summary_lines(
    range_rules: dict[str, tuple[float, float]],
    numeric_summaries: dict[str, NumericSummary],
) -> list[str]:
    lines: list[str] = []

    for column in range_rules:
        if column not in numeric_summaries:
            lines.append(f"- {column}: no valid numeric values found")
            continue

        stats = numeric_summaries[column]
        lines.append(
            f"- {column}: min={stats.min_value:.3f}, max={stats.max_value:.3f}, "
            f"out_of_range={stats.out_of_range_count}"
        )

    return lines


def _evaluate_missing_patterns(result: DatasetAuditResult) -> list[str]:
    reasons: list[str] = []
    known_high_missing = KNOWN_HIGH_MISSING_COLUMNS.get(result.name, set())

    if result.total_rows == 0:
        reasons.append("No rows available, missing-pattern analysis is inconclusive")
        return reasons

    for column in result.columns:
        missing_ratio = result.missing_counts.get(column, 0) / result.total_rows
        if missing_ratio > 0.5 and column not in known_high_missing:
            reasons.append(
                f"{column} has high missingness ({missing_ratio * 100:.2f}%)"
            )

    if result.invalid_timestamps > 0:
        reasons.append(f"Invalid timestamps detected ({result.invalid_timestamps} rows)")

    return reasons


def audit_air_quality(air_quality_path: Path) -> DatasetAuditResult:
    result = DatasetAuditResult(name="UCI Air Quality")

    if not air_quality_path.exists():
        raise FileNotFoundError(f"Missing file: {air_quality_path}")

    missing_tokens = {"", "-200", "-200,0", "-200.0"}
    timestamp_values: list[datetime] = []
    seen_timestamps: set[datetime] = set()
    previous_timestamp: datetime | None = None
    current_runs, max_runs = _initialize_missing_runs(AIR_QUALITY_KEY_COLUMNS)

    with air_quality_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.reader(csv_file, delimiter=";")
        raw_header = next(reader)
        header = [_clean_token(column) for column in raw_header if _clean_token(column)]

        result.columns = header
        result.surprise_columns = [
            column for column in header if column not in AIR_QUALITY_EXPECTED_COLUMNS
        ]

        for row in reader:
            if not row:
                continue

            row_values = list(row)
            if len(row_values) < len(header):
                row_values.extend([""] * (len(header) - len(row_values)))
            row_values = row_values[: len(header)]
            row_dict = dict(zip(header, row_values))

            date_missing = _is_missing(row_dict.get("Date", ""), missing_tokens)
            time_missing = _is_missing(row_dict.get("Time", ""), missing_tokens)
            non_time_columns = [column for column in header if column not in {"Date", "Time"}]
            all_non_time_missing = all(
                _is_missing(row_dict.get(column, ""), missing_tokens)
                for column in non_time_columns
            )
            if date_missing and time_missing and all_non_time_missing:
                continue

            result.total_rows += 1

            for column in header:
                if _is_missing(row_dict.get(column, ""), missing_tokens):
                    result.missing_counts[column] += 1

            _update_numeric_summaries(
                row_values=row_dict,
                range_rules=AIR_QUALITY_RANGE_RULES,
                numeric_summaries=result.numeric_summaries,
                missing_tokens=missing_tokens,
                decimal_comma=True,
            )

            date_value = _clean_token(row_dict.get("Date", ""))
            time_value = _clean_token(row_dict.get("Time", ""))
            timestamp = None

            if date_value and time_value:
                try:
                    timestamp = datetime.strptime(
                        f"{date_value} {time_value}", "%d/%m/%Y %H.%M.%S"
                    )
                except ValueError:
                    result.invalid_timestamps += 1
            else:
                result.invalid_timestamps += 1

            if timestamp is not None:
                if timestamp in seen_timestamps:
                    result.duplicate_timestamps += 1
                seen_timestamps.add(timestamp)
                timestamp_values.append(timestamp)

                if previous_timestamp is not None and timestamp < previous_timestamp:
                    result.out_of_order_transitions += 1
                previous_timestamp = timestamp

            if _update_missing_runs(
                row_values=row_dict,
                key_columns=AIR_QUALITY_KEY_COLUMNS,
                missing_tokens=missing_tokens,
                current_runs=current_runs,
                max_runs=max_runs,
            ):
                result.all_key_missing_rows += 1

    result.max_missing_run = max_runs
    result.continuity_gaps, result.missing_hours = _compute_continuity_metrics(
        timestamps=timestamp_values
    )

    return result


def _station_name_from_path(file_path: Path) -> str:
    stem = file_path.stem
    return stem.replace("PRSA_Data_", "")


def audit_beijing_multi_site(beijing_dir: Path) -> DatasetAuditResult:
    result = DatasetAuditResult(name="UCI Beijing Multi-Site Air Quality")

    if not beijing_dir.exists():
        raise FileNotFoundError(f"Missing directory: {beijing_dir}")

    station_files = sorted(beijing_dir.glob("PRSA_Data_*.csv"))
    if not station_files:
        raise FileNotFoundError(
            f"No station files found in {beijing_dir} with pattern PRSA_Data_*.csv"
        )

    missing_tokens = {"", "NA"}
    aggregate_continuity_gaps = 0
    aggregate_missing_hours = 0
    current_runs_global, max_runs_global = _initialize_missing_runs(BEIJING_KEY_COLUMNS)

    for station_file in station_files:
        station_name = _station_name_from_path(station_file)
        station_timestamps: list[datetime] = []
        seen_timestamps: set[datetime] = set()
        previous_timestamp: datetime | None = None

        with station_file.open("r", encoding="utf-8", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            if reader.fieldnames is None:
                continue

            fieldnames = [_clean_token(name) for name in reader.fieldnames]
            if not result.columns:
                result.columns = fieldnames
                result.surprise_columns = [
                    column
                    for column in fieldnames
                    if column not in BEIJING_EXPECTED_COLUMNS
                ]

            if fieldnames != result.columns:
                result.station_issues.append(
                    f"{station_name}: column layout differs from first station file"
                )

            current_runs_station, max_runs_station = _initialize_missing_runs(
                BEIJING_KEY_COLUMNS
            )

            for row in reader:
                result.total_rows += 1

                for column in result.columns:
                    if _is_missing(row.get(column, ""), missing_tokens):
                        result.missing_counts[column] += 1

                _update_numeric_summaries(
                    row_values=row,
                    range_rules=BEIJING_RANGE_RULES,
                    numeric_summaries=result.numeric_summaries,
                    missing_tokens=missing_tokens,
                    decimal_comma=False,
                )

                timestamp = None
                try:
                    year = int(_clean_token(row.get("year", "")))
                    month = int(_clean_token(row.get("month", "")))
                    day = int(_clean_token(row.get("day", "")))
                    hour = int(_clean_token(row.get("hour", "")))
                    timestamp = datetime(year, month, day, hour)
                except ValueError:
                    result.invalid_timestamps += 1

                if timestamp is not None:
                    if timestamp in seen_timestamps:
                        result.duplicate_timestamps += 1
                    seen_timestamps.add(timestamp)
                    station_timestamps.append(timestamp)

                    if previous_timestamp is not None and timestamp < previous_timestamp:
                        result.out_of_order_transitions += 1
                    previous_timestamp = timestamp

                row_all_missing = _update_missing_runs(
                    row_values=row,
                    key_columns=BEIJING_KEY_COLUMNS,
                    missing_tokens=missing_tokens,
                    current_runs=current_runs_station,
                    max_runs=max_runs_station,
                )

                if row_all_missing:
                    result.all_key_missing_rows += 1

            for column in BEIJING_KEY_COLUMNS:
                if max_runs_station[column] > max_runs_global[column]:
                    max_runs_global[column] = max_runs_station[column]
                current_runs_global[column] = current_runs_station[column]

        station_gaps, station_missing_hours = _compute_continuity_metrics(station_timestamps)
        aggregate_continuity_gaps += station_gaps
        aggregate_missing_hours += station_missing_hours

        if station_gaps > 0:
            result.station_issues.append(
                f"{station_name}: continuity gaps={station_gaps}, missing_hours={station_missing_hours}"
            )

    result.max_missing_run = max_runs_global
    result.continuity_gaps = aggregate_continuity_gaps
    result.missing_hours = aggregate_missing_hours

    return result


def build_report(
    air_quality_result: DatasetAuditResult,
    beijing_result: DatasetAuditResult,
    top_missing: int,
) -> str:
    lines: list[str] = []
    lines.append("Phase 2 Data Inspection Report")
    lines.append("=" * 32)
    lines.append("")

    for result, range_rules in (
        (beijing_result, BEIJING_RANGE_RULES),
        (air_quality_result, AIR_QUALITY_RANGE_RULES),
    ):
        unexplained_missing = _evaluate_missing_patterns(result)
        no_surprise_columns = len(result.surprise_columns) == 0
        no_duplicate_timestamps = result.duplicate_timestamps == 0
        no_unexplained_missing = len(unexplained_missing) == 0

        lines.append(f"Dataset: {result.name}")
        lines.append(f"Rows inspected: {result.total_rows}")
        lines.append(f"Columns ({len(result.columns)}): {', '.join(result.columns)}")
        lines.append(
            f"No surprise columns: {'YES' if no_surprise_columns else 'NO'}"
        )
        if not no_surprise_columns:
            lines.append(f"- Surprise columns: {', '.join(result.surprise_columns)}")

        lines.append("Missing-value summary (top columns by missing ratio):")
        lines.extend(_build_missing_summary_lines(result=result, top_k=top_missing))

        lines.append("Time continuity check:")
        lines.append(f"- Invalid timestamps: {result.invalid_timestamps}")
        lines.append(f"- Duplicate timestamps: {result.duplicate_timestamps}")
        lines.append(
            f"- Out-of-order transitions in file order: {result.out_of_order_transitions}"
        )
        lines.append(
            f"- Continuity gaps (sorted unique timestamps): {result.continuity_gaps}"
        )
        lines.append(f"- Estimated missing hours from gaps: {result.missing_hours}")

        lines.append("Sensor-range sanity check:")
        lines.extend(
            _build_range_summary_lines(
                range_rules=range_rules,
                numeric_summaries=result.numeric_summaries,
            )
        )

        lines.append("Missing-pattern diagnostics:")
        lines.append(
            f"- Rows with all key pollutant columns missing: {result.all_key_missing_rows}"
        )

        for column in sorted(result.max_missing_run):
            lines.append(
                f"- Max consecutive missing run for {column}: {result.max_missing_run[column]}"
            )

        lines.append(
            f"No duplicate timestamps: {'YES' if no_duplicate_timestamps else 'NO'}"
        )
        lines.append(
            f"No unexplained missing-value patterns: {'YES' if no_unexplained_missing else 'NO'}"
        )

        if unexplained_missing:
            lines.append("- Missing-value issues to review:")
            for issue in unexplained_missing:
                lines.append(f"  - {issue}")

        if result.station_issues:
            lines.append("Station-level continuity notes:")
            for note in result.station_issues:
                lines.append(f"- {note}")

        lines.append("-" * 32)
        lines.append("")

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 2 data inspection for raw columns, missingness, "
            "time continuity, and sensor-range sanity checks."
        )
    )
    parser.add_argument(
        "--root-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Project root directory containing the data folder.",
    )
    parser.add_argument(
        "--top-missing",
        type=int,
        default=10,
        help="Number of columns to display in missing-value summary.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = args.root_dir.resolve()

    air_quality_path = project_root / "data" / "air+quality" / "AirQualityUCI.csv"
    beijing_dir = project_root / "data" / "beijing+multi+site+air+quality+data"

    air_quality_result = audit_air_quality(air_quality_path)
    beijing_result = audit_beijing_multi_site(beijing_dir)

    report = build_report(
        air_quality_result=air_quality_result,
        beijing_result=beijing_result,
        top_missing=max(args.top_missing, 1),
    )

    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
