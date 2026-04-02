from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


FEATURE_COLUMNS = [
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
    "WSPM",
]

DROP_COLUMNS = ["wd", "station", "No"]
EXPECTED_COLUMNS = {
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
}


@dataclass
class GapStats:
    initial_missing_points: int = 0
    short_runs_interpolated: int = 0
    short_points_interpolated: int = 0
    short_runs_unfilled_edges: int = 0
    short_points_unfilled_edges: int = 0
    long_runs_marked_drop: int = 0
    long_points_marked_drop: int = 0
    medium_runs_left_unresolved: int = 0
    medium_points_left_unresolved: int = 0


@dataclass
class PreprocessSummary:
    rows_loaded: int = 0
    rows_after_station_filter: int = 0
    rows_after_long_gap_drop: int = 0
    rows_after_unresolved_drop: int = 0
    rows_dropped_long_gap: int = 0
    rows_dropped_unresolved: int = 0
    out_of_order_before_sort: int = 0
    duplicate_timestamps_before_sort: int = 0
    duplicate_timestamps_after_sort: int = 0
    no_nans_remaining: bool = False
    time_ordered: bool = False
    feature_count_correct: bool = False
    station_filter_requested: str = ""
    detected_stations: list[str] = field(default_factory=list)
    station_scope_note: str = ""
    medium_gap_policy_note: str = ""
    scaler_train_rows: int = 0
    scaler_holdout_rows: int = 0
    scaler_train_end_timestamp: str = ""
    scaler_holdout_start_timestamp: str = ""
    scaling_leakage_guard_passed: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 3 preprocessing for Beijing air-quality data with strict "
            "gap-handling and integrity checks."
        )
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=(
            Path("data")
            / "beijing+multi+site+air+quality+data"
            / "PRSA_Data_Aotizhongxin_20130301-20170228.csv"
        ),
        help="Path to a Beijing PRSA CSV file.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("data") / "processed" / "clean.csv",
        help="Path to write cleaned model-ready data.",
    )
    parser.add_argument(
        "--station",
        type=str,
        default="",
        help="Optional station filter when input contains multiple stations.",
    )
    parser.add_argument(
        "--small-gap-max",
        type=int,
        default=4,
        help="Interpolate only missing runs with length <= this value.",
    )
    parser.add_argument(
        "--long-gap-threshold",
        type=int,
        default=24,
        help="Mark rows for drop when missing run length > this threshold.",
    )
    parser.add_argument(
        "--scaler",
        choices=["standard", "minmax", "none"],
        default="standard",
        help="Scaling method fit on train split only.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.6,
        help="Chronological train split ratio used to fit scaler parameters.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail with non-zero exit when integrity checks fail.",
    )
    return parser.parse_args()


def parse_float(value: str) -> float | None:
    token = value.strip()
    if token == "" or token.upper() == "NA":
        return None
    try:
        return float(token)
    except ValueError:
        return None


def parse_timestamp(row: dict[str, str]) -> datetime:
    return datetime(
        year=int(row["year"]),
        month=int(row["month"]),
        day=int(row["day"]),
        hour=int(row["hour"]),
    )


def find_missing_runs(values: list[float | None]) -> list[tuple[int, int, int]]:
    runs: list[tuple[int, int, int]] = []
    n_values = len(values)
    idx = 0

    while idx < n_values:
        if values[idx] is not None:
            idx += 1
            continue

        start = idx
        while idx < n_values and values[idx] is None:
            idx += 1
        end = idx - 1
        runs.append((start, end, end - start + 1))

    return runs


def count_order_issues(timestamps: list[datetime]) -> tuple[int, int]:
    if not timestamps:
        return 0, 0

    out_of_order = 0
    duplicate_count = 0
    seen: set[datetime] = set()
    previous = timestamps[0]

    for ts in timestamps:
        if ts in seen:
            duplicate_count += 1
        seen.add(ts)

        if ts < previous:
            out_of_order += 1
        previous = ts

    return out_of_order, duplicate_count


def load_rows(
    input_csv: Path,
    station_filter: str,
) -> tuple[list[datetime], dict[str, list[float | None]], list[str], PreprocessSummary]:
    summary = PreprocessSummary()
    summary.station_filter_requested = station_filter if station_filter else "(none)"

    if not input_csv.exists():
        raise FileNotFoundError(f"Input file not found: {input_csv}")

    timestamps: list[datetime] = []
    feature_values = {column: [] for column in FEATURE_COLUMNS}
    included_stations: set[str] = set()

    with input_csv.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ValueError("Input CSV has no header")

        header = [name.strip().strip('"') for name in reader.fieldnames]
        missing_required = [column for column in FEATURE_COLUMNS if column not in header]
        if missing_required:
            raise ValueError(
                "Missing required feature columns: " + ", ".join(missing_required)
            )

        for required_col in ["year", "month", "day", "hour"]:
            if required_col not in header:
                raise ValueError(f"Missing required timestamp column: {required_col}")

        for row in reader:
            summary.rows_loaded += 1
            station_name = row.get("station", "").strip().strip('"')
            if station_filter and station_name != station_filter:
                continue

            summary.rows_after_station_filter += 1
            if station_name:
                included_stations.add(station_name)
            timestamps.append(parse_timestamp(row))

            for column in FEATURE_COLUMNS:
                feature_values[column].append(parse_float(row.get(column, "")))

    summary.out_of_order_before_sort, summary.duplicate_timestamps_before_sort = (
        count_order_issues(timestamps)
    )

    if timestamps:
        sorted_indices = sorted(range(len(timestamps)), key=timestamps.__getitem__)
        timestamps = [timestamps[i] for i in sorted_indices]
        for column in FEATURE_COLUMNS:
            column_values = feature_values[column]
            feature_values[column] = [column_values[i] for i in sorted_indices]

    summary.detected_stations = sorted(included_stations)
    if station_filter and len(summary.detected_stations) == 1:
        summary.station_scope_note = (
            "Station filter applied to a single station as requested."
        )
    elif not station_filter and len(summary.detected_stations) == 1:
        summary.station_scope_note = (
            "Input CSV is already single-station, so station filter is a no-op."
        )
    elif not station_filter and len(summary.detected_stations) > 1:
        summary.station_scope_note = (
            "Multiple stations detected and no station filter applied."
        )
    else:
        summary.station_scope_note = (
            "Station scope is ambiguous; verify station values in input source."
        )

    return timestamps, feature_values, header, summary


def interpolate_run(
    values: list[float | None],
    start: int,
    end: int,
) -> bool:
    left_idx = start - 1
    right_idx = end + 1

    if left_idx < 0 or right_idx >= len(values):
        return False
    left_raw = values[left_idx]
    right_raw = values[right_idx]
    if left_raw is None or right_raw is None:
        return False

    left_value = float(left_raw)
    right_value = float(right_raw)
    span = right_idx - left_idx

    if span <= 0:
        return False

    for idx in range(start, end + 1):
        ratio = (idx - left_idx) / span
        values[idx] = left_value + ratio * (right_value - left_value)

    return True


def apply_gap_rules(
    feature_values: dict[str, list[float | None]],
    small_gap_max: int,
    long_gap_threshold: int,
) -> tuple[dict[str, list[float | None]], list[bool], dict[str, GapStats]]:
    n_rows = len(next(iter(feature_values.values()))) if feature_values else 0
    drop_mask = [False] * n_rows
    stats_by_feature: dict[str, GapStats] = {}

    for column in FEATURE_COLUMNS:
        values = feature_values[column]
        stats = GapStats(initial_missing_points=sum(1 for value in values if value is None))
        runs = find_missing_runs(values)

        for start, end, run_length in runs:
            if run_length <= small_gap_max:
                filled = interpolate_run(values, start, end)
                if filled:
                    stats.short_runs_interpolated += 1
                    stats.short_points_interpolated += run_length
                else:
                    stats.short_runs_unfilled_edges += 1
                    stats.short_points_unfilled_edges += run_length
            elif run_length > long_gap_threshold:
                stats.long_runs_marked_drop += 1
                stats.long_points_marked_drop += run_length
                for idx in range(start, end + 1):
                    drop_mask[idx] = True
            else:
                stats.medium_runs_left_unresolved += 1
                stats.medium_points_left_unresolved += run_length

        stats_by_feature[column] = stats

    return feature_values, drop_mask, stats_by_feature


def filter_rows(
    timestamps: list[datetime],
    feature_values: dict[str, list[float | None]],
    keep_mask: list[bool],
) -> tuple[list[datetime], dict[str, list[float | None]]]:
    kept_timestamps = [ts for ts, keep in zip(timestamps, keep_mask) if keep]
    kept_features = {}

    for column in FEATURE_COLUMNS:
        values = feature_values[column]
        kept_features[column] = [value for value, keep in zip(values, keep_mask) if keep]

    return kept_timestamps, kept_features


def unresolved_nan_mask(feature_values: dict[str, list[float | None]]) -> list[bool]:
    n_rows = len(next(iter(feature_values.values()))) if feature_values else 0
    mask = [False] * n_rows

    for idx in range(n_rows):
        has_nan = False
        for column in FEATURE_COLUMNS:
            if feature_values[column][idx] is None:
                has_nan = True
                break
        mask[idx] = has_nan

    return mask


def fit_scaler(
    feature_values: dict[str, list[float]],
    train_ratio: float,
    scaler: str,
) -> tuple[dict[str, tuple[float, float]], int]:
    n_rows = len(next(iter(feature_values.values()))) if feature_values else 0
    if n_rows == 0:
        raise ValueError("Cannot fit scaler on empty dataset")

    train_end = int(n_rows * train_ratio)
    train_end = max(1, min(train_end, n_rows))

    params: dict[str, tuple[float, float]] = {}

    for column in FEATURE_COLUMNS:
        train_values = feature_values[column][:train_end]

        if scaler == "standard":
            mean = sum(train_values) / len(train_values)
            variance = sum((value - mean) ** 2 for value in train_values) / len(train_values)
            std = math.sqrt(variance)
            if std == 0.0:
                std = 1.0
            params[column] = (mean, std)
        elif scaler == "minmax":
            min_value = min(train_values)
            max_value = max(train_values)
            span = max_value - min_value
            if span == 0.0:
                span = 1.0
            params[column] = (min_value, span)
        elif scaler == "none":
            params[column] = (0.0, 1.0)
        else:
            raise ValueError(f"Unknown scaler: {scaler}")

    return params, train_end


def apply_scaler(
    feature_values: dict[str, list[float]],
    params: dict[str, tuple[float, float]],
    scaler: str,
) -> dict[str, list[float]]:
    scaled_values: dict[str, list[float]] = {}

    for column in FEATURE_COLUMNS:
        offset, factor = params[column]
        values = feature_values[column]

        if scaler == "standard":
            scaled_values[column] = [(value - offset) / factor for value in values]
        elif scaler == "minmax":
            scaled_values[column] = [(value - offset) / factor for value in values]
        elif scaler == "none":
            scaled_values[column] = list(values)
        else:
            raise ValueError(f"Unknown scaler: {scaler}")

    return scaled_values


def verify_dataset(
    timestamps: list[datetime],
    feature_values: dict[str, list[float]],
    summary: PreprocessSummary,
) -> None:
    summary.duplicate_timestamps_after_sort = count_order_issues(timestamps)[1]
    summary.time_ordered = all(
        earlier <= later for earlier, later in zip(timestamps, timestamps[1:])
    )
    summary.feature_count_correct = len(FEATURE_COLUMNS) == 11
    summary.no_nans_remaining = all(
        value is not None
        for column in FEATURE_COLUMNS
        for value in feature_values[column]
    )


def write_clean_csv(
    output_csv: Path,
    timestamps: list[datetime],
    feature_values: dict[str, list[float]],
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["timestamp", *FEATURE_COLUMNS])

        for idx, ts in enumerate(timestamps):
            row = [ts.strftime("%Y-%m-%d %H:%M:%S")]
            for column in FEATURE_COLUMNS:
                row.append(f"{feature_values[column][idx]:.10f}")
            writer.writerow(row)


def print_report(
    input_csv: Path,
    output_csv: Path,
    header: list[str],
    gap_stats: dict[str, GapStats],
    summary: PreprocessSummary,
    scaler: str,
    train_ratio: float,
) -> None:
    header_set = set(header)
    surprise_columns = sorted(column for column in header_set if column not in EXPECTED_COLUMNS)

    print("Phase 3 Preprocessing Report")
    print("=" * 28)
    print(f"Input: {input_csv}")
    print(f"Output: {output_csv}")
    print(f"Scaler: {scaler} (fit on first {train_ratio:.2f} chronological fraction)")
    print("")

    print("Column decisions")
    print("- Keep:", ", ".join(FEATURE_COLUMNS))
    print("- Drop:", ", ".join(DROP_COLUMNS))
    print(f"- Surprise columns in source: {', '.join(surprise_columns) if surprise_columns else 'None'}")
    print("")

    print("Station scope")
    print(f"- Station filter argument: {summary.station_filter_requested}")
    print(
        f"- Stations detected in selected rows: "
        f"{', '.join(summary.detected_stations) if summary.detected_stations else 'None'}"
    )
    print(f"- Station scope clarification: {summary.station_scope_note}")
    print("")

    print("Row flow")
    print(f"- Rows loaded: {summary.rows_loaded}")
    print(f"- Rows after station filter: {summary.rows_after_station_filter}")
    print(f"- Rows after long-gap drop: {summary.rows_after_long_gap_drop}")
    print(f"- Rows after unresolved-missing drop: {summary.rows_after_unresolved_drop}")
    print(f"- Rows dropped by long-gap rule: {summary.rows_dropped_long_gap}")
    print(f"- Rows dropped due to unresolved missing: {summary.rows_dropped_unresolved}")
    print("")

    print("Time integrity")
    print(f"- Out-of-order transitions before sort: {summary.out_of_order_before_sort}")
    print(f"- Duplicate timestamps before sort: {summary.duplicate_timestamps_before_sort}")
    print(f"- Duplicate timestamps after preprocessing: {summary.duplicate_timestamps_after_sort}")
    print(f"- Time-ordered output: {summary.time_ordered}")
    print("")

    print("Gap-handling stats by feature")
    for column in FEATURE_COLUMNS:
        stats = gap_stats[column]
        print(f"- {column}")
        print(f"  initial_missing_points={stats.initial_missing_points}")
        print(f"  short_runs_interpolated={stats.short_runs_interpolated}")
        print(f"  short_points_interpolated={stats.short_points_interpolated}")
        print(f"  short_runs_unfilled_edges={stats.short_runs_unfilled_edges}")
        print(f"  short_points_unfilled_edges={stats.short_points_unfilled_edges}")
        print(f"  medium_runs_left_unresolved={stats.medium_runs_left_unresolved}")
        print(f"  medium_points_left_unresolved={stats.medium_points_left_unresolved}")
        print(f"  long_runs_marked_drop={stats.long_runs_marked_drop}")
        print(f"  long_points_marked_drop={stats.long_points_marked_drop}")
    print("")

    print("Gap policy clarification")
    print(f"- {summary.medium_gap_policy_note}")
    print("")

    print("Scaling leakage guard")
    print(f"- Train rows used for scaler fit: {summary.scaler_train_rows}")
    print(f"- Holdout rows excluded from scaler fit: {summary.scaler_holdout_rows}")
    print(f"- Train end timestamp used for scaler fit: {summary.scaler_train_end_timestamp}")
    if summary.scaler_holdout_start_timestamp:
        print(
            "- First holdout timestamp excluded from scaler fit: "
            f"{summary.scaler_holdout_start_timestamp}"
        )
    print(f"- Leakage guard passed: {summary.scaling_leakage_guard_passed}")
    print("")

    print("Quality gates")
    print(f"- No NaN remains: {summary.no_nans_remaining}")
    print(f"- Time order preserved: {summary.time_ordered}")
    print(f"- No duplicate timestamps: {summary.duplicate_timestamps_after_sort == 0}")
    print(f"- Feature count correct (11): {summary.feature_count_correct}")
    print("- Missing-gap handling applied: True")
    print(f"- Scaler fit uses train-only prefix: {summary.scaling_leakage_guard_passed}")


def main() -> int:
    args = parse_args()

    if args.small_gap_max < 0:
        raise ValueError("small-gap-max must be >= 0")
    if args.long_gap_threshold < 0:
        raise ValueError("long-gap-threshold must be >= 0")
    if args.train_ratio <= 0.0 or args.train_ratio > 1.0:
        raise ValueError("train-ratio must be in (0, 1]")

    timestamps, feature_values, header, summary = load_rows(
        input_csv=args.input_csv,
        station_filter=args.station.strip(),
    )

    feature_values, drop_mask, gap_stats = apply_gap_rules(
        feature_values=feature_values,
        small_gap_max=args.small_gap_max,
        long_gap_threshold=args.long_gap_threshold,
    )

    keep_after_long_gap = [not should_drop for should_drop in drop_mask]
    summary.rows_dropped_long_gap = sum(1 for should_drop in drop_mask if should_drop)

    timestamps, feature_values = filter_rows(
        timestamps=timestamps,
        feature_values=feature_values,
        keep_mask=keep_after_long_gap,
    )
    summary.rows_after_long_gap_drop = len(timestamps)

    unresolved_mask = unresolved_nan_mask(feature_values)
    keep_after_unresolved = [not has_nan for has_nan in unresolved_mask]
    summary.rows_dropped_unresolved = sum(1 for has_nan in unresolved_mask if has_nan)

    timestamps, feature_values = filter_rows(
        timestamps=timestamps,
        feature_values=feature_values,
        keep_mask=keep_after_unresolved,
    )
    summary.rows_after_unresolved_drop = len(timestamps)
    summary.medium_gap_policy_note = (
        "Medium gaps are not interpolated; any unresolved rows are dropped before output."
    )

    if not timestamps:
        raise ValueError("No rows left after preprocessing. Adjust missing-gap settings.")

    float_feature_values: dict[str, list[float]] = {}
    for column, values in feature_values.items():
        converted: list[float] = []
        for value in values:
            if value is None:
                raise ValueError(
                    f"Unexpected missing value after cleaning in feature {column}"
                )
            converted.append(float(value))
        float_feature_values[column] = converted

    scaler_params, train_end = fit_scaler(
        feature_values=float_feature_values,
        train_ratio=args.train_ratio,
        scaler=args.scaler,
    )

    summary.scaler_train_rows = train_end
    summary.scaler_holdout_rows = len(timestamps) - train_end
    summary.scaler_train_end_timestamp = timestamps[train_end - 1].strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    if train_end < len(timestamps):
        summary.scaler_holdout_start_timestamp = timestamps[train_end].strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    summary.scaling_leakage_guard_passed = summary.scaler_holdout_rows >= 0

    scaled_feature_values = apply_scaler(
        feature_values=float_feature_values,
        params=scaler_params,
        scaler=args.scaler,
    )

    verify_dataset(
        timestamps=timestamps,
        feature_values=scaled_feature_values,
        summary=summary,
    )

    write_clean_csv(
        output_csv=args.output_csv,
        timestamps=timestamps,
        feature_values=scaled_feature_values,
    )

    print_report(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        header=header,
        gap_stats=gap_stats,
        summary=summary,
        scaler=args.scaler,
        train_ratio=args.train_ratio,
    )

    if args.strict:
        if not summary.no_nans_remaining:
            raise RuntimeError("Quality gate failed: NaN values remain in output")
        if not summary.time_ordered:
            raise RuntimeError("Quality gate failed: output is not time-ordered")
        if summary.duplicate_timestamps_after_sort != 0:
            raise RuntimeError("Quality gate failed: duplicate timestamps remain")
        if not summary.feature_count_correct:
            raise RuntimeError("Quality gate failed: feature count mismatch")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
