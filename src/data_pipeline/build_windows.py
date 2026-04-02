from __future__ import annotations

import argparse
import ast
import csv
import struct
import sys
from array import array
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


EXPECTED_FEATURE_COLUMNS = [
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


@dataclass
class SplitInfo:
    name: str
    row_start: int
    row_end_exclusive: int
    row_count: int
    window_count: int
    shape: tuple[int, int, int]
    ts_start: datetime
    ts_end: datetime
    first_window_start_idx: int
    last_window_end_idx: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build chronological train/val/test windows from clean time-series data "
            "without cross-split overlap leakage."
        )
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("data") / "processed" / "clean.csv",
        help="Preprocessed clean CSV input path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data") / "windows",
        help="Directory for train/val/test .npy outputs.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=12,
        help="Sliding window size.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Sliding stride.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.6,
        help="Chronological train ratio.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Chronological validation ratio.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail with non-zero exit if any verification fails.",
    )
    return parser.parse_args()


def load_clean_data(input_csv: Path) -> tuple[list[datetime], list[list[float]], list[str]]:
    if not input_csv.exists():
        raise FileNotFoundError(f"Input file not found: {input_csv}")

    timestamps: list[datetime] = []
    rows: list[list[float]] = []

    with input_csv.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ValueError("Input CSV has no header")

        fieldnames = [name.strip() for name in reader.fieldnames]
        if not fieldnames or fieldnames[0] != "timestamp":
            raise ValueError("First column must be 'timestamp'")

        feature_columns = fieldnames[1:]
        if feature_columns != EXPECTED_FEATURE_COLUMNS:
            raise ValueError(
                "Feature columns mismatch. Expected: "
                + ", ".join(EXPECTED_FEATURE_COLUMNS)
                + " | Found: "
                + ", ".join(feature_columns)
            )

        for row in reader:
            ts_text = row.get("timestamp", "").strip()
            try:
                ts_value = datetime.strptime(ts_text, "%Y-%m-%d %H:%M:%S")
            except ValueError as exc:
                raise ValueError(f"Invalid timestamp value: {ts_text}") from exc

            feature_values: list[float] = []
            for column in EXPECTED_FEATURE_COLUMNS:
                value_text = row.get(column, "").strip()
                if value_text == "":
                    raise ValueError(
                        f"Empty feature value found in column {column} at {ts_text}"
                    )
                feature_values.append(float(value_text))

            timestamps.append(ts_value)
            rows.append(feature_values)

    return timestamps, rows, EXPECTED_FEATURE_COLUMNS.copy()


def verify_strict_time_order(timestamps: list[datetime]) -> None:
    if len(timestamps) < 2:
        raise ValueError("At least two rows are required for temporal validation")

    for previous, current in zip(timestamps, timestamps[1:]):
        if current <= previous:
            raise ValueError(
                "Input time order check failed: timestamps must be strictly increasing"
            )


def expected_window_count(row_count: int, window_size: int, stride: int) -> int:
    if row_count < window_size:
        return 0
    return ((row_count - window_size) // stride) + 1


def flatten_windows(
    split_rows: list[list[float]],
    window_size: int,
    stride: int,
) -> tuple[array, int]:
    row_count = len(split_rows)
    window_count = expected_window_count(row_count, window_size, stride)
    flat_data = array("f")

    for start_idx in range(0, row_count - window_size + 1, stride):
        end_idx = start_idx + window_size
        for row in split_rows[start_idx:end_idx]:
            flat_data.extend(row)

    return flat_data, window_count


def write_npy_float32(path: Path, shape: tuple[int, int, int], data: array) -> None:
    header_text = (
        "{'descr': '<f4', 'fortran_order': False, 'shape': " + str(shape) + ", }"
    )
    header_bytes = header_text.encode("latin1")

    magic = b"\x93NUMPY"
    version = b"\x01\x00"
    preamble_len = len(magic) + len(version) + 2
    padding_len = (-((preamble_len + len(header_bytes) + 1) % 16)) % 16
    full_header = header_bytes + (b" " * padding_len) + b"\n"

    if len(full_header) >= 65536:
        raise ValueError("NPY header too large for version 1.0")

    payload = array("f", data)
    if sys.byteorder != "little":
        payload.byteswap()

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as output_file:
        output_file.write(magic)
        output_file.write(version)
        output_file.write(struct.pack("<H", len(full_header)))
        output_file.write(full_header)
        output_file.write(payload.tobytes())


def read_npy_shape(path: Path) -> tuple[int, int, int]:
    with path.open("rb") as input_file:
        magic = input_file.read(6)
        if magic != b"\x93NUMPY":
            raise ValueError(f"Invalid NPY magic in file: {path}")

        version = input_file.read(2)
        if version != b"\x01\x00":
            raise ValueError(f"Unsupported NPY version in file: {path}")

        header_len_bytes = input_file.read(2)
        header_len = struct.unpack("<H", header_len_bytes)[0]
        header_text = input_file.read(header_len).decode("latin1")
        header_obj = ast.literal_eval(header_text.strip())

    shape = header_obj.get("shape")
    if not isinstance(shape, tuple) or len(shape) != 3:
        raise ValueError(f"Unexpected NPY shape metadata in file: {path}")
    return shape


def build_split_info(
    name: str,
    row_start: int,
    row_end_exclusive: int,
    window_size: int,
    stride: int,
    timestamps: list[datetime],
) -> SplitInfo:
    row_count = row_end_exclusive - row_start
    window_count = expected_window_count(row_count, window_size, stride)
    shape = (window_count, window_size, len(EXPECTED_FEATURE_COLUMNS))

    if row_count <= 0:
        raise ValueError(f"{name} split has no rows")

    ts_start = timestamps[row_start]
    ts_end = timestamps[row_end_exclusive - 1]

    if window_count <= 0:
        raise ValueError(
            f"{name} split has {row_count} rows but needs at least {window_size} rows"
        )

    first_window_start_idx = row_start
    last_window_start_idx = row_start + (window_count - 1) * stride
    last_window_end_idx = last_window_start_idx + window_size - 1

    if last_window_end_idx >= row_end_exclusive:
        raise ValueError(f"{name} split windowing exceeds split boundary")

    return SplitInfo(
        name=name,
        row_start=row_start,
        row_end_exclusive=row_end_exclusive,
        row_count=row_count,
        window_count=window_count,
        shape=shape,
        ts_start=ts_start,
        ts_end=ts_end,
        first_window_start_idx=first_window_start_idx,
        last_window_end_idx=last_window_end_idx,
    )


def verify_cross_split_order(
    train_info: SplitInfo,
    val_info: SplitInfo,
    test_info: SplitInfo,
    timestamps: list[datetime],
) -> tuple[bool, bool]:
    no_overlap = (
        train_info.last_window_end_idx < val_info.first_window_start_idx
        and val_info.last_window_end_idx < test_info.first_window_start_idx
    )

    strict_time_order = (
        timestamps[train_info.last_window_end_idx]
        < timestamps[val_info.first_window_start_idx]
        and timestamps[val_info.last_window_end_idx]
        < timestamps[test_info.first_window_start_idx]
    )

    return no_overlap, strict_time_order


def format_split_line(split: SplitInfo) -> str:
    return (
        f"- {split.name}: rows={split.row_count}, windows={split.window_count}, "
        f"shape={split.shape}, row_range=[{split.row_start}, {split.row_end_exclusive - 1}], "
        f"time_range=[{split.ts_start.strftime('%Y-%m-%d %H:%M:%S')} -> "
        f"{split.ts_end.strftime('%Y-%m-%d %H:%M:%S')}]"
    )


def main() -> int:
    args = parse_args()

    if args.window_size <= 0:
        raise ValueError("window-size must be > 0")
    if args.stride <= 0:
        raise ValueError("stride must be > 0")
    if args.train_ratio <= 0 or args.val_ratio <= 0:
        raise ValueError("train-ratio and val-ratio must be > 0")
    if args.train_ratio + args.val_ratio >= 1.0:
        raise ValueError("train-ratio + val-ratio must be < 1.0")

    timestamps, rows, feature_columns = load_clean_data(args.input_csv)
    verify_strict_time_order(timestamps)

    total_rows = len(rows)
    train_end = int(total_rows * args.train_ratio)
    val_end = int(total_rows * (args.train_ratio + args.val_ratio))

    if train_end <= 0 or val_end <= train_end or val_end >= total_rows:
        raise ValueError("Invalid split boundaries for the given dataset size and ratios")

    train_info = build_split_info(
        name="train",
        row_start=0,
        row_end_exclusive=train_end,
        window_size=args.window_size,
        stride=args.stride,
        timestamps=timestamps,
    )
    val_info = build_split_info(
        name="val",
        row_start=train_end,
        row_end_exclusive=val_end,
        window_size=args.window_size,
        stride=args.stride,
        timestamps=timestamps,
    )
    test_info = build_split_info(
        name="test",
        row_start=val_end,
        row_end_exclusive=total_rows,
        window_size=args.window_size,
        stride=args.stride,
        timestamps=timestamps,
    )

    no_overlap, strict_time_boundaries = verify_cross_split_order(
        train_info=train_info,
        val_info=val_info,
        test_info=test_info,
        timestamps=timestamps,
    )

    train_flat, train_count = flatten_windows(
        split_rows=rows[train_info.row_start : train_info.row_end_exclusive],
        window_size=args.window_size,
        stride=args.stride,
    )
    val_flat, val_count = flatten_windows(
        split_rows=rows[val_info.row_start : val_info.row_end_exclusive],
        window_size=args.window_size,
        stride=args.stride,
    )
    test_flat, test_count = flatten_windows(
        split_rows=rows[test_info.row_start : test_info.row_end_exclusive],
        window_size=args.window_size,
        stride=args.stride,
    )

    if train_count != train_info.window_count:
        raise ValueError("Train window count mismatch")
    if val_count != val_info.window_count:
        raise ValueError("Validation window count mismatch")
    if test_count != test_info.window_count:
        raise ValueError("Test window count mismatch")

    window_count_sanity = (
        train_count > 0
        and val_count > 0
        and test_count > 0
        and train_count <= train_info.row_count
        and val_count <= val_info.row_count
        and test_count <= test_info.row_count
    )

    train_path = args.output_dir / "train.npy"
    val_path = args.output_dir / "val.npy"
    test_path = args.output_dir / "test.npy"

    write_npy_float32(train_path, train_info.shape, train_flat)
    write_npy_float32(val_path, val_info.shape, val_flat)
    write_npy_float32(test_path, test_info.shape, test_flat)

    train_shape_on_disk = read_npy_shape(train_path)
    val_shape_on_disk = read_npy_shape(val_path)
    test_shape_on_disk = read_npy_shape(test_path)

    shapes_ok = (
        train_shape_on_disk == train_info.shape
        and val_shape_on_disk == val_info.shape
        and test_shape_on_disk == test_info.shape
        and train_info.shape[1:] == (args.window_size, len(feature_columns))
        and val_info.shape[1:] == (args.window_size, len(feature_columns))
        and test_info.shape[1:] == (args.window_size, len(feature_columns))
    )

    print("Phase 4 Windowing Report")
    print("=" * 24)
    print(f"Input CSV: {args.input_csv}")
    print(f"Output directory: {args.output_dir}")
    print(f"Window size: {args.window_size}")
    print(f"Stride: {args.stride}")
    print(f"Features: {len(feature_columns)}")
    print(f"Total rows: {total_rows}")
    print("")
    print("Split summary")
    print(format_split_line(train_info))
    print(format_split_line(val_info))
    print(format_split_line(test_info))
    print("")
    print("Saved array shapes")
    print(f"- train.npy: {train_shape_on_disk}")
    print(f"- val.npy: {val_shape_on_disk}")
    print(f"- test.npy: {test_shape_on_disk}")
    print("")
    print("Verification")
    print(f"- Shape check (N, {args.window_size}, {len(feature_columns)}): {shapes_ok}")
    print(f"- Time order boundaries strict (train < val < test): {strict_time_boundaries}")
    print(f"- No cross-split window overlap leakage: {no_overlap}")
    print(f"- Window count sanity: {window_count_sanity}")
    print(
        "- Boundary timestamps: "
        f"train_last_window_end={timestamps[train_info.last_window_end_idx].strftime('%Y-%m-%d %H:%M:%S')}, "
        f"val_first_window_start={timestamps[val_info.first_window_start_idx].strftime('%Y-%m-%d %H:%M:%S')}, "
        f"val_last_window_end={timestamps[val_info.last_window_end_idx].strftime('%Y-%m-%d %H:%M:%S')}, "
        f"test_first_window_start={timestamps[test_info.first_window_start_idx].strftime('%Y-%m-%d %H:%M:%S')}"
    )

    if args.strict:
        if not shapes_ok:
            raise RuntimeError("Strict check failed: shape validation failed")
        if not strict_time_boundaries:
            raise RuntimeError("Strict check failed: split time boundaries are not strict")
        if not no_overlap:
            raise RuntimeError("Strict check failed: cross-split window overlap detected")
        if not window_count_sanity:
            raise RuntimeError("Strict check failed: window count sanity check failed")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
