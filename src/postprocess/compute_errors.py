from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

try:
    from src.modeling.model_autoencoder import TinyAutoencoder
except ModuleNotFoundError:
    from model_autoencoder import TinyAutoencoder


EPS = 1e-12


@dataclass
class ErrorStats:
    split: str
    expected_count: int
    count: int
    coverage_ok: bool
    finite_ok: bool
    non_negative_ok: bool
    not_constant_ok: bool
    min_error: float
    max_error: float
    mean_error: float
    std_error: float
    p90: float
    p95: float
    p99: float
    tail_span_p99_p90: float


@dataclass
class ErrorSummary:
    checkpoint_path: str
    device: str
    input_dim: int
    window_size: int
    feature_count: int
    train: ErrorStats
    val: ErrorStats
    test: ErrorStats
    warnings: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute reconstruction error distributions for train/val/test windows "
            "using a trained tiny autoencoder checkpoint."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("data") / "training" / "tiny_ae_best.pt",
        help="Model checkpoint path.",
    )
    parser.add_argument(
        "--train-npy",
        type=Path,
        default=Path("data") / "windows" / "train.npy",
        help="Train windows path.",
    )
    parser.add_argument(
        "--val-npy",
        type=Path,
        default=Path("data") / "windows" / "val.npy",
        help="Validation windows path.",
    )
    parser.add_argument(
        "--test-npy",
        type=Path,
        default=Path("data") / "windows" / "test.npy",
        help="Test windows path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("errors"),
        help="Output directory for extracted errors.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Batch size for inference.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic setup.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: auto, cpu, cuda.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if core verification checks fail.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def choose_device(device_arg: str) -> torch.device:
    normalized = device_arg.strip().lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if normalized in {"cpu", "cuda"}:
        if normalized == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device(normalized)

    raise ValueError("device must be one of: auto, cpu, cuda")


def load_windows(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Window file not found: {path}")

    data = np.load(path, allow_pickle=False)
    if data.ndim != 3:
        raise ValueError(f"Expected 3D windows at {path}, got shape {data.shape}")

    if not np.isfinite(data).all():
        raise ValueError(f"Non-finite values in windows file: {path}")

    return data.astype(np.float32, copy=False)


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[TinyAutoencoder, dict]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint missing 'model_state_dict'")

    input_dim = int(checkpoint.get("input_dim", 0))
    if input_dim <= 0:
        raise ValueError("Checkpoint missing valid input_dim")

    hidden_sizes_raw = checkpoint.get("hidden_sizes", [64, 32])
    hidden_sizes = tuple(int(value) for value in hidden_sizes_raw)
    activation = str(checkpoint.get("activation", "relu"))

    model = TinyAutoencoder(
        input_dim=input_dim,
        hidden_sizes=hidden_sizes,
        activation=activation,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    meta = {
        "input_dim": input_dim,
        "hidden_sizes": hidden_sizes,
        "activation": activation,
    }
    return model, meta


def validate_window_shapes(
    train_windows: np.ndarray,
    val_windows: np.ndarray,
    test_windows: np.ndarray,
    expected_input_dim: int,
) -> tuple[int, int]:
    base_shape = train_windows.shape[1:]
    if val_windows.shape[1:] != base_shape or test_windows.shape[1:] != base_shape:
        raise ValueError(
            "Window shape mismatch across splits: "
            f"train={train_windows.shape}, val={val_windows.shape}, test={test_windows.shape}"
        )

    window_size = int(base_shape[0])
    feature_count = int(base_shape[1])
    input_dim = window_size * feature_count

    if input_dim != expected_input_dim:
        raise ValueError(
            "Checkpoint input_dim mismatch with windows: "
            f"checkpoint={expected_input_dim}, computed={input_dim}"
        )

    return window_size, feature_count


def compute_errors(
    model: TinyAutoencoder,
    windows: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    dataset = TensorDataset(torch.from_numpy(windows))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    output = np.empty((windows.shape[0],), dtype=np.float64)
    cursor = 0

    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            errors = model.reconstruction_error(batch, reduction="none")
            errors_np = errors.detach().cpu().numpy().astype(np.float64)
            n_batch = int(errors_np.shape[0])
            output[cursor : cursor + n_batch] = errors_np
            cursor += n_batch

    if cursor != windows.shape[0]:
        raise RuntimeError(
            f"Error extraction coverage mismatch: expected {windows.shape[0]}, got {cursor}"
        )

    return output


def summarize_split(name: str, errors: np.ndarray, expected_count: int) -> ErrorStats:
    count = int(errors.shape[0])
    coverage_ok = bool(count == expected_count)
    finite_ok = bool(np.isfinite(errors).all())

    # MSE-based reconstruction error should be non-negative.
    non_negative_ok = bool(np.all(errors >= -1e-12))
    if not non_negative_ok:
        errors = np.maximum(errors, 0.0)

    std_value = float(np.std(errors))
    not_constant_ok = bool(std_value > 1e-12 and (float(errors.max()) - float(errors.min())) > 1e-12)

    p90, p95, p99 = np.quantile(errors, [0.90, 0.95, 0.99]).astype(np.float64)

    return ErrorStats(
        split=name,
        expected_count=expected_count,
        count=count,
        coverage_ok=coverage_ok,
        finite_ok=finite_ok,
        non_negative_ok=non_negative_ok,
        not_constant_ok=not_constant_ok,
        min_error=float(np.min(errors)),
        max_error=float(np.max(errors)),
        mean_error=float(np.mean(errors)),
        std_error=std_value,
        p90=float(p90),
        p95=float(p95),
        p99=float(p99),
        tail_span_p99_p90=float(p99 - p90),
    )


def save_errors(output_dir: Path, train_err: np.ndarray, val_err: np.ndarray, test_err: np.ndarray) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "train_errors.npy", train_err.astype(np.float32), allow_pickle=False)
    np.save(output_dir / "val_errors.npy", val_err.astype(np.float32), allow_pickle=False)
    np.save(output_dir / "test_errors.npy", test_err.astype(np.float32), allow_pickle=False)


def save_summary(output_dir: Path, summary: ErrorSummary) -> None:
    summary_json = output_dir / "error_summary.json"
    summary_csv = output_dir / "error_percentiles.csv"

    with summary_json.open("w", encoding="utf-8") as output_file:
        json.dump(asdict(summary), output_file, indent=2)

    rows = [summary.train, summary.val, summary.test]
    with summary_csv.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(
            [
                "split",
                "count",
                "mean",
                "std",
                "p90",
                "p95",
                "p99",
                "tail_span_p99_p90",
                "coverage_ok",
                "finite_ok",
                "non_negative_ok",
                "not_constant_ok",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.split,
                    row.count,
                    f"{row.mean_error:.10f}",
                    f"{row.std_error:.10f}",
                    f"{row.p90:.10f}",
                    f"{row.p95:.10f}",
                    f"{row.p99:.10f}",
                    f"{row.tail_span_p99_p90:.10f}",
                    row.coverage_ok,
                    row.finite_ok,
                    row.non_negative_ok,
                    row.not_constant_ok,
                ]
            )


def build_warnings(train: ErrorStats, val: ErrorStats, test: ErrorStats) -> list[str]:
    warnings: list[str] = []

    if test.std_error < 1e-5:
        warnings.append("Test error variance is extremely low; potential over-reconstruction risk.")
    if test.std_error > max(train.std_error, EPS) * 5.0:
        warnings.append("Test error variance is much higher than train variance; check stability.")
    if test.mean_error < train.mean_error * 0.7:
        warnings.append("Test mean error is much lower than train mean error; check for leakage/data mismatch.")
    if test.tail_span_p99_p90 <= 1e-4:
        warnings.append("Test upper-tail span (p99-p90) is very small; anomaly separation may be weak.")

    return warnings


def print_split(stats: ErrorStats) -> None:
    print(f"- {stats.split}: count={stats.count}")
    print(
        "  checks: "
        f"coverage={stats.coverage_ok}, finite={stats.finite_ok}, "
        f"non_negative={stats.non_negative_ok}, not_constant={stats.not_constant_ok}"
    )
    print(
        "  distribution: "
        f"mean={stats.mean_error:.8f}, std={stats.std_error:.8f}, "
        f"p90={stats.p90:.8f}, p95={stats.p95:.8f}, p99={stats.p99:.8f}, "
        f"tail_span_p99_p90={stats.tail_span_p99_p90:.8f}"
    )


def main() -> int:
    args = parse_args()

    if args.batch_size <= 0:
        raise ValueError("batch-size must be > 0")

    set_seed(args.seed)
    device = choose_device(args.device)

    train_windows = load_windows(args.train_npy)
    val_windows = load_windows(args.val_npy)
    test_windows = load_windows(args.test_npy)

    model, meta = load_model(args.checkpoint, device)

    window_size, feature_count = validate_window_shapes(
        train_windows=train_windows,
        val_windows=val_windows,
        test_windows=test_windows,
        expected_input_dim=int(meta["input_dim"]),
    )

    train_errors = compute_errors(model, train_windows, args.batch_size, device)
    val_errors = compute_errors(model, val_windows, args.batch_size, device)
    test_errors = compute_errors(model, test_windows, args.batch_size, device)

    train_stats = summarize_split("train", train_errors, expected_count=int(train_windows.shape[0]))
    val_stats = summarize_split("val", val_errors, expected_count=int(val_windows.shape[0]))
    test_stats = summarize_split("test", test_errors, expected_count=int(test_windows.shape[0]))

    warnings = build_warnings(train=train_stats, val=val_stats, test=test_stats)

    summary = ErrorSummary(
        checkpoint_path=str(args.checkpoint),
        device=str(device),
        input_dim=int(meta["input_dim"]),
        window_size=window_size,
        feature_count=feature_count,
        train=train_stats,
        val=val_stats,
        test=test_stats,
        warnings=warnings,
    )

    save_errors(args.output_dir, train_errors, val_errors, test_errors)
    save_summary(args.output_dir, summary)

    print("Error Extraction Report")
    print("=" * 23)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {device}")
    print(f"Input dim: {meta['input_dim']}")
    print(f"Window shape: ({window_size}, {feature_count})")
    print(f"Output directory: {args.output_dir}")
    print("")

    print_split(train_stats)
    print_split(val_stats)
    print_split(test_stats)

    print("")
    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"- {warning}")
    else:
        print("Warnings: none")

    if args.strict:
        for stats in (train_stats, val_stats, test_stats):
            if not stats.coverage_ok:
                raise RuntimeError(f"Strict mode failed: {stats.split} coverage mismatch")
            if not stats.finite_ok:
                raise RuntimeError(f"Strict mode failed: {stats.split} contains non-finite values")
            if not stats.non_negative_ok:
                raise RuntimeError(f"Strict mode failed: {stats.split} has negative errors")
            if not stats.not_constant_ok:
                raise RuntimeError(f"Strict mode failed: {stats.split} error distribution is constant")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
