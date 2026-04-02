from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

EPS = 1e-12


@dataclass(frozen=True)
class FuzzyConfig:
    breakpoints: tuple[float, float, float, float]
    rule_weights: tuple[float, float, float]
    decision_threshold: float


@dataclass
class FuzzyResult:
    sample_count: int
    calibration_sample_count: int
    score_count: int
    decision_count: int
    score_coverage_ok: bool
    decision_coverage_ok: bool
    breakpoint_source: str
    threshold_source: str
    score_min: float
    score_max: float
    score_mean: float
    score_std: float
    anomaly_count: int
    anomaly_rate: float
    sensitivity_min_rate: float
    sensitivity_max_rate: float
    reproducibility_max_abs_diff: float
    reproducibility_ok: bool
    monotonicity_ok: bool


@dataclass
class ThresholdSensitivityPoint:
    threshold: float
    anomaly_count: int
    anomaly_rate: float


def parse_tuple_of_floats(text: str, expected_len: int, name: str) -> tuple[float, ...]:
    values: list[float] = []
    for token in text.split(","):
        stripped = token.strip()
        if not stripped:
            continue
        values.append(float(stripped))

    if len(values) != expected_len:
        raise ValueError(
            f"{name} must contain exactly {expected_len} comma-separated values"
        )

    return tuple(values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert reconstruction errors into fuzzy anomaly scores using low/medium/high "
            "memberships and rule-based defuzzification."
        )
    )
    parser.add_argument(
        "--errors-npy",
        type=Path,
        default=None,
        help="Optional input .npy with reconstruction errors (1D or flattenable).",
    )
    parser.add_argument(
        "--errors-csv",
        type=Path,
        default=None,
        help="Optional input CSV containing error values.",
    )
    parser.add_argument(
        "--errors-column",
        type=str,
        default="reconstruction_error",
        help="Column name used when --errors-csv is provided.",
    )
    parser.add_argument(
        "--calibration-errors-npy",
        type=Path,
        default=None,
        help="Optional calibration errors .npy (recommended: validation reconstruction errors).",
    )
    parser.add_argument(
        "--calibration-errors-csv",
        type=Path,
        default=None,
        help="Optional calibration errors CSV.",
    )
    parser.add_argument(
        "--calibration-errors-column",
        type=str,
        default="reconstruction_error",
        help="Column name used when --calibration-errors-csv is provided.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data") / "fuzzy",
        help="Output directory for fuzzy score artifacts.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="fuzzy",
        help="Prefix for output artifact files.",
    )
    parser.add_argument(
        "--breakpoints",
        type=str,
        default="0.02,0.06,0.12,0.20",
        help=(
            "Four strictly increasing breakpoints a,b,c,d for memberships: "
            "low shoulder (a,b), medium triangle (b,c,d), high shoulder (c,d)."
        ),
    )
    parser.add_argument(
        "--auto-breakpoints",
        action="store_true",
        help=(
            "Derive breakpoints from calibration error quantiles "
            "(0.50, 0.75, 0.90, 0.97)."
        ),
    )
    parser.add_argument(
        "--rule-weights",
        type=str,
        default="0.0,0.5,1.0",
        help="Rule weights for low,medium,high memberships.",
    )
    parser.add_argument(
        "--decision-threshold",
        type=float,
        default=0.60,
        help="Decision rule: anomaly if fuzzy_score >= threshold.",
    )
    parser.add_argument(
        "--auto-threshold",
        action="store_true",
        help=(
            "Calibrate threshold from calibration fuzzy-score quantile using "
            "target anomaly rate."
        ),
    )
    parser.add_argument(
        "--target-anomaly-rate",
        type=float,
        default=0.05,
        help="Target anomaly rate used when --auto-threshold is enabled.",
    )
    parser.add_argument(
        "--sensitivity-thresholds",
        type=str,
        default="0.40,0.50,0.60,0.70,0.80",
        help="Comma-separated thresholds for sensitivity sweep.",
    )
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="Run deterministic reproducibility and monotonicity checks.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if coverage or reproducibility checks fail.",
    )
    return parser.parse_args()


def validate_config(config: FuzzyConfig) -> None:
    a, b, c, d = config.breakpoints
    if not (a < b < c < d):
        raise ValueError(
            "breakpoints must be strictly increasing as a < b < c < d"
        )

    low_w, med_w, high_w = config.rule_weights
    if not (0.0 <= low_w <= med_w <= high_w <= 1.0):
        raise ValueError(
            "rule weights must satisfy 0 <= low <= medium <= high <= 1"
        )

    if not (0.0 < config.decision_threshold < 1.0):
        raise ValueError("decision-threshold must be in (0, 1)")


def parse_threshold_list(text: str) -> tuple[float, ...]:
    values: list[float] = []
    for token in text.split(","):
        stripped = token.strip()
        if not stripped:
            continue
        threshold = float(stripped)
        if not (0.0 < threshold < 1.0):
            raise ValueError("Each sensitivity threshold must be in (0, 1)")
        values.append(threshold)

    if not values:
        raise ValueError("At least one sensitivity threshold is required")

    unique_sorted = sorted(set(values))
    return tuple(unique_sorted)


def load_errors_from_npy(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Errors file not found: {path}")

    values = np.load(path, allow_pickle=False)
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    return values


def load_errors_from_csv(path: Path, column_name: str) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Errors CSV not found: {path}")

    values: list[float] = []
    with path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header")
        if column_name not in reader.fieldnames:
            raise ValueError(
                f"Column '{column_name}' not found in CSV header: {reader.fieldnames}"
            )

        for row in reader:
            raw = row.get(column_name, "").strip()
            if raw == "":
                raise ValueError(f"Empty error value in column '{column_name}'")
            values.append(float(raw))

    return np.asarray(values, dtype=np.float64)


def validate_error_vector(values: np.ndarray, role_name: str) -> np.ndarray:
    if values.size == 0:
        raise ValueError(f"{role_name} input is empty")
    if not np.isfinite(values).all():
        raise ValueError(f"{role_name} input contains non-finite values")
    if np.any(values < 0.0):
        raise ValueError(
            f"{role_name} input contains negative values, but reconstruction errors "
            "should be non-negative"
        )

    return values


def load_error_source(
    errors_npy: Path | None,
    errors_csv: Path | None,
    errors_column: str,
    role_name: str,
    required: bool,
) -> np.ndarray | None:
    source_count = int(errors_npy is not None) + int(errors_csv is not None)

    if source_count == 0:
        if required:
            raise ValueError(
                f"Provide one {role_name} source: --{role_name}-npy or --{role_name}-csv"
            )
        return None

    if source_count > 1:
        raise ValueError(
            f"Provide only one {role_name} source, not both .npy and .csv"
        )

    if errors_npy is not None:
        values = load_errors_from_npy(errors_npy)
    else:
        assert errors_csv is not None
        values = load_errors_from_csv(errors_csv, errors_column)

    return validate_error_vector(values, role_name=role_name)


def load_primary_errors(args: argparse.Namespace) -> np.ndarray:
    values = load_error_source(
        errors_npy=args.errors_npy,
        errors_csv=args.errors_csv,
        errors_column=args.errors_column,
        role_name="errors",
        required=True,
    )
    assert values is not None
    return values


def load_calibration_errors(args: argparse.Namespace) -> np.ndarray | None:
    return load_error_source(
        errors_npy=args.calibration_errors_npy,
        errors_csv=args.calibration_errors_csv,
        errors_column=args.calibration_errors_column,
        role_name="calibration-errors",
        required=False,
    )


def should_require_explicit_calibration(args: argparse.Namespace) -> bool:
    return args.auto_breakpoints or args.auto_threshold


def resolve_calibration_errors(
    args: argparse.Namespace,
    primary_errors: np.ndarray,
) -> tuple[np.ndarray, bool]:
    calibration_errors = load_calibration_errors(args)
    explicit_source = calibration_errors is not None

    if calibration_errors is None:
        calibration_errors = primary_errors

    if should_require_explicit_calibration(args) and not explicit_source:
        message = (
            "No explicit calibration errors provided; using primary errors for calibration. "
            "Prefer validation reconstruction errors to avoid optimistic calibration."
        )
        if args.strict:
            raise RuntimeError(
                "Strict mode requires explicit calibration error source when "
                "--auto-breakpoints or --auto-threshold is enabled"
            )
        print(f"WARNING: {message}")

    return calibration_errors, explicit_source


def load_errors(args: argparse.Namespace) -> np.ndarray:
    errors = load_primary_errors(args)

    # Compatibility shim so old call sites still work after introducing calibration inputs.
    return errors


def calibrate_threshold_from_scores(
    scores: np.ndarray,
    target_anomaly_rate: float,
) -> float:
    if not (0.0 < target_anomaly_rate < 1.0):
        raise ValueError("target-anomaly-rate must be in (0, 1)")

    quantile = 1.0 - target_anomaly_rate
    threshold = float(np.quantile(scores, quantile))
    return float(np.clip(threshold, 0.0, 1.0))


def threshold_sensitivity(
    scores: np.ndarray,
    thresholds: tuple[float, ...],
) -> list[ThresholdSensitivityPoint]:
    points: list[ThresholdSensitivityPoint] = []
    for threshold in thresholds:
        decisions = fuzzy_decision(scores, threshold)
        points.append(
            ThresholdSensitivityPoint(
                threshold=float(threshold),
                anomaly_count=int(decisions.sum()),
                anomaly_rate=float(decisions.mean()),
            )
        )

    return points
    if source_count == 0:
        raise ValueError("Provide one error source: --errors-npy or --errors-csv")
    if source_count > 1:
        raise ValueError("Provide only one error source, not both")

    if args.errors_npy is not None:
        errors = load_errors_from_npy(args.errors_npy)
    else:
        errors = load_errors_from_csv(args.errors_csv, args.errors_column)

    if errors.size == 0:
        raise ValueError("Error input is empty")
    if not np.isfinite(errors).all():
        raise ValueError("Error input contains non-finite values")
    if np.any(errors < 0.0):
        raise ValueError(
            "Error input contains negative values, but reconstruction error should be non-negative"
        )

    return errors


def calibrate_breakpoints(errors: np.ndarray) -> tuple[float, float, float, float]:
    quantiles = np.quantile(errors, [0.50, 0.75, 0.90, 0.97]).astype(np.float64)
    if np.allclose(quantiles, quantiles[0]):
        base = float(quantiles[0])
        return (base, base + 1e-6, base + 2e-6, base + 3e-6)

    adjusted = quantiles.copy()
    for idx in range(1, len(adjusted)):
        if adjusted[idx] <= adjusted[idx - 1]:
            adjusted[idx] = adjusted[idx - 1] + 1e-8

    return (
        float(adjusted[0]),
        float(adjusted[1]),
        float(adjusted[2]),
        float(adjusted[3]),
    )


def membership_low(errors: np.ndarray, a: float, b: float) -> np.ndarray:
    low = np.zeros_like(errors, dtype=np.float64)
    low = np.where(errors <= a, 1.0, low)
    slope = (b - errors) / (b - a + EPS)
    low = np.where((errors > a) & (errors < b), slope, low)
    return np.clip(low, 0.0, 1.0)


def membership_medium(errors: np.ndarray, b: float, c: float, d: float) -> np.ndarray:
    medium = np.zeros_like(errors, dtype=np.float64)

    left_slope = (errors - b) / (c - b + EPS)
    right_slope = (d - errors) / (d - c + EPS)

    medium = np.where((errors > b) & (errors < c), left_slope, medium)
    medium = np.where(errors == c, 1.0, medium)
    medium = np.where((errors > c) & (errors < d), right_slope, medium)

    return np.clip(medium, 0.0, 1.0)


def membership_high(errors: np.ndarray, c: float, d: float) -> np.ndarray:
    high = np.zeros_like(errors, dtype=np.float64)
    slope = (errors - c) / (d - c + EPS)

    high = np.where((errors > c) & (errors < d), slope, high)
    high = np.where(errors >= d, 1.0, high)

    return np.clip(high, 0.0, 1.0)


def fuzzy_anomaly_score(
    errors: np.ndarray,
    config: FuzzyConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    a, b, c, d = config.breakpoints
    low_w, med_w, high_w = config.rule_weights

    mu_low = membership_low(errors, a, b)
    mu_medium = membership_medium(errors, b, c, d)
    mu_high = membership_high(errors, c, d)

    numerator = (mu_low * low_w) + (mu_medium * med_w) + (mu_high * high_w)
    denominator = mu_low + mu_medium + mu_high + EPS

    score = numerator / denominator
    score = np.clip(score, 0.0, 1.0)

    return score, mu_low, mu_medium, mu_high


def fuzzy_decision(scores: np.ndarray, threshold: float) -> np.ndarray:
    return (scores >= threshold).astype(np.int64)


def reproducibility_and_monotonicity_check(config: FuzzyConfig) -> tuple[float, bool, bool]:
    deterministic_errors = np.linspace(0.0, 1.0, 4096, dtype=np.float64)

    score_1, _, _, _ = fuzzy_anomaly_score(deterministic_errors, config)
    score_2, _, _, _ = fuzzy_anomaly_score(deterministic_errors, config)

    max_abs_diff = float(np.max(np.abs(score_1 - score_2)))
    reproducibility_ok = bool(max_abs_diff <= 1e-12)

    score_deltas = np.diff(score_1)
    monotonicity_ok = bool(np.all(score_deltas >= -1e-10))

    return max_abs_diff, reproducibility_ok, monotonicity_ok


def save_outputs(
    output_dir: Path,
    output_prefix: str,
    errors: np.ndarray,
    mu_low: np.ndarray,
    mu_medium: np.ndarray,
    mu_high: np.ndarray,
    scores: np.ndarray,
    decisions: np.ndarray,
    result: FuzzyResult,
    config: FuzzyConfig,
    sensitivity_points: list[ThresholdSensitivityPoint],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f"{output_prefix}_scores.csv"
    score_npy_path = output_dir / f"{output_prefix}_scores.npy"
    decision_npy_path = output_dir / f"{output_prefix}_decision.npy"
    summary_json_path = output_dir / f"{output_prefix}_summary.json"
    sensitivity_csv_path = output_dir / f"{output_prefix}_threshold_sensitivity.csv"

    np.save(score_npy_path, scores.astype(np.float32), allow_pickle=False)
    np.save(decision_npy_path, decisions.astype(np.int8), allow_pickle=False)

    with csv_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(
            [
                "index",
                "error",
                "mu_low",
                "mu_medium",
                "mu_high",
                "fuzzy_score",
                "decision",
            ]
        )
        for idx in range(errors.shape[0]):
            writer.writerow(
                [
                    idx,
                    f"{float(errors[idx]):.10f}",
                    f"{float(mu_low[idx]):.10f}",
                    f"{float(mu_medium[idx]):.10f}",
                    f"{float(mu_high[idx]):.10f}",
                    f"{float(scores[idx]):.10f}",
                    int(decisions[idx]),
                ]
            )

    with sensitivity_csv_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["threshold", "anomaly_count", "anomaly_rate"])
        for point in sensitivity_points:
            writer.writerow(
                [
                    f"{point.threshold:.6f}",
                    point.anomaly_count,
                    f"{point.anomaly_rate:.10f}",
                ]
            )

    payload = {
        "config": {
            "breakpoints": list(config.breakpoints),
            "rule_weights": list(config.rule_weights),
            "decision_threshold": config.decision_threshold,
        },
        "result": asdict(result),
        "sensitivity": [asdict(point) for point in sensitivity_points],
        "fuzzy_score_formula": (
            "score = (mu_low*w_low + mu_medium*w_medium + mu_high*w_high) / "
            "(mu_low + mu_medium + mu_high + eps)"
        ),
        "decision_rule": "decision = 1 if score >= threshold else 0",
    }

    with summary_json_path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2)


def build_result(
    errors: np.ndarray,
    calibration_errors: np.ndarray,
    scores: np.ndarray,
    decisions: np.ndarray,
    breakpoint_source: str,
    threshold_source: str,
    sensitivity_points: list[ThresholdSensitivityPoint],
    max_abs_diff: float,
    reproducibility_ok: bool,
    monotonicity_ok: bool,
) -> FuzzyResult:
    sample_count = int(errors.shape[0])
    calibration_sample_count = int(calibration_errors.shape[0])
    score_count = int(scores.shape[0])
    decision_count = int(decisions.shape[0])

    sensitivity_rates = [point.anomaly_rate for point in sensitivity_points]
    sensitivity_min_rate = float(min(sensitivity_rates))
    sensitivity_max_rate = float(max(sensitivity_rates))

    return FuzzyResult(
        sample_count=sample_count,
        calibration_sample_count=calibration_sample_count,
        score_count=score_count,
        decision_count=decision_count,
        score_coverage_ok=bool(score_count == sample_count),
        decision_coverage_ok=bool(decision_count == sample_count),
        breakpoint_source=breakpoint_source,
        threshold_source=threshold_source,
        score_min=float(scores.min()),
        score_max=float(scores.max()),
        score_mean=float(scores.mean()),
        score_std=float(scores.std()),
        anomaly_count=int(decisions.sum()),
        anomaly_rate=float(decisions.mean()),
        sensitivity_min_rate=sensitivity_min_rate,
        sensitivity_max_rate=sensitivity_max_rate,
        reproducibility_max_abs_diff=max_abs_diff,
        reproducibility_ok=reproducibility_ok,
        monotonicity_ok=monotonicity_ok,
    )


def run_with_errors(args: argparse.Namespace, config: FuzzyConfig) -> int:
    errors = load_primary_errors(args)
    calibration_errors, explicit_calibration_source = resolve_calibration_errors(
        args,
        primary_errors=errors,
    )

    breakpoint_source = "manual"
    threshold_source = "manual"

    if args.auto_breakpoints:
        config = FuzzyConfig(
            breakpoints=calibrate_breakpoints(calibration_errors),
            rule_weights=config.rule_weights,
            decision_threshold=config.decision_threshold,
        )
        validate_config(config)
        breakpoint_source = (
            "validation-calibrated"
            if explicit_calibration_source
            else "primary-calibrated"
        )

    if args.auto_threshold:
        calibration_scores, _, _, _ = fuzzy_anomaly_score(calibration_errors, config)
        calibrated_threshold = calibrate_threshold_from_scores(
            calibration_scores,
            target_anomaly_rate=float(args.target_anomaly_rate),
        )
        config = FuzzyConfig(
            breakpoints=config.breakpoints,
            rule_weights=config.rule_weights,
            decision_threshold=calibrated_threshold,
        )
        validate_config(config)
        threshold_source = (
            "validation-quantile"
            if explicit_calibration_source
            else "primary-quantile"
        )

    scores, mu_low, mu_medium, mu_high = fuzzy_anomaly_score(errors, config)
    decisions = fuzzy_decision(scores, config.decision_threshold)

    sensitivity_thresholds = parse_threshold_list(args.sensitivity_thresholds)
    sensitivity_points = threshold_sensitivity(scores, sensitivity_thresholds)

    max_abs_diff, reproducibility_ok, monotonicity_ok = (
        reproducibility_and_monotonicity_check(config)
    )

    result = build_result(
        errors=errors,
        calibration_errors=calibration_errors,
        scores=scores,
        decisions=decisions,
        breakpoint_source=breakpoint_source,
        threshold_source=threshold_source,
        sensitivity_points=sensitivity_points,
        max_abs_diff=max_abs_diff,
        reproducibility_ok=reproducibility_ok,
        monotonicity_ok=monotonicity_ok,
    )

    save_outputs(
        output_dir=args.output_dir,
        output_prefix=args.output_prefix,
        errors=errors,
        mu_low=mu_low,
        mu_medium=mu_medium,
        mu_high=mu_high,
        scores=scores,
        decisions=decisions,
        result=result,
        config=config,
        sensitivity_points=sensitivity_points,
    )

    print("Fuzzy Threshold Report")
    print("=" * 22)
    print(f"Samples: {result.sample_count}")
    print(f"Calibration samples: {result.calibration_sample_count}")
    print(f"Breakpoint source: {result.breakpoint_source}")
    print(f"Threshold source: {result.threshold_source}")
    print(f"Breakpoints: {config.breakpoints}")
    print(f"Rule weights (low,medium,high): {config.rule_weights}")
    print(f"Decision threshold: {config.decision_threshold}")
    print(f"Score range: [{result.score_min:.6f}, {result.score_max:.6f}]")
    print(f"Anomaly count: {result.anomaly_count}")
    print(f"Anomaly rate: {result.anomaly_rate:.6f}")
    print(
        "Threshold sensitivity anomaly-rate range: "
        f"[{result.sensitivity_min_rate:.6f}, {result.sensitivity_max_rate:.6f}]"
    )
    print(f"Score coverage: {result.score_coverage_ok}")
    print(f"Decision coverage: {result.decision_coverage_ok}")
    print(f"Reproducibility max abs diff: {result.reproducibility_max_abs_diff:.3e}")
    print(f"Reproducibility stable: {result.reproducibility_ok}")
    print(f"Monotonicity stable: {result.monotonicity_ok}")
    print(f"Output directory: {args.output_dir}")

    if args.strict:
        if not result.score_coverage_ok or not result.decision_coverage_ok:
            raise RuntimeError(
                "Strict mode failed: fuzzy layer did not produce full sample coverage"
            )
        if not result.reproducibility_ok:
            raise RuntimeError(
                "Strict mode failed: fuzzy score reproducibility check did not pass"
            )
        if not result.monotonicity_ok:
            raise RuntimeError(
                "Strict mode failed: monotonicity check did not pass"
            )

    return 0


def run_self_check_only(args: argparse.Namespace, config: FuzzyConfig) -> int:
    max_abs_diff, reproducibility_ok, monotonicity_ok = (
        reproducibility_and_monotonicity_check(config)
    )

    print("Fuzzy Threshold Self-Check")
    print("=" * 27)
    print(f"Breakpoints: {config.breakpoints}")
    print(f"Rule weights (low,medium,high): {config.rule_weights}")
    print(f"Decision threshold: {config.decision_threshold}")
    print(f"Reproducibility max abs diff: {max_abs_diff:.3e}")
    print(f"Reproducibility stable: {reproducibility_ok}")
    print(f"Monotonicity stable: {monotonicity_ok}")

    if args.strict and not reproducibility_ok:
        raise RuntimeError(
            "Strict mode failed: self-check reproducibility did not pass"
        )

    return 0


def main() -> int:
    args = parse_args()

    if not (0.0 < float(args.target_anomaly_rate) < 1.0):
        raise ValueError("target-anomaly-rate must be in (0, 1)")

    breakpoints = parse_tuple_of_floats(args.breakpoints, expected_len=4, name="breakpoints")
    rule_weights = parse_tuple_of_floats(
        args.rule_weights,
        expected_len=3,
        name="rule-weights",
    )

    config = FuzzyConfig(
        breakpoints=(
            float(breakpoints[0]),
            float(breakpoints[1]),
            float(breakpoints[2]),
            float(breakpoints[3]),
        ),
        rule_weights=(
            float(rule_weights[0]),
            float(rule_weights[1]),
            float(rule_weights[2]),
        ),
        decision_threshold=float(args.decision_threshold),
    )
    validate_config(config)

    if args.self_check:
        return run_self_check_only(args, config)

    return run_with_errors(args, config)


if __name__ == "__main__":
    raise SystemExit(main())
