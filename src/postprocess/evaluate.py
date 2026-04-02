from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    from src.postprocess.fuzzy_threshold import (
        FuzzyConfig,
        calibrate_breakpoints,
        calibrate_threshold_from_scores,
        fuzzy_anomaly_score,
        fuzzy_decision,
        validate_config,
    )
except ModuleNotFoundError:
    from fuzzy_threshold import (
        FuzzyConfig,
        calibrate_breakpoints,
        calibrate_threshold_from_scores,
        fuzzy_anomaly_score,
        fuzzy_decision,
        validate_config,
    )


BASELINE_NAMES = [
    "isolation_forest",
    "one_class_svm",
    "local_outlier_factor",
]


@dataclass
class MethodMetrics:
    method: str
    sample_count: int
    true_anomaly_count: int
    predicted_anomaly_count: int
    true_anomaly_rate: float
    predicted_anomaly_rate: float
    tp: int
    fp: int
    tn: int
    fn: int
    precision: float
    recall: float
    f1: float
    accuracy: float
    pr_auc: float


@dataclass
class EvalSummary:
    pseudo_label_protocol: str
    pseudo_label_source: str
    sample_count: int
    methods: list[MethodMetrics]
    fixed_threshold: float
    fuzzy_breakpoints: tuple[float, float, float, float]
    fuzzy_threshold: float
    metric_replay_match: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 9 evaluation: compute final metrics from saved anomaly scores/predictions "
            "and verify metrics are reproducible from exported prediction files."
        )
    )
    parser.add_argument(
        "--val-errors-npy",
        type=Path,
        default=Path("errors") / "val_errors.npy",
        help="Validation reconstruction errors (.npy) used for calibration.",
    )
    parser.add_argument(
        "--test-errors-npy",
        type=Path,
        default=Path("errors") / "test_errors.npy",
        help="Test reconstruction errors (.npy) used for final prediction.",
    )
    parser.add_argument(
        "--test-npy",
        type=Path,
        default=Path("data") / "windows" / "test.npy",
        help="Test windows file for count consistency checks.",
    )
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=Path("data") / "baselines",
        help="Directory holding baseline *_test_scores.npy and *_test_pred.npy files.",
    )
    parser.add_argument(
        "--labels-npy",
        type=Path,
        default=None,
        help="Optional explicit binary labels (.npy). If given, pseudo-labeling is skipped.",
    )
    parser.add_argument(
        "--pseudo-label-protocol",
        choices=["baseline-consensus", "val-quantile"],
        default="baseline-consensus",
        help=(
            "Pseudo-label strategy when labels are not provided. "
            "baseline-consensus uses baseline predictions; val-quantile uses validation-error quantile."
        ),
    )
    parser.add_argument(
        "--consensus-k",
        type=int,
        default=2,
        help="For baseline-consensus protocol: anomaly if at least k baselines predict anomaly.",
    )
    parser.add_argument(
        "--target-anomaly-rate",
        type=float,
        default=0.05,
        help="Target anomaly rate used for quantile threshold calibration.",
    )
    parser.add_argument(
        "--fixed-threshold",
        type=float,
        default=None,
        help="Optional manual fixed threshold on reconstruction error. If omitted, uses val quantile.",
    )
    parser.add_argument(
        "--fuzzy-rule-weights",
        type=str,
        default="0.0,0.5,1.0",
        help="Fuzzy rule weights low,medium,high.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data") / "evaluation",
        help="Directory for evaluation tables, predictions, and plots.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip plot generation if you only need metrics and prediction tables.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if core integrity/reproducibility checks do not pass.",
    )
    return parser.parse_args()


def parse_float_triplet(raw: str, name: str) -> tuple[float, float, float]:
    parts = [item.strip() for item in raw.split(",") if item.strip()]
    if len(parts) != 3:
        raise ValueError(f"{name} must contain exactly 3 comma-separated values")
    return (float(parts[0]), float(parts[1]), float(parts[2]))


def load_npy_1d(path: Path, name: str) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"{name} file not found: {path}")
    array = np.load(path, allow_pickle=False)
    array = np.asarray(array).reshape(-1)
    if array.size == 0:
        raise ValueError(f"{name} is empty: {path}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values: {path}")
    return array


def load_test_count(test_npy: Path) -> int:
    if not test_npy.exists():
        raise FileNotFoundError(f"test-npy not found: {test_npy}")
    data = np.load(test_npy, allow_pickle=False)
    if data.ndim != 3:
        raise ValueError(f"Expected 3D test windows at {test_npy}, got {data.shape}")
    return int(data.shape[0])


def ensure_binary(name: str, values: np.ndarray) -> np.ndarray:
    array = np.asarray(values).reshape(-1)
    unique = set(int(v) for v in np.unique(array))
    if not unique.issubset({0, 1}):
        raise ValueError(f"{name} must be binary (0/1). Found values: {sorted(unique)}")
    return array.astype(np.int8, copy=False)


def load_baseline_artifacts(baseline_dir: Path, expected_count: int) -> dict[str, dict[str, np.ndarray]]:
    artifacts: dict[str, dict[str, np.ndarray]] = {}

    for baseline in BASELINE_NAMES:
        score_path = baseline_dir / f"{baseline}_test_scores.npy"
        pred_path = baseline_dir / f"{baseline}_test_pred.npy"

        if not score_path.exists() or not pred_path.exists():
            continue

        score = load_npy_1d(score_path, f"{baseline} scores")
        pred = ensure_binary(f"{baseline} predictions", load_npy_1d(pred_path, f"{baseline} predictions"))

        if int(score.shape[0]) != expected_count:
            raise ValueError(
                f"{baseline} score count mismatch: expected {expected_count}, got {score.shape[0]}"
            )
        if int(pred.shape[0]) != expected_count:
            raise ValueError(
                f"{baseline} prediction count mismatch: expected {expected_count}, got {pred.shape[0]}"
            )

        artifacts[baseline] = {
            "score": score.astype(np.float64, copy=False),
            "pred": pred.astype(np.int8, copy=False),
        }

    return artifacts


def build_pseudo_labels(
    args: argparse.Namespace,
    val_errors: np.ndarray,
    test_errors: np.ndarray,
    baseline_artifacts: dict[str, dict[str, np.ndarray]],
) -> tuple[np.ndarray, str, str]:
    if args.labels_npy is not None:
        labels = ensure_binary("labels", load_npy_1d(args.labels_npy, "labels"))
        if labels.shape[0] != test_errors.shape[0]:
            raise ValueError(
                f"labels length mismatch: expected {test_errors.shape[0]}, got {labels.shape[0]}"
            )
        return labels, "explicit-labels", str(args.labels_npy)

    if args.pseudo_label_protocol == "baseline-consensus":
        if not baseline_artifacts:
            raise RuntimeError(
                "Pseudo-label protocol baseline-consensus requires baseline artifacts in baseline-dir"
            )

        k = int(args.consensus_k)
        n_baselines = len(baseline_artifacts)
        if k <= 0 or k > n_baselines:
            raise ValueError(
                f"consensus-k must be in [1, {n_baselines}] for available baselines"
            )

        stacked = np.stack([payload["pred"] for payload in baseline_artifacts.values()], axis=0)
        labels = (stacked.sum(axis=0) >= k).astype(np.int8)
        source = f"baseline-consensus(k={k},n={n_baselines})"
        return labels, "baseline-consensus", source

    if args.pseudo_label_protocol == "val-quantile":
        threshold = float(np.quantile(val_errors, 1.0 - float(args.target_anomaly_rate)))
        labels = (test_errors >= threshold).astype(np.int8)
        source = (
            "val-quantile(" +
            f"target_rate={float(args.target_anomaly_rate):.6f},threshold={threshold:.10f}" +
            ")"
        )
        return labels, "val-quantile", source

    raise ValueError(f"Unsupported pseudo-label protocol: {args.pseudo_label_protocol}")


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator / denominator)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, score: np.ndarray, method: str) -> MethodMetrics:
    y_true = ensure_binary("y_true", y_true)
    y_pred = ensure_binary("y_pred", y_pred)

    if y_true.shape[0] != y_pred.shape[0] or y_true.shape[0] != score.shape[0]:
        raise ValueError("Metric inputs must have the same length")

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1 = safe_divide(2.0 * precision * recall, precision + recall)
    accuracy = safe_divide(tp + tn, y_true.shape[0])

    pr_auc = float("nan")
    if len(np.unique(y_true)) > 1:
        try:
            from sklearn.metrics import average_precision_score

            pr_auc = float(average_precision_score(y_true, score))
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "scikit-learn is required for PR-AUC in evaluate.py"
            ) from exc

    return MethodMetrics(
        method=method,
        sample_count=int(y_true.shape[0]),
        true_anomaly_count=int(np.sum(y_true)),
        predicted_anomaly_count=int(np.sum(y_pred)),
        true_anomaly_rate=float(np.mean(y_true)),
        predicted_anomaly_rate=float(np.mean(y_pred)),
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        precision=precision,
        recall=recall,
        f1=f1,
        accuracy=accuracy,
        pr_auc=pr_auc,
    )


def calibrate_fixed_threshold(
    val_errors: np.ndarray,
    target_anomaly_rate: float,
    manual_threshold: float | None,
) -> float:
    if manual_threshold is not None:
        return float(manual_threshold)
    return float(np.quantile(val_errors, 1.0 - target_anomaly_rate))


def compute_fuzzy_predictions(
    val_errors: np.ndarray,
    test_errors: np.ndarray,
    target_anomaly_rate: float,
    rule_weights: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float, float], float]:
    breakpoints = calibrate_breakpoints(val_errors)
    base_config = FuzzyConfig(
        breakpoints=breakpoints,
        rule_weights=rule_weights,
        decision_threshold=0.60,
    )
    validate_config(base_config)

    val_scores, _, _, _ = fuzzy_anomaly_score(val_errors, base_config)
    threshold = calibrate_threshold_from_scores(
        val_scores,
        target_anomaly_rate=target_anomaly_rate,
    )

    final_config = FuzzyConfig(
        breakpoints=breakpoints,
        rule_weights=rule_weights,
        decision_threshold=float(threshold),
    )
    validate_config(final_config)

    test_scores, _, _, _ = fuzzy_anomaly_score(test_errors, final_config)
    test_pred = fuzzy_decision(test_scores, final_config.decision_threshold)
    return (
        test_scores.astype(np.float64, copy=False),
        test_pred.astype(np.int8, copy=False),
        breakpoints,
        float(final_config.decision_threshold),
    )


def save_predictions_csv(
    output_path: Path,
    y_true: np.ndarray,
    test_error_score: np.ndarray,
    method_scores: dict[str, np.ndarray],
    method_preds: dict[str, np.ndarray],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def fmt(value: float) -> str:
        # Keep near-full float precision so metrics can be replayed from CSV.
        return format(float(value), ".17g")

    method_order = sorted(method_scores.keys())
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.writer(output_file)

        header = ["index", "pseudo_label", "test_error"]
        for method in method_order:
            header.append(f"{method}_score")
            header.append(f"{method}_pred")
        writer.writerow(header)

        n = int(y_true.shape[0])
        for idx in range(n):
            row: list[Any] = [idx, int(y_true[idx]), fmt(float(test_error_score[idx]))]
            for method in method_order:
                row.append(fmt(float(method_scores[method][idx])))
                row.append(int(method_preds[method][idx]))
            writer.writerow(row)


def save_metrics(
    csv_path: Path,
    json_path: Path,
    summary: EvalSummary,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(
            [
                "method",
                "sample_count",
                "true_anomaly_count",
                "predicted_anomaly_count",
                "true_anomaly_rate",
                "predicted_anomaly_rate",
                "tp",
                "fp",
                "tn",
                "fn",
                "precision",
                "recall",
                "f1",
                "accuracy",
                "pr_auc",
            ]
        )
        for metric in summary.methods:
            writer.writerow(
                [
                    metric.method,
                    metric.sample_count,
                    metric.true_anomaly_count,
                    metric.predicted_anomaly_count,
                    f"{metric.true_anomaly_rate:.10f}",
                    f"{metric.predicted_anomaly_rate:.10f}",
                    metric.tp,
                    metric.fp,
                    metric.tn,
                    metric.fn,
                    f"{metric.precision:.10f}",
                    f"{metric.recall:.10f}",
                    f"{metric.f1:.10f}",
                    f"{metric.accuracy:.10f}",
                    f"{metric.pr_auc:.10f}" if not math.isnan(metric.pr_auc) else "nan",
                ]
            )

    with json_path.open("w", encoding="utf-8") as output_file:
        json.dump(asdict(summary), output_file, indent=2)


def load_predictions_for_replay(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found for replay check: {path}")

    with path.open("r", encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)
        if reader.fieldnames is None:
            raise ValueError("Prediction CSV is missing a header")

        rows = list(reader)

    if not rows:
        raise ValueError("Prediction CSV is empty")

    y_true = np.array([int(row["pseudo_label"]) for row in rows], dtype=np.int8)

    score_columns = [name for name in reader.fieldnames if name.endswith("_score")]
    pred_columns = [name for name in reader.fieldnames if name.endswith("_pred")]

    method_scores: dict[str, np.ndarray] = {}
    method_preds: dict[str, np.ndarray] = {}

    for score_col in score_columns:
        method = score_col[: -len("_score")]
        method_scores[method] = np.array([float(row[score_col]) for row in rows], dtype=np.float64)

    for pred_col in pred_columns:
        method = pred_col[: -len("_pred")]
        method_preds[method] = np.array([int(row[pred_col]) for row in rows], dtype=np.int8)

    return y_true, method_scores, method_preds


def verify_metric_replay(
    prediction_csv: Path,
    summary: EvalSummary,
    tolerance: float = 1e-7,
) -> bool:
    y_true, method_scores, method_preds = load_predictions_for_replay(prediction_csv)

    original = {metric.method: metric for metric in summary.methods}

    for method, original_metric in original.items():
        if method not in method_scores or method not in method_preds:
            return False

        replay_metric = compute_metrics(
            y_true=y_true,
            y_pred=method_preds[method],
            score=method_scores[method],
            method=method,
        )

        checks = [
            (original_metric.sample_count, replay_metric.sample_count),
            (original_metric.true_anomaly_count, replay_metric.true_anomaly_count),
            (original_metric.predicted_anomaly_count, replay_metric.predicted_anomaly_count),
            (original_metric.tp, replay_metric.tp),
            (original_metric.fp, replay_metric.fp),
            (original_metric.tn, replay_metric.tn),
            (original_metric.fn, replay_metric.fn),
        ]
        if any(a != b for a, b in checks):
            return False

        float_checks = [
            (original_metric.true_anomaly_rate, replay_metric.true_anomaly_rate),
            (original_metric.predicted_anomaly_rate, replay_metric.predicted_anomaly_rate),
            (original_metric.precision, replay_metric.precision),
            (original_metric.recall, replay_metric.recall),
            (original_metric.f1, replay_metric.f1),
            (original_metric.accuracy, replay_metric.accuracy),
        ]

        for lhs, rhs in float_checks:
            if abs(lhs - rhs) > tolerance:
                return False

        if math.isnan(original_metric.pr_auc) and math.isnan(replay_metric.pr_auc):
            pass
        elif abs(original_metric.pr_auc - replay_metric.pr_auc) > tolerance:
            return False

    return True


def create_plots(
    output_dir: Path,
    val_errors: np.ndarray,
    test_errors: np.ndarray,
    fixed_threshold: float,
    fuzzy_threshold: float,
    y_true: np.ndarray,
    method_scores: dict[str, np.ndarray],
) -> list[Path]:
    try:
        import matplotlib.pyplot as plt
        from sklearn.metrics import precision_recall_curve
    except ModuleNotFoundError:
        return []

    plot_paths: list[Path] = []

    # Plot 1: error distributions + calibrated thresholds.
    dist_path = output_dir / "error_distributions.png"
    fig1, ax1 = plt.subplots(figsize=(9, 5))
    ax1.hist(val_errors, bins=80, alpha=0.45, label="val_errors", density=True)
    ax1.hist(test_errors, bins=80, alpha=0.45, label="test_errors", density=True)
    ax1.axvline(fixed_threshold, color="black", linestyle="--", linewidth=1.5, label="fixed_threshold")
    ax1.axvline(fuzzy_threshold, color="tab:red", linestyle=":", linewidth=1.5, label="fuzzy_threshold")
    ax1.set_title("Reconstruction Error Distributions")
    ax1.set_xlabel("reconstruction error")
    ax1.set_ylabel("density")
    ax1.legend(loc="upper right")
    fig1.tight_layout()
    fig1.savefig(dist_path, dpi=140)
    plt.close(fig1)
    plot_paths.append(dist_path)

    # Plot 2: precision-recall curves from saved scores.
    pr_path = output_dir / "precision_recall_curves.png"
    fig2, ax2 = plt.subplots(figsize=(9, 5))

    for method, score in sorted(method_scores.items()):
        if len(np.unique(y_true)) <= 1:
            continue
        precision, recall, _ = precision_recall_curve(y_true, score)
        ax2.plot(recall, precision, linewidth=1.6, label=method)

    ax2.set_title("Precision-Recall Curves")
    ax2.set_xlabel("recall")
    ax2.set_ylabel("precision")
    ax2.set_xlim(0.0, 1.0)
    ax2.set_ylim(0.0, 1.05)
    ax2.legend(loc="lower left")
    fig2.tight_layout()
    fig2.savefig(pr_path, dpi=140)
    plt.close(fig2)
    plot_paths.append(pr_path)

    return plot_paths


def main() -> int:
    args = parse_args()

    if not (0.0 < float(args.target_anomaly_rate) < 1.0):
        raise ValueError("target-anomaly-rate must be in (0, 1)")

    rule_weights = parse_float_triplet(args.fuzzy_rule_weights, "fuzzy-rule-weights")
    if not (rule_weights[0] <= rule_weights[1] <= rule_weights[2]):
        raise ValueError("fuzzy-rule-weights must be non-decreasing")

    val_errors = load_npy_1d(args.val_errors_npy, "val_errors").astype(np.float64, copy=False)
    test_errors = load_npy_1d(args.test_errors_npy, "test_errors").astype(np.float64, copy=False)

    expected_test_count = load_test_count(args.test_npy)
    if int(test_errors.shape[0]) != expected_test_count:
        raise ValueError(
            "test error count mismatch with test windows: "
            f"errors={test_errors.shape[0]}, windows={expected_test_count}"
        )

    baseline_artifacts = load_baseline_artifacts(
        baseline_dir=args.baseline_dir,
        expected_count=expected_test_count,
    )

    y_true, protocol, source = build_pseudo_labels(
        args=args,
        val_errors=val_errors,
        test_errors=test_errors,
        baseline_artifacts=baseline_artifacts,
    )

    fixed_threshold = calibrate_fixed_threshold(
        val_errors=val_errors,
        target_anomaly_rate=float(args.target_anomaly_rate),
        manual_threshold=args.fixed_threshold,
    )
    fixed_pred = (test_errors >= fixed_threshold).astype(np.int8)

    fuzzy_scores, fuzzy_pred, fuzzy_breakpoints, fuzzy_threshold = compute_fuzzy_predictions(
        val_errors=val_errors,
        test_errors=test_errors,
        target_anomaly_rate=float(args.target_anomaly_rate),
        rule_weights=rule_weights,
    )

    method_scores: dict[str, np.ndarray] = {
        "fixed_error_threshold": test_errors.astype(np.float64, copy=False),
        "fuzzy_threshold": fuzzy_scores.astype(np.float64, copy=False),
    }
    method_preds: dict[str, np.ndarray] = {
        "fixed_error_threshold": fixed_pred.astype(np.int8, copy=False),
        "fuzzy_threshold": fuzzy_pred.astype(np.int8, copy=False),
    }

    for baseline_name, payload in baseline_artifacts.items():
        method_scores[baseline_name] = payload["score"].astype(np.float64, copy=False)
        method_preds[baseline_name] = payload["pred"].astype(np.int8, copy=False)

    metrics: list[MethodMetrics] = []
    for method in sorted(method_scores.keys()):
        metric = compute_metrics(
            y_true=y_true,
            y_pred=method_preds[method],
            score=method_scores[method],
            method=method,
        )
        metrics.append(metric)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions_csv = output_dir / "evaluation_predictions.csv"
    metrics_csv = output_dir / "evaluation_metrics.csv"
    metrics_json = output_dir / "evaluation_metrics.json"
    protocol_json = output_dir / "evaluation_protocol.json"

    save_predictions_csv(
        output_path=predictions_csv,
        y_true=y_true,
        test_error_score=test_errors,
        method_scores=method_scores,
        method_preds=method_preds,
    )

    summary = EvalSummary(
        pseudo_label_protocol=protocol,
        pseudo_label_source=source,
        sample_count=int(y_true.shape[0]),
        methods=metrics,
        fixed_threshold=float(fixed_threshold),
        fuzzy_breakpoints=fuzzy_breakpoints,
        fuzzy_threshold=float(fuzzy_threshold),
        metric_replay_match=False,
    )

    replay_ok = verify_metric_replay(prediction_csv=predictions_csv, summary=summary)
    summary.metric_replay_match = bool(replay_ok)

    save_metrics(csv_path=metrics_csv, json_path=metrics_json, summary=summary)

    with protocol_json.open("w", encoding="utf-8") as output_file:
        json.dump(
            {
                "pseudo_label_protocol": protocol,
                "pseudo_label_source": source,
                "target_anomaly_rate": float(args.target_anomaly_rate),
                "fixed_threshold": float(fixed_threshold),
                "fuzzy_breakpoints": list(fuzzy_breakpoints),
                "fuzzy_threshold": float(fuzzy_threshold),
                "methods": sorted(method_scores.keys()),
            },
            output_file,
            indent=2,
        )

    plot_paths: list[Path] = []
    if not args.skip_plots:
        plot_paths = create_plots(
            output_dir=output_dir,
            val_errors=val_errors,
            test_errors=test_errors,
            fixed_threshold=float(fixed_threshold),
            fuzzy_threshold=float(fuzzy_threshold),
            y_true=y_true,
            method_scores=method_scores,
        )

    print("Evaluation Report")
    print("=" * 17)
    print(f"Pseudo-label protocol: {protocol}")
    print(f"Pseudo-label source: {source}")
    print(f"Samples: {summary.sample_count}")
    print(f"Fixed threshold: {fixed_threshold:.10f}")
    print(f"Fuzzy breakpoints: {fuzzy_breakpoints}")
    print(f"Fuzzy threshold: {fuzzy_threshold:.10f}")
    print(f"Metric replay match: {summary.metric_replay_match}")
    print("")

    for metric in summary.methods:
        pr_auc_text = "nan" if math.isnan(metric.pr_auc) else f"{metric.pr_auc:.6f}"
        print(
            f"- {metric.method}: "
            f"pred_rate={metric.predicted_anomaly_rate:.4f}, "
            f"precision={metric.precision:.4f}, recall={metric.recall:.4f}, "
            f"f1={metric.f1:.4f}, accuracy={metric.accuracy:.4f}, pr_auc={pr_auc_text}"
        )

    print("")
    print(f"Predictions CSV: {predictions_csv}")
    print(f"Metrics CSV: {metrics_csv}")
    print(f"Metrics JSON: {metrics_json}")
    print(f"Protocol JSON: {protocol_json}")
    if plot_paths:
        print("Plot files:")
        for path in plot_paths:
            print(f"- {path}")
    else:
        print("Plot files: none (matplotlib not available or --skip-plots used)")

    if args.strict:
        if not summary.metric_replay_match:
            raise RuntimeError(
                "Strict mode failed: metrics cannot be replayed exactly from saved predictions"
            )
        if len(np.unique(y_true)) <= 1:
            raise RuntimeError(
                "Strict mode failed: pseudo/label vector has a single class, metrics are not informative"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
