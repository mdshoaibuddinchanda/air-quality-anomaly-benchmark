from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path


EXPECTED_WINDOW_SIZE = 12
EXPECTED_FEATURE_COUNT = 11


@dataclass
class BaselineMetrics:
    name: str
    train_samples: int
    test_samples: int
    flattened_dim: int
    threshold: float
    predicted_anomalies: int
    predicted_anomaly_rate: float
    train_score_mean: float
    train_score_std: float
    test_score_mean: float
    test_score_std: float
    score_coverage_ok: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run recognized anomaly-detection baselines on pre-built windows and "
            "save per-sample scores plus summary metrics."
        )
    )
    parser.add_argument(
        "--train-npy",
        type=Path,
        default=Path("data") / "windows" / "train.npy",
        help="Train windows (.npy) path.",
    )
    parser.add_argument(
        "--test-npy",
        type=Path,
        default=Path("data") / "windows" / "test.npy",
        help="Test windows (.npy) path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data") / "baselines",
        help="Directory for baseline outputs.",
    )
    parser.add_argument(
        "--contamination",
        type=float,
        default=0.05,
        help="Assumed anomaly fraction used for thresholding and model hyperparameters.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for stochastic baselines.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any baseline does not produce a score for each test sample.",
    )
    return parser.parse_args()


def import_dependencies():
    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "numpy is required for baselines.py. Install it in the active environment."
        ) from exc

    try:
        from sklearn.ensemble import IsolationForest
        from sklearn.neighbors import LocalOutlierFactor
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import OneClassSVM
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "scikit-learn is required for baselines.py. "
            "Install scikit-learn in the active environment."
        ) from exc

    return np, IsolationForest, LocalOutlierFactor, OneClassSVM, StandardScaler


def load_windows(np, path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Window file not found: {path}")

    data = np.load(path, allow_pickle=False)
    if data.ndim != 3:
        raise ValueError(f"Expected 3D windows at {path}, got shape {data.shape}")

    if data.shape[1] != EXPECTED_WINDOW_SIZE or data.shape[2] != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            f"Unexpected window shape at {path}: {data.shape}. "
            f"Expected (*, {EXPECTED_WINDOW_SIZE}, {EXPECTED_FEATURE_COUNT})"
        )

    if not np.isfinite(data).all():
        raise ValueError(f"Non-finite values found in window file: {path}")

    return data.astype(np.float32, copy=False)


def flatten_windows(np, windows):
    return windows.reshape(windows.shape[0], windows.shape[1] * windows.shape[2])


def anomaly_score(np, model, x):
    if hasattr(model, "score_samples"):
        normality = model.score_samples(x)
    elif hasattr(model, "decision_function"):
        normality = model.decision_function(x)
    else:
        raise ValueError("Model does not expose score_samples or decision_function")

    scores = -normality
    if scores.ndim != 1:
        raise ValueError(f"Expected 1D anomaly scores, got shape {scores.shape}")
    return scores.astype(np.float64)


def save_baseline_outputs(
    np,
    output_dir: Path,
    baseline_name: str,
    test_scores,
    test_pred,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f"{baseline_name}_test_scores.csv"
    score_npy_path = output_dir / f"{baseline_name}_test_scores.npy"
    pred_npy_path = output_dir / f"{baseline_name}_test_pred.npy"

    np.save(score_npy_path, test_scores.astype(np.float32), allow_pickle=False)
    np.save(pred_npy_path, test_pred.astype(np.int8), allow_pickle=False)

    with csv_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["index", "anomaly_score", "predicted_label"])
        for idx, (score, pred) in enumerate(zip(test_scores, test_pred)):
            writer.writerow([idx, f"{float(score):.10f}", int(pred)])


def run_single_baseline(
    np,
    name: str,
    model,
    x_train,
    x_test,
    contamination: float,
):
    model.fit(x_train)

    train_scores = anomaly_score(np, model, x_train)
    test_scores = anomaly_score(np, model, x_test)

    threshold = float(np.quantile(train_scores, 1.0 - contamination))
    test_pred = (test_scores >= threshold).astype(np.int64)

    score_coverage_ok = bool(test_scores.shape[0] == x_test.shape[0])

    metrics = BaselineMetrics(
        name=name,
        train_samples=int(x_train.shape[0]),
        test_samples=int(x_test.shape[0]),
        flattened_dim=int(x_train.shape[1]),
        threshold=threshold,
        predicted_anomalies=int(test_pred.sum()),
        predicted_anomaly_rate=float(test_pred.mean()),
        train_score_mean=float(train_scores.mean()),
        train_score_std=float(train_scores.std()),
        test_score_mean=float(test_scores.mean()),
        test_score_std=float(test_scores.std()),
        score_coverage_ok=score_coverage_ok,
    )

    return metrics, test_scores, test_pred


def main() -> int:
    args = parse_args()

    if args.contamination <= 0.0 or args.contamination >= 0.5:
        raise ValueError("contamination must be in (0, 0.5)")

    np, IsolationForest, LocalOutlierFactor, OneClassSVM, StandardScaler = (
        import_dependencies()
    )

    train_windows = load_windows(np, args.train_npy)
    test_windows = load_windows(np, args.test_npy)

    x_train = flatten_windows(np, train_windows)
    x_test = flatten_windows(np, test_windows)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    baselines = {
        "isolation_forest": IsolationForest(
            n_estimators=300,
            contamination=args.contamination,
            random_state=args.random_state,
            n_jobs=-1,
        ),
        "one_class_svm": OneClassSVM(
            kernel="rbf",
            gamma="scale",
            nu=args.contamination,
        ),
        "local_outlier_factor": LocalOutlierFactor(
            n_neighbors=35,
            contamination=args.contamination,
            novelty=True,
        ),
    }

    all_metrics: list[BaselineMetrics] = []
    for name, model in baselines.items():
        metrics, test_scores, test_pred = run_single_baseline(
            np=np,
            name=name,
            model=model,
            x_train=x_train_scaled,
            x_test=x_test_scaled,
            contamination=args.contamination,
        )

        save_baseline_outputs(
            np=np,
            output_dir=args.output_dir,
            baseline_name=name,
            test_scores=test_scores,
            test_pred=test_pred,
        )
        all_metrics.append(metrics)

    metrics_path = args.output_dir / "baseline_metrics.json"
    summary_path = args.output_dir / "baseline_summary.csv"

    with metrics_path.open("w", encoding="utf-8") as output_file:
        json.dump([asdict(metric) for metric in all_metrics], output_file, indent=2)

    with summary_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(
            [
                "baseline",
                "train_samples",
                "test_samples",
                "flattened_dim",
                "threshold",
                "predicted_anomalies",
                "predicted_anomaly_rate",
                "train_score_mean",
                "train_score_std",
                "test_score_mean",
                "test_score_std",
                "score_coverage_ok",
            ]
        )
        for metric in all_metrics:
            writer.writerow(
                [
                    metric.name,
                    metric.train_samples,
                    metric.test_samples,
                    metric.flattened_dim,
                    f"{metric.threshold:.10f}",
                    metric.predicted_anomalies,
                    f"{metric.predicted_anomaly_rate:.10f}",
                    f"{metric.train_score_mean:.10f}",
                    f"{metric.train_score_std:.10f}",
                    f"{metric.test_score_mean:.10f}",
                    f"{metric.test_score_std:.10f}",
                    metric.score_coverage_ok,
                ]
            )

    all_coverage_ok = all(metric.score_coverage_ok for metric in all_metrics)

    print("Baseline Run Report")
    print("=" * 19)
    print(f"Train windows: {train_windows.shape}")
    print(f"Test windows: {test_windows.shape}")
    print(f"Flattened input dim: {x_train.shape[1]}")
    print(f"Output directory: {args.output_dir}")
    print("")

    for metric in all_metrics:
        print(
            f"- {metric.name}: test_scores={metric.test_samples}, "
            f"predicted_anomalies={metric.predicted_anomalies}, "
            f"anomaly_rate={metric.predicted_anomaly_rate:.4f}, "
            f"coverage_ok={metric.score_coverage_ok}"
        )

    print("")
    print(f"Coverage check across all baselines: {all_coverage_ok}")

    if args.strict and not all_coverage_ok:
        raise RuntimeError(
            "Strict mode failed: at least one baseline did not produce full test coverage"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
