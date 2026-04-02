from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

try:
    from src.modeling.model_autoencoder import TinyAutoencoder, count_parameters
except ModuleNotFoundError:
    from model_autoencoder import TinyAutoencoder, count_parameters


@dataclass
class EpochLog:
    epoch: int
    train_loss: float
    val_loss: float


@dataclass
class TrainSummary:
    train_samples: int
    val_samples: int
    window_size: int
    feature_count: int
    input_dim: int
    parameter_count: int
    best_epoch: int
    best_val_loss: float
    final_train_loss: float
    final_val_loss: float
    train_loss_decreased: bool
    val_loss_improved: bool
    checkpoint_reload_ok: bool
    reload_max_abs_diff: float


def parse_hidden_sizes(text: str) -> tuple[int, ...]:
    values: list[int] = []
    for token in text.split(","):
        stripped = token.strip()
        if not stripped:
            continue
        values.append(int(stripped))

    if not values:
        raise ValueError("At least one hidden size is required")

    if any(value <= 0 for value in values):
        raise ValueError("All hidden sizes must be positive")

    return tuple(values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the tiny autoencoder on train windows and monitor validation loss "
            "without touching test windows."
        )
    )
    parser.add_argument(
        "--train-npy",
        type=Path,
        default=Path("data") / "windows" / "train.npy",
        help="Path to train windows (.npy).",
    )
    parser.add_argument(
        "--val-npy",
        type=Path,
        default=Path("data") / "windows" / "val.npy",
        help="Path to validation windows (.npy).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data") / "training",
        help="Directory for checkpoints and history artifacts.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="tiny_ae",
        help="Run name used as output file prefix.",
    )
    parser.add_argument(
        "--hidden-sizes",
        type=str,
        default="64,32",
        help="Comma-separated hidden sizes.",
    )
    parser.add_argument(
        "--activation",
        type=str,
        default="relu",
        help="Activation function passed to TinyAutoencoder.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Mini-batch size.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Optimizer learning rate.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="Adam weight decay.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global random seed.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device override: auto, cpu, cuda.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if core verification checks do not pass.",
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
        raise ValueError(f"Expected 3D array at {path}, got shape {data.shape}")

    if not np.isfinite(data).all():
        raise ValueError(f"Non-finite values found in {path}")

    return data.astype(np.float32, copy=False)


def make_loader(
    windows: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    tensor = torch.from_numpy(windows)
    dataset = TensorDataset(tensor)
    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        generator=generator,
    )


def epoch_pass(
    model: TinyAutoencoder,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> float:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_count = 0

    for (batch,) in loader:
        batch = batch.to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        reconstructed = model(batch)
        loss = criterion(reconstructed, batch)

        if is_train:
            loss.backward()
            optimizer.step()

        batch_size = int(batch.shape[0])
        total_loss += float(loss.detach().item()) * batch_size
        total_count += batch_size

    if total_count == 0:
        raise RuntimeError("Empty loader encountered during epoch pass")

    return total_loss / total_count


def evaluate_reload_consistency(
    model: TinyAutoencoder,
    checkpoint_path: Path,
    sample_batch: torch.Tensor,
    device: torch.device,
) -> tuple[bool, float]:
    state = torch.load(checkpoint_path, map_location=device)

    reloaded_model = TinyAutoencoder(
        input_dim=model.input_dim,
        hidden_sizes=model.hidden_sizes,
        activation=model.activation,
    ).to(device)
    reloaded_model.load_state_dict(state["model_state_dict"])
    reloaded_model.eval()

    model.eval()
    with torch.no_grad():
        original_out = model(sample_batch.to(device)).detach().cpu()
        reload_out = reloaded_model(sample_batch.to(device)).detach().cpu()

    max_abs_diff = float(torch.max(torch.abs(original_out - reload_out)).item())
    return bool(max_abs_diff <= 1e-8), max_abs_diff


def save_history_csv(path: Path, history: list[EpochLog]) -> None:
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["epoch", "train_loss", "val_loss"])
        for entry in history:
            writer.writerow(
                [
                    entry.epoch,
                    f"{entry.train_loss:.10f}",
                    f"{entry.val_loss:.10f}",
                ]
            )


def main() -> int:
    args = parse_args()

    if args.batch_size <= 0:
        raise ValueError("batch-size must be > 0")
    if args.epochs <= 0:
        raise ValueError("epochs must be > 0")
    if args.learning_rate <= 0.0:
        raise ValueError("learning-rate must be > 0")
    if args.weight_decay < 0.0:
        raise ValueError("weight-decay must be >= 0")

    set_seed(args.seed)
    device = choose_device(args.device)

    train_windows = load_windows(args.train_npy)
    val_windows = load_windows(args.val_npy)

    if train_windows.shape[1:] != val_windows.shape[1:]:
        raise ValueError(
            "Shape mismatch between train and val windows: "
            f"{train_windows.shape} vs {val_windows.shape}"
        )

    if train_windows.shape[0] == 0 or val_windows.shape[0] == 0:
        raise ValueError("Train/val windows must be non-empty")

    window_size = int(train_windows.shape[1])
    feature_count = int(train_windows.shape[2])
    input_dim = window_size * feature_count

    hidden_sizes = parse_hidden_sizes(args.hidden_sizes)
    model = TinyAutoencoder(
        input_dim=input_dim,
        hidden_sizes=hidden_sizes,
        activation=args.activation,
    ).to(device)

    parameter_count = count_parameters(model)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    train_loader = make_loader(
        windows=train_windows,
        batch_size=args.batch_size,
        shuffle=True,
        seed=args.seed,
    )
    val_loader = make_loader(
        windows=val_windows,
        batch_size=args.batch_size,
        shuffle=False,
        seed=args.seed,
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = output_dir / f"{args.run_name}_best.pt"
    history_csv_path = output_dir / f"{args.run_name}_history.csv"
    summary_json_path = output_dir / f"{args.run_name}_summary.json"

    history: list[EpochLog] = []
    best_val_loss = float("inf")
    best_epoch = -1

    for epoch_idx in range(1, args.epochs + 1):
        train_loss = epoch_pass(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        val_loss = epoch_pass(
            model=model,
            loader=val_loader,
            criterion=criterion,
            optimizer=None,
            device=device,
        )

        history.append(EpochLog(epoch=epoch_idx, train_loss=train_loss, val_loss=val_loss))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch_idx
            torch.save(
                {
                    "epoch": epoch_idx,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "window_size": window_size,
                    "feature_count": feature_count,
                    "input_dim": input_dim,
                    "hidden_sizes": list(hidden_sizes),
                    "activation": args.activation,
                    "seed": args.seed,
                },
                checkpoint_path,
            )

    if best_epoch < 0:
        raise RuntimeError("Training finished without a best checkpoint")

    if not checkpoint_path.exists():
        raise RuntimeError("Expected checkpoint was not saved")

    save_history_csv(history_csv_path, history)

    train_loss_decreased = history[-1].train_loss < history[0].train_loss
    val_loss_improved = best_val_loss < history[0].val_loss

    sample_batch = torch.from_numpy(val_windows[: min(8, val_windows.shape[0])])
    checkpoint_reload_ok, reload_max_abs_diff = evaluate_reload_consistency(
        model=model,
        checkpoint_path=checkpoint_path,
        sample_batch=sample_batch,
        device=device,
    )

    summary = TrainSummary(
        train_samples=int(train_windows.shape[0]),
        val_samples=int(val_windows.shape[0]),
        window_size=window_size,
        feature_count=feature_count,
        input_dim=input_dim,
        parameter_count=parameter_count,
        best_epoch=best_epoch,
        best_val_loss=float(best_val_loss),
        final_train_loss=float(history[-1].train_loss),
        final_val_loss=float(history[-1].val_loss),
        train_loss_decreased=bool(train_loss_decreased),
        val_loss_improved=bool(val_loss_improved),
        checkpoint_reload_ok=bool(checkpoint_reload_ok),
        reload_max_abs_diff=float(reload_max_abs_diff),
    )

    with summary_json_path.open("w", encoding="utf-8") as output_file:
        json.dump(asdict(summary), output_file, indent=2)

    print("Training Report")
    print("=" * 15)
    print(f"Device: {device}")
    print(f"Train windows: {train_windows.shape}")
    print(f"Val windows: {val_windows.shape}")
    print(f"Model input_dim: {input_dim}")
    print(f"Model parameters: {parameter_count}")
    print(f"Best epoch: {best_epoch}")
    print(f"Best val loss: {best_val_loss:.8f}")
    print(f"Final train loss: {history[-1].train_loss:.8f}")
    print(f"Final val loss: {history[-1].val_loss:.8f}")
    print(f"Train loss decreased: {train_loss_decreased}")
    print(f"Val loss improved vs epoch1: {val_loss_improved}")
    print(f"Checkpoint reload ok: {checkpoint_reload_ok}")
    print(f"Reload max abs diff: {reload_max_abs_diff:.3e}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"History CSV: {history_csv_path}")
    print(f"Summary JSON: {summary_json_path}")

    if args.strict:
        if not train_loss_decreased:
            raise RuntimeError("Strict mode failed: train loss did not decrease")
        if not checkpoint_reload_ok:
            raise RuntimeError("Strict mode failed: checkpoint reload consistency failed")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
