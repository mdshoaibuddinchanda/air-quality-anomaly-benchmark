from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn


SUPPORTED_ACTIVATIONS = {"relu", "gelu", "tanh", "elu", "leaky_relu"}


@dataclass
class ModelCheckResult:
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    flat_input_shape: tuple[int, ...]
    flat_output_shape: tuple[int, ...]
    bottleneck_dim: int
    bottleneck_ratio: float
    bottleneck_too_aggressive: bool
    identity_mapping_risk: bool
    parameter_count: int
    is_small_model: bool
    training_check_skipped: bool
    training_initial_loss: float | None
    training_final_loss: float | None
    training_relative_improvement: float | None
    training_stable: bool | None


def make_activation(name: str) -> nn.Module:
    normalized = name.lower().strip()
    if normalized == "relu":
        return nn.ReLU()
    if normalized == "gelu":
        return nn.GELU()
    if normalized == "tanh":
        return nn.Tanh()
    if normalized == "elu":
        return nn.ELU()
    if normalized == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.01)

    raise ValueError(
        f"Unsupported activation '{name}'. Supported: {sorted(SUPPORTED_ACTIVATIONS)}"
    )


def build_mlp(dims: Sequence[int], activation: str) -> nn.Sequential:
    if len(dims) < 2:
        raise ValueError("At least two dimensions are required to build an MLP")

    layers: list[nn.Module] = []
    for idx in range(len(dims) - 1):
        in_dim = int(dims[idx])
        out_dim = int(dims[idx + 1])
        if in_dim <= 0 or out_dim <= 0:
            raise ValueError("All layer dimensions must be positive")

        layers.append(nn.Linear(in_dim, out_dim))
        if idx < len(dims) - 2:
            layers.append(make_activation(activation))

    return nn.Sequential(*layers)


class TinyAutoencoder(nn.Module):
    """Tiny MLP autoencoder for fixed-size flattened windows."""

    def __init__(
        self,
        input_dim: int,
        hidden_sizes: Sequence[int] = (64, 32),
        activation: str = "relu",
    ) -> None:
        super().__init__()

        if input_dim <= 0:
            raise ValueError("input_dim must be > 0")
        if not hidden_sizes:
            raise ValueError("hidden_sizes cannot be empty")
        if any(size <= 0 for size in hidden_sizes):
            raise ValueError("All hidden sizes must be > 0")

        self.input_dim = int(input_dim)
        self.hidden_sizes = tuple(int(size) for size in hidden_sizes)
        self.activation = activation

        encoder_dims = [self.input_dim, *self.hidden_sizes]
        decoder_dims = [
            self.hidden_sizes[-1],
            *reversed(self.hidden_sizes[:-1]),
            self.input_dim,
        ]

        self.encoder = build_mlp(encoder_dims, activation=self.activation)
        self.decoder = build_mlp(decoder_dims, activation=self.activation)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        flat = self._flatten_input(x)
        return self.encoder(flat)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        reshape_to_sequence = x.dim() == 3
        batch_size = x.shape[0]

        flat = self._flatten_input(x)
        latent = self.encoder(flat)
        reconstructed_flat = self.decoder(latent)

        if reshape_to_sequence:
            window_size = x.shape[1]
            feature_count = x.shape[2]
            return reconstructed_flat.reshape(batch_size, window_size, feature_count)

        return reconstructed_flat

    def reconstruction_error(
        self,
        x: torch.Tensor,
        reduction: str = "none",
    ) -> torch.Tensor:
        reconstructed = self.forward(x)
        squared_error = (reconstructed - x) ** 2

        if x.dim() == 3:
            per_sample = squared_error.mean(dim=(1, 2))
        elif x.dim() == 2:
            per_sample = squared_error.mean(dim=1)
        else:
            raise ValueError("Input tensor must be 2D or 3D")

        if reduction == "none":
            return per_sample
        if reduction == "mean":
            return per_sample.mean()

        raise ValueError("reduction must be 'none' or 'mean'")

    def _flatten_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() not in (2, 3):
            raise ValueError("Input tensor must be 2D (batch, dim) or 3D (batch, time, feat)")

        batch_size = x.shape[0]
        flat = x.reshape(batch_size, -1)
        if flat.shape[1] != self.input_dim:
            raise ValueError(
                f"Flattened input dimension mismatch: got {flat.shape[1]}, "
                f"expected {self.input_dim}"
            )
        return flat


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def parse_hidden_sizes(text: str) -> tuple[int, ...]:
    values: list[int] = []
    for token in text.split(","):
        stripped = token.strip()
        if not stripped:
            continue
        values.append(int(stripped))

    if not values:
        raise ValueError("At least one hidden size is required")

    return tuple(values)


def training_stability_check(
    model: TinyAutoencoder,
    window_size: int,
    num_features: int,
    num_samples: int,
    epochs: int,
    learning_rate: float,
    min_relative_improvement: float,
    seed: int = 42,
) -> tuple[float, float, float, bool]:
    if num_samples <= 0:
        raise ValueError("num_samples must be > 0")
    if epochs <= 0:
        raise ValueError("epochs must be > 0")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be > 0")

    torch.manual_seed(seed)
    model.train()

    latent_basis_dim = min(8, num_features)
    latent = torch.randn(num_samples, window_size, latent_basis_dim)
    projector = torch.randn(latent_basis_dim, num_features)
    data = torch.tanh(torch.einsum("btl,lf->btf", latent, projector))
    data = data + 0.01 * torch.randn_like(data)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    losses: list[float] = []

    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        reconstructed = model(data)
        loss = criterion(reconstructed, data)
        if not torch.isfinite(loss):
            return float("inf"), float("inf"), 0.0, False

        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().item()))

    initial_loss = losses[0]
    final_loss = losses[-1]
    relative_improvement = (initial_loss - final_loss) / max(initial_loss, 1e-8)
    stable = (
        final_loss < initial_loss
        and relative_improvement >= min_relative_improvement
        and final_loss > 0.0
    )

    return initial_loss, final_loss, relative_improvement, stable


def self_check(
    window_size: int = 12,
    num_features: int = 11,
    hidden_sizes: Sequence[int] = (64, 32),
    activation: str = "relu",
    batch_size: int = 8,
    max_params: int = 50_000,
    min_bottleneck_ratio: float = 0.08,
    max_bottleneck_ratio: float = 0.85,
    min_bottleneck_dim: int = 8,
    run_training_check: bool = True,
    training_samples: int = 512,
    training_epochs: int = 60,
    training_lr: float = 1e-3,
    min_training_improvement: float = 0.10,
) -> ModelCheckResult:
    if window_size <= 0 or num_features <= 0 or batch_size <= 0:
        raise ValueError("window_size, num_features, and batch_size must be > 0")
    if min_bottleneck_ratio <= 0.0 or min_bottleneck_ratio >= 1.0:
        raise ValueError("min_bottleneck_ratio must be in (0, 1)")
    if max_bottleneck_ratio <= 0.0 or max_bottleneck_ratio > 1.0:
        raise ValueError("max_bottleneck_ratio must be in (0, 1]")
    if min_bottleneck_ratio >= max_bottleneck_ratio:
        raise ValueError("min_bottleneck_ratio must be smaller than max_bottleneck_ratio")
    if min_bottleneck_dim <= 0:
        raise ValueError("min_bottleneck_dim must be > 0")
    if min_training_improvement <= 0.0:
        raise ValueError("min_training_improvement must be > 0")

    input_dim = window_size * num_features
    model = TinyAutoencoder(
        input_dim=input_dim,
        hidden_sizes=hidden_sizes,
        activation=activation,
    )

    x_seq = torch.randn(batch_size, window_size, num_features)
    y_seq = model(x_seq)
    if y_seq.shape != x_seq.shape:
        raise RuntimeError(
            f"Sequence shape mismatch: input={tuple(x_seq.shape)}, output={tuple(y_seq.shape)}"
        )

    x_flat = x_seq.reshape(batch_size, input_dim)
    y_flat = model(x_flat)
    if y_flat.shape != x_flat.shape:
        raise RuntimeError(
            f"Flat shape mismatch: input={tuple(x_flat.shape)}, output={tuple(y_flat.shape)}"
        )

    parameter_count = count_parameters(model)
    is_small_model = parameter_count <= max_params
    bottleneck_dim = int(hidden_sizes[-1])
    bottleneck_ratio = bottleneck_dim / float(input_dim)
    bottleneck_too_aggressive = (
        bottleneck_dim < min_bottleneck_dim or bottleneck_ratio < min_bottleneck_ratio
    )
    identity_mapping_risk = (
        bottleneck_dim >= input_dim or bottleneck_ratio > max_bottleneck_ratio
    )

    training_check_skipped = not run_training_check
    training_initial_loss: float | None = None
    training_final_loss: float | None = None
    training_relative_improvement: float | None = None
    training_stable: bool | None = None

    if run_training_check:
        (
            training_initial_loss,
            training_final_loss,
            training_relative_improvement,
            training_stable,
        ) = training_stability_check(
            model=model,
            window_size=window_size,
            num_features=num_features,
            num_samples=training_samples,
            epochs=training_epochs,
            learning_rate=training_lr,
            min_relative_improvement=min_training_improvement,
        )

    return ModelCheckResult(
        input_shape=tuple(x_seq.shape),
        output_shape=tuple(y_seq.shape),
        flat_input_shape=tuple(x_flat.shape),
        flat_output_shape=tuple(y_flat.shape),
        bottleneck_dim=bottleneck_dim,
        bottleneck_ratio=bottleneck_ratio,
        bottleneck_too_aggressive=bottleneck_too_aggressive,
        identity_mapping_risk=identity_mapping_risk,
        parameter_count=parameter_count,
        is_small_model=is_small_model,
        training_check_skipped=training_check_skipped,
        training_initial_loss=training_initial_loss,
        training_final_loss=training_final_loss,
        training_relative_improvement=training_relative_improvement,
        training_stable=training_stable,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Define and sanity-check a tiny PyTorch autoencoder for window "
            "reconstruction-based anomaly detection."
        )
    )
    parser.add_argument("--window-size", type=int, default=12, help="Window length")
    parser.add_argument("--num-features", type=int, default=11, help="Feature count")
    parser.add_argument(
        "--hidden-sizes",
        type=str,
        default="64,32",
        help="Comma-separated hidden sizes, e.g. 64,32",
    )
    parser.add_argument(
        "--activation",
        type=str,
        default="relu",
        help=f"Activation function: {sorted(SUPPORTED_ACTIVATIONS)}",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for self-check forward pass",
    )
    parser.add_argument(
        "--max-params",
        type=int,
        default=50_000,
        help="Maximum parameter budget considered tiny",
    )
    parser.add_argument(
        "--min-bottleneck-ratio",
        type=float,
        default=0.08,
        help="Minimum acceptable bottleneck/input ratio.",
    )
    parser.add_argument(
        "--max-bottleneck-ratio",
        type=float,
        default=0.85,
        help="Maximum acceptable bottleneck/input ratio to avoid identity mapping.",
    )
    parser.add_argument(
        "--min-bottleneck-dim",
        type=int,
        default=8,
        help="Minimum acceptable bottleneck dimension.",
    )
    parser.add_argument(
        "--skip-training-check",
        action="store_true",
        help="Skip optimization smoke test (not recommended for strict validation).",
    )
    parser.add_argument(
        "--training-samples",
        type=int,
        default=512,
        help="Synthetic sample count for training stability smoke test.",
    )
    parser.add_argument(
        "--training-epochs",
        type=int,
        default=60,
        help="Epoch count for training stability smoke test.",
    )
    parser.add_argument(
        "--training-lr",
        type=float,
        default=1e-3,
        help="Learning rate for training stability smoke test.",
    )
    parser.add_argument(
        "--min-training-improvement",
        type=float,
        default=0.10,
        help="Minimum relative loss improvement required for stable training.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Fail if tiny-budget, architecture-sanity, or training-stability checks fail."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    hidden_sizes = parse_hidden_sizes(args.hidden_sizes)

    result = self_check(
        window_size=args.window_size,
        num_features=args.num_features,
        hidden_sizes=hidden_sizes,
        activation=args.activation,
        batch_size=args.batch_size,
        max_params=args.max_params,
        min_bottleneck_ratio=args.min_bottleneck_ratio,
        max_bottleneck_ratio=args.max_bottleneck_ratio,
        min_bottleneck_dim=args.min_bottleneck_dim,
        run_training_check=not args.skip_training_check,
        training_samples=args.training_samples,
        training_epochs=args.training_epochs,
        training_lr=args.training_lr,
        min_training_improvement=args.min_training_improvement,
    )

    print("Tiny Autoencoder Self-Check")
    print("=" * 27)
    print(f"Input shape: {result.input_shape}")
    print(f"Output shape: {result.output_shape}")
    print(f"Flat input shape: {result.flat_input_shape}")
    print(f"Flat output shape: {result.flat_output_shape}")
    print(f"Bottleneck dim: {result.bottleneck_dim}")
    print(f"Bottleneck ratio: {result.bottleneck_ratio:.4f}")
    print(f"Bottleneck too aggressive: {result.bottleneck_too_aggressive}")
    print(f"Identity mapping risk: {result.identity_mapping_risk}")
    print(f"Parameter count: {result.parameter_count}")
    print(f"Within tiny-model budget: {result.is_small_model}")

    if result.training_check_skipped:
        print("Training stability check: skipped")
    else:
        print(
            f"Training initial loss: {result.training_initial_loss:.6f}"
        )
        print(
            f"Training final loss: {result.training_final_loss:.6f}"
        )
        print(
            "Training relative improvement: "
            f"{result.training_relative_improvement:.4f}"
        )
        print(f"Training stable: {result.training_stable}")

    if args.strict and not result.is_small_model:
        raise RuntimeError(
            "Strict mode failed: model exceeds max-params tiny-model budget"
        )

    if args.strict and result.bottleneck_too_aggressive:
        raise RuntimeError(
            "Strict mode failed: bottleneck is too aggressive for stable reconstruction"
        )

    if args.strict and result.identity_mapping_risk:
        raise RuntimeError(
            "Strict mode failed: bottleneck is too wide and risks identity mapping"
        )

    if args.strict and not result.training_check_skipped and not result.training_stable:
        raise RuntimeError(
            "Strict mode failed: training stability check did not meet minimum improvement"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
