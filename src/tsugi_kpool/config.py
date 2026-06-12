"""Configuration dataclass for K-Pool LoRA + Infinity sideband runtime.

Inherits LoraConfig field semantics from peft and adds the K-Pool routing
parameters plus the Infinity sideband + aggregator parameters.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from collections.abc import Iterable
from typing import Optional


_DEFAULT_RECOMMENDED_EPS_FLOOR = 1e-12


def _require_finite_at_least(name: str, value: float, minimum: float) -> None:
    if not math.isfinite(value) or value < minimum:
        raise ValueError(f"{name} must be finite and >= {minimum}; got {value!r}")


def _require_finite_greater_than(name: str, value: float, minimum: float) -> None:
    if not math.isfinite(value) or value <= minimum:
        raise ValueError(f"{name} must be finite and > {minimum}; got {value!r}")


def _require_finite_in_range(
    name: str, value: float, minimum: float, maximum: float
) -> None:
    if not math.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(
            f"{name} must be finite and in [{minimum}, {maximum}]; got {value!r}"
        )


def recommend_buffer_convergence_eps(
    variance_samples: Iterable[float],
    *,
    quantile: float = 0.5,
    floor: float = _DEFAULT_RECOMMENDED_EPS_FLOOR,
) -> float:
    """Recommend ``buffer_convergence_eps`` from observed variance samples.

    Non-finite samples are ignored. The returned value is the requested quantile
    of the remaining samples, linearly interpolated between neighboring samples,
    and never below ``floor``.
    """

    if not math.isfinite(quantile) or not 0.0 <= quantile <= 1.0:
        raise ValueError(f"quantile must be finite and in [0, 1]; got {quantile!r}")
    _require_finite_greater_than("floor", floor, 0.0)

    finite_samples: list[float] = []
    for sample in variance_samples:
        value = float(sample)
        if math.isfinite(value):
            finite_samples.append(value)

    if not finite_samples:
        return floor

    finite_samples.sort()
    if len(finite_samples) == 1:
        return max(finite_samples[0], floor)

    position = quantile * (len(finite_samples) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        eps = finite_samples[lower]
    else:
        weight = position - lower
        eps = finite_samples[lower] * (1.0 - weight) + finite_samples[upper] * weight
    return max(eps, floor)


@dataclass
class KPoolLoraConfig:
    """K-Pool LoRA SDK configuration.

    Fields fall into three groups:

    1. Standard LoRA fields (mirror peft.LoraConfig).
    2. K-Pool routing fields (n_adapters, k_active, routing_strategy).
    3. Infinity runtime fields (sideband, aggregator, buffer).

    The runtime fields default to values that mirror vanilla synchronous
    LoRA so that turning the SDK on does not silently change behavior. Set
    sideband_enabled=True and aggregation_mode="buffer_convergence" to
    activate the plesiochronous path.
    """

    # --- standard LoRA fields ---
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    target_modules: tuple[str, ...] = ("q_proj", "v_proj")
    bias: str = "none"

    # --- K-Pool routing (App. 64/060,315) ---
    n_adapters: int = 8
    k_active: int = 2
    routing_strategy: str = "round_robin"  # "round_robin" | "loss_aware" | "random"
    routing_seed: int = 0

    # --- Infinity runtime (App. 64/055,093) ---
    sideband_enabled: bool = False
    # Loopback by default (secure-by-default); set to the rank's reachable
    # NIC address for real multi-node runs.
    sideband_addr: str = "tcp://127.0.0.1:51820"
    sideband_peers: tuple[str, ...] = field(default_factory=tuple)
    sideband_heartbeat_ms: int = 50
    max_drift_ms: int = 250

    aggregation_mode: str = "synchronous"  # "synchronous" | "buffer_convergence"
    buffer_capacity: int = 32
    buffer_convergence_eps: float = 1e-3

    diagnostics_dir: Optional[str] = None

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.r)) or self.r < 1:
            raise ValueError(f"r must be >= 1; got {self.r}")
        _require_finite_in_range("lora_dropout", float(self.lora_dropout), 0.0, 1.0)
        if not (1 <= self.k_active <= self.n_adapters):
            raise ValueError(
                f"k_active must satisfy 1 <= k_active <= n_adapters; "
                f"got k_active={self.k_active}, n_adapters={self.n_adapters}"
            )
        if (
            not math.isfinite(float(self.sideband_heartbeat_ms))
            or self.sideband_heartbeat_ms < 1
        ):
            raise ValueError(
                f"sideband_heartbeat_ms must be >= 1; "
                f"got {self.sideband_heartbeat_ms}"
            )
        _require_finite_at_least("max_drift_ms", float(self.max_drift_ms), 0.0)
        if not math.isfinite(float(self.buffer_capacity)) or self.buffer_capacity < 2:
            raise ValueError(f"buffer_capacity must be >= 2; got {self.buffer_capacity}")
        _require_finite_greater_than(
            "buffer_convergence_eps", float(self.buffer_convergence_eps), 0.0
        )
        if self.routing_strategy not in {"round_robin", "loss_aware", "random"}:
            raise ValueError(f"unknown routing_strategy: {self.routing_strategy}")
        if self.aggregation_mode not in {"synchronous", "buffer_convergence"}:
            raise ValueError(f"unknown aggregation_mode: {self.aggregation_mode}")
        if self.sideband_enabled and self.aggregation_mode == "synchronous":
            raise ValueError(
                "sideband_enabled=True requires aggregation_mode='buffer_convergence'"
            )
