"""Configuration dataclass for K-Pool LoRA + Infinity sideband runtime.

Inherits LoraConfig field semantics from peft and adds the K-Pool routing
parameters plus the Infinity sideband + aggregator parameters.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


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
    sideband_addr: str = "tcp://0.0.0.0:51820"
    sideband_peers: tuple[str, ...] = field(default_factory=tuple)
    sideband_heartbeat_ms: int = 50
    max_drift_ms: int = 250

    aggregation_mode: str = "synchronous"  # "synchronous" | "buffer_convergence"
    buffer_capacity: int = 32
    buffer_convergence_eps: float = 1e-3

    diagnostics_dir: Optional[str] = None

    def __post_init__(self) -> None:
        if not (1 <= self.k_active <= self.n_adapters):
            raise ValueError(
                f"k_active must satisfy 1 <= k_active <= n_adapters; "
                f"got k_active={self.k_active}, n_adapters={self.n_adapters}"
            )
        if self.routing_strategy not in {"round_robin", "loss_aware", "random"}:
            raise ValueError(f"unknown routing_strategy: {self.routing_strategy}")
        if self.aggregation_mode not in {"synchronous", "buffer_convergence"}:
            raise ValueError(f"unknown aggregation_mode: {self.aggregation_mode}")
        if self.sideband_enabled and self.aggregation_mode == "synchronous":
            raise ValueError(
                "sideband_enabled=True requires aggregation_mode='buffer_convergence'"
            )
