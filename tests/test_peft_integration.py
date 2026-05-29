"""Real-peft integration tests.

The earlier `tests/test_runtime.py` exercises the SDK against a `_ToyLoRA`
fixture that mimics peft adapter naming. These tests exercise the SDK
against an actual peft.LoraConfig + peft.add_adapter multi-adapter
construction on a custom nn.Module, so the parameter names and
requires_grad semantics match what user code sees in the wild.

Requires: torch + peft (the `dev` extras).
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
peft = pytest.importorskip("peft")

import torch.nn as nn
from peft import LoraConfig, get_peft_model

from tsugi_kpool.config import KPoolLoraConfig
from tsugi_kpool.runtime import (
    apply_kpool_step,
    get_runtime,
    plesio_init,
    plesio_shutdown,
    post_backward_step,
)


class _TinyAttn(nn.Module):
    """Minimal nn.Module with named Linear layers peft can target."""

    def __init__(self, dim: int = 16) -> None:
        super().__init__()
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.q_proj(x) + self.v_proj(x)


def _build_peft_multi_adapter(n_adapters: int, dim: int = 16) -> nn.Module:
    """Construct a peft-wrapped model with `n_adapters` LoRA adapters
    named adapter_0 ... adapter_{n-1}. Mirrors `kpool_run.py:268-278`."""
    base = _TinyAttn(dim=dim)
    cfg = LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.0,
    )
    model = get_peft_model(base, cfg, adapter_name="adapter_0")
    for i in range(1, n_adapters):
        model.add_adapter(f"adapter_{i}", cfg)
    return model


def test_peft_param_names_use_adapter_int_segment() -> None:
    """Sanity check: peft >=0.12 names LoRA params with adapter_<i> as
    a literal segment, which is what `_discover_adapter_params` expects."""
    model = _build_peft_multi_adapter(n_adapters=4)
    segments_seen: set[str] = set()
    for name, _ in model.named_parameters():
        for part in name.split("."):
            if part.startswith("adapter_"):
                segments_seen.add(part)
    assert segments_seen >= {"adapter_0", "adapter_1", "adapter_2", "adapter_3"}


def test_discover_finds_all_peft_adapters() -> None:
    """plesio_init's adapter-discovery must find all N adapters (8 params
    each: A and B for q_proj + v_proj), not only the currently-active one.
    """
    model = _build_peft_multi_adapter(n_adapters=4)
    cfg = KPoolLoraConfig(
        n_adapters=4,
        k_active=2,
        routing_strategy="round_robin",
        sideband_enabled=True,
        aggregation_mode="buffer_convergence",
        sideband_addr="tcp://127.0.0.1:0",
        sideband_peers=(),
        buffer_capacity=4,
        buffer_convergence_eps=1e9,
        max_drift_ms=1_000_000,
    )
    plesio_init(model, cfg, sender_id="peft-test-node")
    try:
        rt = get_runtime(model)
        for i in range(cfg.n_adapters):
            # 4 LoRA matrices per adapter (q_proj.lora_A, q_proj.lora_B,
            # v_proj.lora_A, v_proj.lora_B)
            assert len(rt.adapter_params[i]) == 4, (
                f"adapter {i} should have 4 params, "
                f"got {len(rt.adapter_params[i])}"
            )
    finally:
        plesio_shutdown(model)


def test_apply_kpool_step_with_peft_multi_adapter() -> None:
    """apply_kpool_step must activate K adapters simultaneously through
    the peft LoraModel API (not the PeftModel single-adapter-only API)."""
    model = _build_peft_multi_adapter(n_adapters=4)
    cfg = KPoolLoraConfig(
        n_adapters=4,
        k_active=2,
        routing_strategy="round_robin",
        sideband_enabled=True,
        aggregation_mode="buffer_convergence",
        sideband_addr="tcp://127.0.0.1:0",
        sideband_peers=(),
        buffer_capacity=4,
        buffer_convergence_eps=1e9,
        max_drift_ms=1_000_000,
    )
    plesio_init(model, cfg, sender_id="peft-test-node")
    try:
        active = apply_kpool_step(model, step=0)
        assert active == (0, 1)
        assert set(model.active_adapters) == {"adapter_0", "adapter_1"}
    finally:
        plesio_shutdown(model)


def test_real_peft_backward_hooks_populate_buffer() -> None:
    """End-to-end real-peft forward+backward; check the active adapter
    parameters' backward hooks push gradient snapshots into the elastic
    buffer."""
    model = _build_peft_multi_adapter(n_adapters=4, dim=8)
    cfg = KPoolLoraConfig(
        n_adapters=4,
        k_active=2,
        routing_strategy="round_robin",
        sideband_enabled=True,
        aggregation_mode="buffer_convergence",
        sideband_addr="tcp://127.0.0.1:0",
        sideband_peers=(),
        buffer_capacity=8,
        buffer_convergence_eps=1e9,
        max_drift_ms=1_000_000,
    )
    plesio_init(model, cfg, sender_id="peft-test-node")
    try:
        apply_kpool_step(model, step=0)  # selects active adapters -> (0, 1)
        torch.manual_seed(0)
        x = torch.randn(2, 8)
        target = torch.randn(2, 8)
        y = model(x)
        loss = ((y - target) ** 2).mean()
        loss.backward()
        rt = get_runtime(model)
        # Active adapters 0 and 1 should each have at least one snapshot
        # in the buffer (one per LoRA Parameter that fired its hook).
        assert rt.aggregator.buffer.occupancy(0) >= 1, (
            f"adapter 0 buffer is empty after backward; "
            f"occupancy={rt.aggregator.buffer.occupancy(0)}"
        )
        assert rt.aggregator.buffer.occupancy(1) >= 1
        # Inactive adapters 2 and 3 should have empty buffers (their LoRA
        # forward branch was disabled by peft set_adapter, so backward
        # does not produce gradients for them).
        assert rt.aggregator.buffer.occupancy(2) == 0
        assert rt.aggregator.buffer.occupancy(3) == 0
    finally:
        plesio_shutdown(model)


def test_real_peft_full_step_cycle() -> None:
    """plesio_init -> apply_kpool_step -> forward -> backward ->
    post_backward_step on real peft. With a generous buffer_convergence_eps,
    the post_backward_step should FIRE on the active adapters."""
    model = _build_peft_multi_adapter(n_adapters=4, dim=8)
    cfg = KPoolLoraConfig(
        n_adapters=4,
        k_active=2,
        routing_strategy="round_robin",
        sideband_enabled=True,
        aggregation_mode="buffer_convergence",
        sideband_addr="tcp://127.0.0.1:0",
        sideband_peers=(),
        buffer_capacity=8,
        buffer_convergence_eps=1e9,    # very generous; always FIRE-eligible
        max_drift_ms=1_000_000,
    )
    plesio_init(model, cfg, sender_id="peft-test-node")
    try:
        torch.manual_seed(0)
        # Three real training steps so the buffer has multiple snapshots
        # per adapter and FIRE-eligibility is non-trivial.
        for step in range(3):
            active = apply_kpool_step(model, step=step)
            x = torch.randn(2, 8)
            target = torch.randn(2, 8)
            y = model(x)
            loss = ((y - target) ** 2).mean()
            loss.backward()
            decisions = post_backward_step(model, step=step, active=active)
            assert len(decisions) == cfg.k_active
            # Clear .grad manually since we are not running optimizer.step
            for name, p in model.named_parameters():
                if p.grad is not None:
                    p.grad = None
        rt = get_runtime(model)
        # At least one FIRE should have happened across the 6 decisions
        # (2 adapters x 3 steps) once variance-on-stack converges.
        total_fires = sum(rt.aggregator.fire_count.values())
        assert total_fires >= 1, (
            f"expected at least one FIRE decision; "
            f"fire_count={rt.aggregator.fire_count}, "
            f"hold_count={rt.aggregator.hold_count}"
        )
    finally:
        plesio_shutdown(model)
