"""Tests for pre_forward_step (FSDP no_sync()-gated reduce-scatter path).

Exercises the runtime's prior-step HOLD prediction:
- step 0 -> nullcontext (no prior decisions)
- step N where any active adapter's prior decision was FIRE -> nullcontext
- step N where all active adapters' prior decisions were HOLD -> model.no_sync()

Uses a _SyncableToyLoRA subclass that adds a no_sync() context manager
mirroring FSDP's signature, so the gating logic can be verified without
needing torch.distributed / actual FSDP.
"""
from __future__ import annotations

import contextlib

import pytest

torch = pytest.importorskip("torch")

import torch.nn as nn

from tsugi_kpool.aggregator import AggregationDecision
from tsugi_kpool.config import KPoolLoraConfig
from tsugi_kpool.runtime import (
    get_runtime,
    plesio_init,
    plesio_shutdown,
    post_backward_step,
    pre_forward_step,
)


class _SyncableToyLoRA(nn.Module):
    """Toy LoRA that exposes `no_sync()` like FSDP does. The no_sync
    context manager records that it was entered so tests can verify the
    gating chose the no_sync path.
    """

    def __init__(self, dim: int = 16, n_adapters: int = 4) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=False)
        for i in range(n_adapters):
            mod = nn.Module()
            mod.A = nn.Parameter(torch.zeros(dim, dim))
            mod.B = nn.Parameter(torch.zeros(dim, dim))
            setattr(self, f"adapter_{i}", mod)
        self._active_names: list[str] = []
        self.no_sync_entered_count = 0

    def set_adapter(self, names: list[str]) -> None:
        self._active_names = list(names)

    @contextlib.contextmanager
    def no_sync(self):
        self.no_sync_entered_count += 1
        yield

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.linear(x)
        for name in self._active_names:
            idx = int(name.removeprefix("adapter_"))
            mod = getattr(self, f"adapter_{idx}")
            y = y + x @ mod.A @ mod.B.T
        return y


@pytest.fixture
def toy_cfg() -> KPoolLoraConfig:
    return KPoolLoraConfig(
        n_adapters=4,
        k_active=2,
        routing_strategy="round_robin",
        sideband_enabled=False,
        aggregation_mode="buffer_convergence",
        buffer_capacity=4,
        buffer_convergence_eps=1e9,
        max_drift_ms=1_000_000,
    )


def _seed_prior_decisions(
    runtime, adapters: tuple[int, ...], fired: bool
) -> None:
    """Inject AggregationDecisions into runtime.last_decisions to
    simulate the result of a prior post_backward_step."""
    for idx in adapters:
        runtime.last_decisions[idx] = AggregationDecision(
            adapter_idx=idx,
            fired=fired,
            variance=0.0 if fired else 1.0,
            reason="fire" if fired else "hold_variance",
        )


def test_step_zero_returns_nullcontext(toy_cfg: KPoolLoraConfig) -> None:
    model = _SyncableToyLoRA()
    plesio_init(model, toy_cfg)
    try:
        ctx = pre_forward_step(model, step=0, active=(0, 1))
        # nullcontext is a context manager that does nothing on enter/exit
        assert isinstance(ctx, contextlib.nullcontext)
        runtime = get_runtime(model)
        assert runtime.gated_step_active is False
        assert model.no_sync_entered_count == 0
    finally:
        plesio_shutdown(model)


def test_all_active_held_returns_no_sync(toy_cfg: KPoolLoraConfig) -> None:
    model = _SyncableToyLoRA()
    plesio_init(model, toy_cfg)
    try:
        runtime = get_runtime(model)
        _seed_prior_decisions(runtime, adapters=(0, 1), fired=False)
        ctx = pre_forward_step(model, step=1, active=(0, 1))
        assert not isinstance(ctx, contextlib.nullcontext)
        assert runtime.gated_step_active is True
        # Enter the context to confirm it's model.no_sync()
        with ctx:
            pass
        assert model.no_sync_entered_count == 1
    finally:
        plesio_shutdown(model)


def test_any_active_fired_returns_nullcontext(toy_cfg: KPoolLoraConfig) -> None:
    model = _SyncableToyLoRA()
    plesio_init(model, toy_cfg)
    try:
        runtime = get_runtime(model)
        # adapter 0 held, adapter 1 fired -> must NOT gate
        _seed_prior_decisions(runtime, adapters=(0,), fired=False)
        _seed_prior_decisions(runtime, adapters=(1,), fired=True)
        ctx = pre_forward_step(model, step=1, active=(0, 1))
        assert isinstance(ctx, contextlib.nullcontext)
        assert runtime.gated_step_active is False
        assert model.no_sync_entered_count == 0
    finally:
        plesio_shutdown(model)


def test_missing_prior_decision_returns_nullcontext(toy_cfg: KPoolLoraConfig) -> None:
    model = _SyncableToyLoRA()
    plesio_init(model, toy_cfg)
    try:
        runtime = get_runtime(model)
        # Only adapter 0 has a prior decision; adapter 1 doesn't.
        _seed_prior_decisions(runtime, adapters=(0,), fired=False)
        ctx = pre_forward_step(model, step=1, active=(0, 1))
        assert isinstance(ctx, contextlib.nullcontext)
        assert runtime.gated_step_active is False
    finally:
        plesio_shutdown(model)


def test_non_fsdp_model_returns_nullcontext(toy_cfg: KPoolLoraConfig) -> None:
    """A model without no_sync (e.g., plain nn.Module, non-FSDP wrap) must
    fall back to nullcontext even when prior decisions all HOLD'd."""

    class _NoSyncLessToy(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(16, 16, bias=False)
            for i in range(4):
                mod = nn.Module()
                mod.A = nn.Parameter(torch.zeros(16, 16))
                mod.B = nn.Parameter(torch.zeros(16, 16))
                setattr(self, f"adapter_{i}", mod)
            self._active_names: list[str] = []

        def set_adapter(self, names: list[str]) -> None:
            self._active_names = list(names)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.linear(x)

    model = _NoSyncLessToy()
    plesio_init(model, toy_cfg)
    try:
        runtime = get_runtime(model)
        _seed_prior_decisions(runtime, adapters=(0, 1), fired=False)
        ctx = pre_forward_step(model, step=1, active=(0, 1))
        assert isinstance(ctx, contextlib.nullcontext)
        assert runtime.gated_step_active is False
    finally:
        plesio_shutdown(model)


def test_gated_post_backward_forces_hold(toy_cfg: KPoolLoraConfig) -> None:
    """When pre_forward_step gated the step, post_backward_step must
    take the forced-HOLD path: zero local grads + emit gated_predict_hold
    decisions; the variance trigger is bypassed."""
    model = _SyncableToyLoRA()
    plesio_init(model, toy_cfg)
    try:
        runtime = get_runtime(model)
        # Seed prior HOLD decisions so pre_forward_step gates.
        _seed_prior_decisions(runtime, adapters=(0, 1), fired=False)
        ctx = pre_forward_step(model, step=1, active=(0, 1))
        assert runtime.gated_step_active is True
        # Simulate a forward+backward by directly populating .grad on
        # active-adapter params (autograd would have done this for real).
        with ctx:
            for idx in (0, 1):
                mod = getattr(model, f"adapter_{idx}")
                mod.A.grad = torch.ones_like(mod.A)
                mod.B.grad = torch.ones_like(mod.B)
        # Note: gated_step_active is set on the runtime so post_backward
        # takes the forced-HOLD path even though the buffer might or
        # might not be populated.
        decisions = post_backward_step(model, step=1, active=(0, 1))
        assert len(decisions) == 2
        for d in decisions:
            assert d.fired is False
            assert d.reason == "gated_predict_hold"
        # Local grads must be zeroed
        for idx in (0, 1):
            mod = getattr(model, f"adapter_{idx}")
            assert torch.all(mod.A.grad == 0)
            assert torch.all(mod.B.grad == 0)
        # gated_step_active must be cleared after the forced-HOLD path
        assert runtime.gated_step_active is False
    finally:
        plesio_shutdown(model)


def test_non_gated_post_backward_uses_variance_trigger(toy_cfg: KPoolLoraConfig) -> None:
    """When pre_forward_step did NOT gate, post_backward_step uses the
    standard variance-trigger path (regression-check that the new gating
    code does not break the existing path)."""
    model = _SyncableToyLoRA()
    plesio_init(model, toy_cfg)
    try:
        runtime = get_runtime(model)
        # No prior decisions, step 0 -> not gated
        ctx = pre_forward_step(model, step=0, active=(0, 1))
        assert runtime.gated_step_active is False
        with ctx:
            for idx in (0, 1):
                mod = getattr(model, f"adapter_{idx}")
                mod.A.grad = torch.ones_like(mod.A)
                mod.B.grad = torch.ones_like(mod.B)
        decisions = post_backward_step(model, step=0, active=(0, 1))
        # With eps=1e9 (very loose), the variance trigger fires by default
        # on the empty buffer path -> "empty_buffer" reason. Either way,
        # the reason must NOT be "gated_predict_hold".
        for d in decisions:
            assert d.reason != "gated_predict_hold"
    finally:
        plesio_shutdown(model)
