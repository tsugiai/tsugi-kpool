"""Integration test for plesio_init / apply_kpool_step / post_backward_step.

Uses a tiny CPU model with peft-style adapter naming so we exercise the
end-to-end SDK glue without needing torch.distributed or a GPU.

Skipped automatically if torch or peft are not installed.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import torch.nn as nn

from tsugi_kpool.config import KPoolLoraConfig
from tsugi_kpool.runtime import (
    apply_kpool_step,
    get_runtime,
    plesio_init,
    plesio_shutdown,
    post_backward_step,
)


class _ToyLoRA(nn.Module):
    """A toy model that mimics the peft adapter-naming convention so the
    runtime's `_discover_adapter_params` finds the right parameters.

    Adapter params are exposed under submodules named `adapter_{i}` so
    that `named_parameters()` yields `adapter_0.A`, `adapter_0.B`, etc.
    The runtime's discovery walks name segments and matches the
    `adapter_<int>` segment exactly. Hanging an `adapter_0_A` flat
    attribute (single segment with appended suffix) would not match;
    the submodule wrap is what makes discovery succeed.
    """

    def __init__(self, dim: int = 16, n_adapters: int = 4) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=False)
        # Submodule layout: model.adapter_i.A + model.adapter_i.B
        for i in range(n_adapters):
            mod = nn.Module()
            mod.A = nn.Parameter(torch.zeros(dim, dim))
            mod.B = nn.Parameter(torch.zeros(dim, dim))
            setattr(self, f"adapter_{i}", mod)
        self._active_names: list[str] = []

    def set_adapter(self, names: list[str]) -> None:
        self._active_names = list(names)

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
        aggregation_mode="synchronous",
        buffer_capacity=4,
        buffer_convergence_eps=1e9,  # very loose, so HOLD never triggers
        max_drift_ms=1_000_000,
    )


def test_plesio_init_raises_without_adapter_pool() -> None:
    # A model with no `adapter_<int>` submodules yields an empty discovery;
    # plesio_init must fail loudly rather than wire up an inert runtime.
    model = nn.Linear(4, 4)
    cfg = KPoolLoraConfig(n_adapters=4, k_active=2)
    with pytest.raises(RuntimeError, match="no adapters named"):
        plesio_init(model, cfg, sender_id="test-node")


def test_plesio_init_finds_adapter_params(toy_cfg: KPoolLoraConfig) -> None:
    # The default cfg has sideband disabled, but plesio_init does not
    # allow sideband_enabled=False with aggregation_mode=buffer_convergence
    # via the constructor; for this test we want the path that does the
    # backward-hook installation, so flip to buffer_convergence + skip
    # sideband by overriding via dataclasses.replace-style construction.
    cfg = KPoolLoraConfig(
        n_adapters=4,
        k_active=2,
        routing_strategy="round_robin",
        sideband_enabled=True,
        aggregation_mode="buffer_convergence",
        sideband_addr="tcp://127.0.0.1:0",  # let OS pick port
        sideband_peers=(),
        buffer_capacity=4,
        buffer_convergence_eps=1e9,
        max_drift_ms=1_000_000,
    )
    model = _ToyLoRA(dim=8, n_adapters=cfg.n_adapters)
    plesio_init(model, cfg, sender_id="test-node")
    try:
        rt = get_runtime(model)
        # Each adapter should have exactly 2 params (A and B).
        for i in range(cfg.n_adapters):
            assert len(rt.adapter_params[i]) == 2, (
                f"adapter {i} should have 2 params, got {len(rt.adapter_params[i])}"
            )
    finally:
        plesio_shutdown(model)


def test_apply_kpool_step_round_robin(toy_cfg: KPoolLoraConfig) -> None:
    cfg = KPoolLoraConfig(
        n_adapters=4,
        k_active=2,
        routing_strategy="round_robin",
        sideband_enabled=True,
        aggregation_mode="buffer_convergence",
        sideband_addr="tcp://127.0.0.1:0",
        sideband_peers=(),
    )
    model = _ToyLoRA(dim=8, n_adapters=cfg.n_adapters)
    plesio_init(model, cfg, sender_id="test-node")
    try:
        active_0 = apply_kpool_step(model, step=0)
        active_1 = apply_kpool_step(model, step=1)
        assert active_0 == (0, 1)
        assert active_1 == (1, 2)
        assert model._active_names == ["adapter_1", "adapter_2"]
    finally:
        plesio_shutdown(model)


def test_backward_hook_populates_buffer() -> None:
    cfg = KPoolLoraConfig(
        n_adapters=4,
        k_active=2,
        routing_strategy="round_robin",
        sideband_enabled=True,
        aggregation_mode="buffer_convergence",
        sideband_addr="tcp://127.0.0.1:0",
        sideband_peers=(),
        buffer_capacity=8,
        buffer_convergence_eps=1e9,    # very loose, never HOLD
        max_drift_ms=1_000_000,
    )
    model = _ToyLoRA(dim=8, n_adapters=cfg.n_adapters)
    plesio_init(model, cfg, sender_id="test-node")
    try:
        # Activate adapters 0,1 and do a synthetic forward+backward
        apply_kpool_step(model, step=0)
        x = torch.randn(2, 8, requires_grad=False)
        target = torch.randn(2, 8)
        # Manually require grad on adapter_0 + adapter_1 params. Match on
        # the submodule prefix `adapter_0.` so we do not accidentally
        # match `adapter_0` as a substring of `adapter_01` etc.
        for name, p in model.named_parameters():
            p.requires_grad_("adapter_0." in name or "adapter_1." in name)
        y = model(x)
        loss = ((y - target) ** 2).mean()
        loss.backward()
        rt = get_runtime(model)
        # Adapter 0 and 1 should now have gradient snapshots in their buffer
        assert rt.aggregator.buffer.occupancy(0) >= 1
        assert rt.aggregator.buffer.occupancy(1) >= 1
        # Adapters 2 and 3 were inactive (params didn't require grad), so
        # backward hooks did not fire for them
        assert rt.aggregator.buffer.occupancy(2) == 0
        assert rt.aggregator.buffer.occupancy(3) == 0
    finally:
        plesio_shutdown(model)


def test_post_backward_step_fires_when_variance_low() -> None:
    cfg = KPoolLoraConfig(
        n_adapters=2,
        k_active=2,
        routing_strategy="round_robin",
        sideband_enabled=True,
        aggregation_mode="buffer_convergence",
        sideband_addr="tcp://127.0.0.1:0",
        sideband_peers=(),
        buffer_capacity=8,
        buffer_convergence_eps=10.0,    # generous: low variance -> FIRE
        max_drift_ms=1_000_000,
    )
    model = _ToyLoRA(dim=4, n_adapters=cfg.n_adapters)
    plesio_init(model, cfg, sender_id="test-node")
    try:
        # Pre-load identical (zero-variance) gradient snapshots into the
        # buffer for adapters 0 and 1 via direct push (bypassing the hook
        # path so the test does not depend on autograd ordering).
        rt = get_runtime(model)
        for _ in range(3):
            rt.aggregator.buffer.push(0, torch.zeros(8))
            rt.aggregator.buffer.push(1, torch.zeros(8))
        apply_kpool_step(model, step=0)
        decisions = post_backward_step(model, step=0, active=(0, 1))
        assert all(d.fired for d in decisions), [
            (d.adapter_idx, d.reason) for d in decisions
        ]
    finally:
        plesio_shutdown(model)


def test_post_backward_step_holds_when_variance_high() -> None:
    cfg = KPoolLoraConfig(
        n_adapters=2,
        k_active=2,
        routing_strategy="round_robin",
        sideband_enabled=True,
        aggregation_mode="buffer_convergence",
        sideband_addr="tcp://127.0.0.1:0",
        sideband_peers=(),
        buffer_capacity=8,
        buffer_convergence_eps=1e-6,   # very strict: any variance -> HOLD
        max_drift_ms=1_000_000,
    )
    model = _ToyLoRA(dim=4, n_adapters=cfg.n_adapters)
    plesio_init(model, cfg, sender_id="test-node")
    try:
        rt = get_runtime(model)
        # Push intentionally noisy snapshots; variance >> eps
        torch.manual_seed(0)
        for _ in range(3):
            rt.aggregator.buffer.push(0, torch.randn(8) * 10.0)
            rt.aggregator.buffer.push(1, torch.randn(8) * 10.0)
        # Also give each adapter a .grad so the HOLD path has something to zero
        for name, p in model.named_parameters():
            if "adapter_0." in name or "adapter_1." in name:
                p.grad = torch.ones_like(p)
        apply_kpool_step(model, step=0)
        decisions = post_backward_step(model, step=0, active=(0, 1))
        assert all(not d.fired for d in decisions)
        # HOLD path should have zeroed the gradients
        for name, p in model.named_parameters():
            if "adapter_0." in name or "adapter_1." in name:
                assert p.grad is not None
                assert torch.allclose(p.grad, torch.zeros_like(p))
    finally:
        plesio_shutdown(model)
