"""Aggregator + elastic buffer tests at the BufferConvergenceAggregator
API surface as implemented at commit be13bcf:
    - ElasticAdapterBuffer.push / .variance / .occupancy / .clear
    - BufferConvergenceAggregator.decide (single-adapter)
    - BufferConvergenceAggregator.process_post_backward (FIRE/HOLD action)
"""
import torch

from tsugi_kpool.aggregator import (
    AggregationDecision,
    BufferConvergenceAggregator,
    ElasticAdapterBuffer,
)
from tsugi_kpool.config import KPoolLoraConfig


def test_buffer_variance_zero_when_identical() -> None:
    cfg = KPoolLoraConfig(buffer_capacity=4)
    buf = ElasticAdapterBuffer(cfg)
    g = torch.zeros(8)
    for _ in range(3):
        buf.push(0, g)
    assert buf.variance(0) == 0.0


def test_buffer_variance_inf_when_underfilled() -> None:
    """The variance trigger fires only when the elastic buffer has at
    least the minimum number of snapshots; an underfilled buffer returns
    inf to mean 'do not fire yet'."""
    cfg = KPoolLoraConfig(buffer_capacity=4)
    buf = ElasticAdapterBuffer(cfg)
    buf.push(0, torch.zeros(8))
    assert buf.variance(0) == float("inf")


def test_buffer_occupancy_tracks_pushes() -> None:
    cfg = KPoolLoraConfig(buffer_capacity=4)
    buf = ElasticAdapterBuffer(cfg)
    assert buf.occupancy(0) == 0
    buf.push(0, torch.zeros(4))
    buf.push(0, torch.zeros(4))
    assert buf.occupancy(0) == 2


def test_buffer_clear_resets_one_adapter() -> None:
    cfg = KPoolLoraConfig(buffer_capacity=4)
    buf = ElasticAdapterBuffer(cfg)
    buf.push(0, torch.zeros(4))
    buf.push(1, torch.zeros(4))
    buf.clear(0)
    assert buf.occupancy(0) == 0
    assert buf.occupancy(1) == 1


def test_decide_hold_when_drift_excessive() -> None:
    cfg = KPoolLoraConfig(buffer_convergence_eps=1.0, max_drift_ms=100)
    agg = BufferConvergenceAggregator(cfg)
    # Pre-populate buffer so occupancy > 0
    agg.buffer.push(0, torch.zeros(4))
    agg.buffer.push(0, torch.zeros(4))
    decision = agg.decide(adapter_idx=0, peer_drift_ms=500.0)
    assert decision.fired is False
    assert decision.reason == "hold_drift"


def test_decide_fires_when_variance_converged() -> None:
    cfg = KPoolLoraConfig(
        buffer_convergence_eps=10.0, buffer_capacity=8, max_drift_ms=1000
    )
    agg = BufferConvergenceAggregator(cfg)
    for _ in range(3):
        agg.buffer.push(0, torch.zeros(4))
    decision = agg.decide(adapter_idx=0, peer_drift_ms=1.0)
    assert decision.fired is True
    assert decision.reason == "fire"
    assert decision.variance == 0.0


def test_decide_holds_when_variance_above_eps() -> None:
    cfg = KPoolLoraConfig(
        buffer_convergence_eps=1e-6, buffer_capacity=8, max_drift_ms=1000
    )
    agg = BufferConvergenceAggregator(cfg)
    torch.manual_seed(0)
    for _ in range(3):
        agg.buffer.push(0, torch.randn(4) * 10.0)
    decision = agg.decide(adapter_idx=0, peer_drift_ms=1.0)
    assert decision.fired is False
    assert decision.reason == "hold_variance"


def test_decide_empty_buffer_returns_empty_buffer_reason() -> None:
    cfg = KPoolLoraConfig()
    agg = BufferConvergenceAggregator(cfg)
    decision = agg.decide(adapter_idx=0, peer_drift_ms=0.0)
    assert decision.fired is False
    assert decision.reason == "empty_buffer"


def test_process_post_backward_zeros_grad_on_hold() -> None:
    cfg = KPoolLoraConfig(
        n_adapters=2,
        k_active=2,
        buffer_convergence_eps=1e-6,
        buffer_capacity=8,
        max_drift_ms=1000,
    )
    agg = BufferConvergenceAggregator(cfg)
    # Two adapter parameters per adapter, each with a non-zero .grad
    params: dict[int, list[torch.nn.Parameter]] = {0: [], 1: []}
    for i in (0, 1):
        for _ in range(2):
            p = torch.nn.Parameter(torch.zeros(4))
            p.grad = torch.ones(4)
            params[i].append(p)
    # Noisy buffer so HOLD fires
    torch.manual_seed(1)
    for _ in range(3):
        agg.buffer.push(0, torch.randn(4) * 5.0)
        agg.buffer.push(1, torch.randn(4) * 5.0)
    decisions = agg.process_post_backward(params, active_adapters=(0, 1), peer_drift_ms=1.0)
    assert all(not d.fired for d in decisions)
    # Gradients should now be zeroed by the HOLD path
    for i in (0, 1):
        for p in params[i]:
            assert torch.allclose(p.grad, torch.zeros_like(p))


def test_aggregation_decision_dataclass_is_frozen() -> None:
    d = AggregationDecision(adapter_idx=0, fired=True, variance=0.0, reason="fire")
    try:
        d.fired = False  # type: ignore[misc]
    except Exception:
        return
    raise AssertionError("AggregationDecision should be frozen")
