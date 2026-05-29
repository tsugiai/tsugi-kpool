"""Deterministic CPU proxies for the patent-credited mechanism behaviors.

These let an outside engineer verify, from the public test suite alone:
  * the eps -> HOLD-rate monotonic step-function shape (the buffer-convergence
    trigger's signature behavior);
  * exactly-K-of-N selection + seed determinism across all routing strategies;
  * that a HOLD defers rather than destroys gradient information (the unit-level
    basis for learning-quality preservation).

They are proxies, not the full cross-rack benchmark (which needs a GPU cluster);
they pin the SHAPE of the behavior on a controlled workload.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from tsugi_kpool.aggregator import BufferConvergenceAggregator  # noqa: E402
from tsugi_kpool.config import KPoolLoraConfig  # noqa: E402
from tsugi_kpool.router import KPoolRouter  # noqa: E402


def test_hold_rate_is_monotone_non_increasing_in_eps() -> None:
    """As buffer_convergence_eps increases, more adapters clear the variance
    threshold and FIRE, so the HOLD rate is monotone non-increasing in eps --
    the testable shadow of the published eps step-function."""
    n = 7
    config = KPoolLoraConfig(
        n_adapters=n, k_active=2, buffer_capacity=4, max_drift_ms=1_000_000
    )
    agg = BufferConvergenceAggregator(config)
    # Give each adapter a buffer with a known, distinct variance spanning
    # several decades. Two scalar snapshots differing by delta give an
    # (unbiased) variance of delta**2 / 2.
    for i in range(n):
        delta = 10.0 ** (i - 3)  # 1e-3 .. 1e3  -> variance 5e-7 .. 5e5
        agg.buffer.push(i, torch.zeros(1))
        agg.buffer.push(i, torch.full((1,), float(delta)))

    eps_grid = [1e-8, 1e-6, 1e-4, 1e-2, 1.0, 1e2, 1e4, 1e8]
    hold_rates = []
    for eps in eps_grid:
        agg.config.buffer_convergence_eps = eps
        # decide() is pure (it does not consume the buffer), so the same
        # fixed buffers can be swept across the eps grid.
        holds = sum(
            1 for i in range(n) if not agg.decide(i, peer_drift_ms=0.0).fired
        )
        hold_rates.append(holds / n)

    for earlier, later in zip(hold_rates, hold_rates[1:]):
        assert later <= earlier, f"HOLD rate increased with eps: {hold_rates}"
    # ... and it is a real transition, not a flat line.
    assert hold_rates[0] == 1.0, "smallest eps should HOLD every adapter"
    assert hold_rates[-1] == 0.0, "largest eps should FIRE every adapter"


@pytest.mark.parametrize("strategy", ["round_robin", "random", "loss_aware"])
def test_router_selects_exactly_k(strategy: str) -> None:
    n, k = 8, 3
    config = KPoolLoraConfig(
        n_adapters=n, k_active=k, routing_strategy=strategy, routing_seed=0
    )
    router = KPoolRouter(config)
    for step in range(20):
        active = router.select(step)
        assert len(active) == k, f"{strategy}: expected {k} active, got {active}"
        assert len(set(active)) == k, f"{strategy}: duplicate adapter in {active}"
        assert all(0 <= idx < n for idx in active), f"{strategy}: out of range {active}"
        assert list(active) == sorted(active), f"{strategy}: not sorted {active}"


def test_random_routing_is_seed_deterministic() -> None:
    def run(seed: int) -> list[tuple[int, ...]]:
        config = KPoolLoraConfig(
            n_adapters=8, k_active=2, routing_strategy="random", routing_seed=seed
        )
        router = KPoolRouter(config)
        return [router.select(s) for s in range(10)]

    seq = run(0)
    assert seq == run(0), "same seed must reproduce the same selection sequence"
    assert len(set(seq)) > 1, "random routing should not be stuck on one subset"


def test_loss_aware_cold_start_then_top_k() -> None:
    n, k = 4, 2
    config = KPoolLoraConfig(
        n_adapters=n, k_active=k, routing_strategy="loss_aware", routing_seed=0
    )
    router = KPoolRouter(config)
    # Cold start (loss record not yet full): still returns exactly K.
    assert len(router.select(0)) == k
    # Record one loss per adapter; adapters 2 and 3 have the highest loss.
    for idx, loss in {0: 0.1, 1: 0.2, 2: 0.9, 3: 0.8}.items():
        router.record_loss(idx, loss)
    assert router.select(1) == (2, 3), "loss_aware should pick the top-K by loss"


def test_hold_zeroes_live_grad_but_preserves_buffer_snapshot() -> None:
    """Learning-quality proxy: a HOLD makes the optimizer step a no-op for the
    adapter (live .grad zeroed) WITHOUT destroying the gradient information --
    the buffered snapshot is an independent clone, so the deferred gradient
    survives for the next aggregation."""
    config = KPoolLoraConfig(
        n_adapters=2,
        k_active=1,
        buffer_convergence_eps=1e-12,
        max_drift_ms=1_000_000,
    )
    agg = BufferConvergenceAggregator(config)
    p = torch.nn.Parameter(torch.zeros(3))
    p.grad = torch.tensor([1.0, 2.0, 3.0])
    agg.buffer.push(0, p.grad)  # as a backward hook would

    # occupancy 1 -> variance inf -> inf < eps is False -> HOLD.
    decisions = agg.process_post_backward(
        {0: [p]}, active_adapters=(0,), peer_drift_ms=0.0
    )
    assert len(decisions) == 1 and not decisions[0].fired, "expected a HOLD"
    assert p.grad is not None and torch.count_nonzero(p.grad).item() == 0, (
        "HOLD should zero the live gradient"
    )
    snap = agg.buffer._per_adapter[0][0]
    assert torch.equal(snap, torch.tensor([1.0, 2.0, 3.0])), (
        "HOLD must not corrupt the buffered snapshot"
    )
