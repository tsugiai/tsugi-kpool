"""Router tests covering all three routing strategies as implemented at
commit be13bcf: round_robin, random, loss_aware."""
import pytest

from tsugi_kpool.config import KPoolLoraConfig
from tsugi_kpool.router import KPoolRouter, adapter_names


def test_round_robin_cycles() -> None:
    cfg = KPoolLoraConfig(n_adapters=4, k_active=2, routing_strategy="round_robin")
    router = KPoolRouter(cfg)
    seen = [router.select(step=i) for i in range(8)]
    assert seen[0] == (0, 1)
    assert seen[1] == (1, 2)
    assert seen[2] == (2, 3)
    assert seen[3] == (0, 3)
    assert seen[4] == seen[0]


def test_random_returns_valid_k_subset() -> None:
    cfg = KPoolLoraConfig(
        n_adapters=4, k_active=2, routing_strategy="random", routing_seed=42
    )
    router = KPoolRouter(cfg)
    for step in range(20):
        active = router.select(step=step)
        assert len(active) == cfg.k_active
        assert len(set(active)) == cfg.k_active
        assert all(0 <= i < cfg.n_adapters for i in active)


def test_random_is_seed_deterministic() -> None:
    cfg = KPoolLoraConfig(
        n_adapters=8, k_active=3, routing_strategy="random", routing_seed=7
    )
    r1 = KPoolRouter(cfg)
    r2 = KPoolRouter(cfg)
    seq1 = [r1.select(step=i) for i in range(10)]
    seq2 = [r2.select(step=i) for i in range(10)]
    assert seq1 == seq2


def test_loss_aware_falls_back_to_round_robin_during_cold_start() -> None:
    """Per router._select_loss_aware: falls back to round_robin while the
    loss record is sparser than n_adapters."""
    cfg = KPoolLoraConfig(n_adapters=4, k_active=2, routing_strategy="loss_aware")
    router = KPoolRouter(cfg)
    cold = router.select(step=0)
    assert cold == (0, 1)  # matches round_robin step 0


def test_loss_aware_picks_highest_loss_when_record_full() -> None:
    cfg = KPoolLoraConfig(n_adapters=4, k_active=2, routing_strategy="loss_aware")
    router = KPoolRouter(cfg)
    # Populate loss record for all adapters; adapters 2 + 3 have higher loss
    router.record_loss(0, 0.1)
    router.record_loss(1, 0.2)
    router.record_loss(2, 0.9)
    router.record_loss(3, 0.8)
    active = router.select(step=0)
    assert set(active) == {2, 3}


def test_unknown_strategy_raises() -> None:
    with pytest.raises(ValueError):
        KPoolLoraConfig(routing_strategy="not_a_strategy")


def test_adapter_names_maps_indices_to_peft_adapter_names() -> None:
    cfg = KPoolLoraConfig(n_adapters=4, k_active=2)

    assert adapter_names(cfg, (0, 3)) == ["adapter_0", "adapter_3"]
    assert adapter_names(cfg, ()) == []
