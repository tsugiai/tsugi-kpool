"""Config validation tests. Working as of week-0 commit."""
import pytest

from tsugi_kpool.config import KPoolLoraConfig


def test_default_config_validates() -> None:
    cfg = KPoolLoraConfig()
    assert cfg.n_adapters == 8
    assert cfg.k_active == 2
    assert cfg.aggregation_mode == "synchronous"
    assert cfg.sideband_enabled is False


def test_k_active_must_be_in_range() -> None:
    with pytest.raises(ValueError, match="k_active"):
        KPoolLoraConfig(n_adapters=4, k_active=0)
    with pytest.raises(ValueError, match="k_active"):
        KPoolLoraConfig(n_adapters=4, k_active=5)


def test_unknown_routing_strategy_rejected() -> None:
    with pytest.raises(ValueError, match="routing_strategy"):
        KPoolLoraConfig(routing_strategy="nope")


def test_sideband_requires_buffer_convergence() -> None:
    with pytest.raises(ValueError, match="aggregation_mode"):
        KPoolLoraConfig(sideband_enabled=True, aggregation_mode="synchronous")


def test_buffer_convergence_with_sideband_ok() -> None:
    cfg = KPoolLoraConfig(
        sideband_enabled=True,
        aggregation_mode="buffer_convergence",
    )
    assert cfg.sideband_enabled
