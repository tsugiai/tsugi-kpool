"""Config validation tests. Working as of week-0 commit."""
import math

import pytest

from tsugi_kpool import (
    recommend_buffer_convergence_eps as exported_recommend_buffer_convergence_eps,
)
from tsugi_kpool.config import KPoolLoraConfig, recommend_buffer_convergence_eps


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


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"buffer_convergence_eps": 0.0}, "buffer_convergence_eps"),
        ({"buffer_convergence_eps": -1e-3}, "buffer_convergence_eps"),
        ({"buffer_convergence_eps": math.inf}, "buffer_convergence_eps"),
        ({"buffer_convergence_eps": math.nan}, "buffer_convergence_eps"),
        ({"buffer_capacity": 1}, "buffer_capacity"),
        ({"max_drift_ms": -1}, "max_drift_ms"),
        ({"sideband_heartbeat_ms": 0}, "sideband_heartbeat_ms"),
        ({"r": 0}, "r"),
        ({"lora_dropout": -0.01}, "lora_dropout"),
        ({"lora_dropout": 1.01}, "lora_dropout"),
    ],
)
def test_invalid_numeric_fields_are_rejected(
    kwargs: dict[str, float | int], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        KPoolLoraConfig(**kwargs)


def test_lora_dropout_accepts_inclusive_endpoints() -> None:
    assert KPoolLoraConfig(lora_dropout=0.0).lora_dropout == 0.0
    assert KPoolLoraConfig(lora_dropout=1.0).lora_dropout == 1.0


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


def test_recommend_buffer_convergence_eps_uses_finite_quantile() -> None:
    samples = [math.nan, math.inf, -math.inf, 1e-4, 1e-3, 1e-2]
    eps = recommend_buffer_convergence_eps(samples)
    assert eps == pytest.approx(1e-3)


def test_recommend_buffer_convergence_eps_interpolates_requested_quantile() -> None:
    eps = recommend_buffer_convergence_eps([1.0, 3.0, 5.0], quantile=0.75)
    assert eps == pytest.approx(4.0)


def test_recommend_buffer_convergence_eps_uses_positive_floor() -> None:
    assert recommend_buffer_convergence_eps([math.nan, math.inf], floor=1e-9) == 1e-9
    assert recommend_buffer_convergence_eps([0.0], floor=1e-9) == 1e-9


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"quantile": -0.1}, "quantile"),
        ({"quantile": 1.1}, "quantile"),
        ({"quantile": math.nan}, "quantile"),
        ({"floor": 0.0}, "floor"),
        ({"floor": math.inf}, "floor"),
    ],
)
def test_recommend_buffer_convergence_eps_rejects_invalid_options(
    kwargs: dict[str, float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        recommend_buffer_convergence_eps([1.0], **kwargs)


def test_recommend_buffer_convergence_eps_is_exported_from_package_root() -> None:
    assert exported_recommend_buffer_convergence_eps([1.0, 2.0, 3.0]) == pytest.approx(
        2.0
    )
