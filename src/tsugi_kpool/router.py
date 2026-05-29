"""K-out-of-N adapter routing.

Implements K-Pool LoRA's adapter selection step. The router decides which
K of N adapters contribute to the current step; peft's native
multi-adapter `set_adapter(list)` API performs the actual masking. Loss-
aware routing requires a recent-loss callback supplied by the training
loop.

Implements the K-out-of-N routing described in the K-Pool LoRA provisional
(US App. 64/060,315).
"""
from __future__ import annotations

import itertools
import random
from typing import TYPE_CHECKING, Iterator

from tsugi_kpool.config import KPoolLoraConfig

if TYPE_CHECKING:
    import torch.nn as nn


# K-of-N adapter routing (relates to K-Pool LoRA, US App. 64/060,315).
#
# The SDK maintains a fixed-size pool of N low-rank adapter parameter
# sets and selects K_active of them to participate in any given step
# (the K-of-N generalization, not a single-active-slot variant). The
# design rationale -- "which K of N adapters participate, and at which
# granularity" -- is documented in `docs/architecture.md`.
class KPoolRouter:
    """Selects K active adapter indices per step from an N-adapter pool.

    Routing strategies:
        round_robin: cycles deterministically through the N-adapter pool.
                     The cursor advances by one adapter per step so the
                     coverage over N steps is uniform.
        random:      uniform random K-subset selection per step seeded by
                     `config.routing_seed`.
        loss_aware:  picks K adapters with highest recent loss
                     contribution; falls back to round_robin until
                     loss_record_step has been called for every adapter
                     at least once.
    """

    def __init__(self, config: KPoolLoraConfig) -> None:
        self.config = config
        self._round_robin_cursor: Iterator[tuple[int, ...]] = self._make_rr_cursor()
        self._rng = random.Random(config.routing_seed)
        self._loss_record: dict[int, float] = {}
        self._last_active: tuple[int, ...] = tuple(range(config.k_active))

    def _make_rr_cursor(self) -> Iterator[tuple[int, ...]]:
        n, k = self.config.n_adapters, self.config.k_active
        return itertools.cycle(
            tuple(sorted((i + j) % n for j in range(k))) for i in range(n)
        )

    # Round-robin, random, and loss-aware are three alternative
    # selection rules for the K-of-N adapter-selection function. They
    # are deliberately simple (no gradient-detached Gaussian-mixture
    # fitting) so the routing behavior is easy to reason about and test.
    def select(self, step: int) -> tuple[int, ...]:
        """Return K adapter indices active for the given training step."""
        strategy = self.config.routing_strategy
        if strategy == "round_robin":
            chosen = next(self._round_robin_cursor)
        elif strategy == "random":
            chosen = tuple(
                sorted(self._rng.sample(range(self.config.n_adapters), self.config.k_active))
            )
        elif strategy == "loss_aware":
            chosen = self._select_loss_aware()
        else:
            raise AssertionError(f"unreachable: {strategy}")
        self._last_active = chosen
        return chosen

    def _select_loss_aware(self) -> tuple[int, ...]:
        """Pick K adapters with the highest recent loss contribution.

        Falls back to round_robin while the loss record is sparse so the
        cold-start period does not over-commit to whichever adapter
        happened to be sampled first.
        """
        if len(self._loss_record) < self.config.n_adapters:
            return next(self._round_robin_cursor)
        sorted_indices = sorted(
            self._loss_record.items(), key=lambda kv: kv[1], reverse=True
        )
        return tuple(sorted(idx for idx, _ in sorted_indices[: self.config.k_active]))

    def record_loss(self, adapter_idx: int, loss_value: float, momentum: float = 0.9) -> None:
        """Exponential-moving-average update of per-adapter loss
        contribution. Called by the training loop's diagnostics path.
        """
        prev = self._loss_record.get(adapter_idx)
        self._loss_record[adapter_idx] = (
            loss_value if prev is None else momentum * prev + (1.0 - momentum) * loss_value
        )

    @property
    def last_active(self) -> tuple[int, ...]:
        return self._last_active


def attach_router(model: "nn.Module", config: KPoolLoraConfig) -> KPoolRouter:
    """Attach a KPoolRouter to an SDK-managed model.

    The router is stored as `model._kpool_router` so the training loop
    (or runtime helpers) can read it without re-importing. The router
    does not install forward hooks itself; the actual masking happens via
    peft's `set_adapter` called from `apply_kpool_step` in runtime.py.
    """
    router = KPoolRouter(config)
    setattr(model, "_kpool_router", router)
    return router


def adapter_names(config: KPoolLoraConfig, active: tuple[int, ...]) -> list[str]:
    """Map K-Pool integer indices to the peft adapter-name convention
    used by both kpool_run.py and tsugi_kpool: 'adapter_{i}'."""
    return [f"adapter_{i}" for i in active]
