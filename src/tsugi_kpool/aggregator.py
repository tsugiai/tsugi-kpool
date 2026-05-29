"""Adapter-gradient elastic buffer + buffer-convergence aggregation rule.

Implements the buffer-convergence aggregation described in the Infinity
provisional (US App. 64/055,093) at LoRA adapter granularity.

Integration model:
    1. Backward hooks installed by plesio_init capture local pre-reduce
       LoRA gradients and push them into the ElasticAdapterBuffer (one
       FIFO per adapter index).
    2. After the backward pass and before optimizer.step, the training
       loop calls `process_post_backward(model, active_adapters,
       peer_drift_ms)`. The aggregator decides per-adapter whether to
       FIRE (let FSDP reduce + optimizer.step apply this step's gradient
       normally) or HOLD (zero the local gradient so optimizer.step is a
       no-op; the buffer retains snapshots for the next decision).
    3. Diagnostics (variance, fire/hold counts) are emitted alongside
       the per-step training log via the diagnostics writer.

Decision rule per adapter:
    HOLD if peer_drift_ms > config.max_drift_ms  (sideband drift exceeded)
    FIRE if variance < config.buffer_convergence_eps
    HOLD otherwise (variance still settling)

The "HOLD" action implements bounded gradient accumulation in the buffer
without committing the optimizer; the "FIRE" action passes the local
gradient through to the standard FSDP reduce-scatter and optimizer step.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque

import torch
from torch import Tensor

from tsugi_kpool.config import KPoolLoraConfig


@dataclass(frozen=True)
class AggregationDecision:
    """Per-adapter post-backward decision."""
    adapter_idx: int
    fired: bool
    variance: float
    reason: str           # "fire", "hold_drift", "hold_variance", "empty_buffer"


# Per-adapter FIFO of recent gradient snapshots. Stores flattened
# gradient tensors and produces a fill-level signal (`occupancy`) for the
# variance-trigger aggregation below. K active adapters accumulate
# independently.
class ElasticAdapterBuffer:
    """FIFO of recent adapter-gradient snapshots used for the variance-
    trigger aggregation. Buffer is keyed by adapter index so K active
    adapters accumulate independently.

    Snapshots are stored as flattened tensors on the SOURCE DEVICE of
    the incoming gradient (typically the same accelerator the model is
    on). Profiling showed that synchronous GPU-to-
    CPU copies inside `push()` dominated the per-step wall-clock under
    FSDP (24 sync stalls per backward × ~30 ms each ≈ 700+ ms overhead
    per step). Keeping snapshots on the source device eliminates those
    sync stalls; the only remaining sync is the `.item()` call in
    `variance()`, which fires once per active adapter per training
    step (typically K=2 syncs per step rather than 24).

    GPU-memory cost: each snapshot is `numel × bytes_per_element` (for
    a typical LoRA rank-4 matrix on a 768-dim transformer block,
    ~3 KB). With buffer_capacity=8 snapshots per adapter and N
    adapters, the per-adapter buffer is on the order of 24-96 KB; the
    full N-adapter buffer ~ 100-400 KB total — negligible vs the 40-80
    GB memory footprint of a typical large-model training run.

    Buffer residency: both CPU-resident and accelerator-resident
    snapshots are valid; source-device retention (this implementation)
    is the high-performance choice and avoids the GPU-to-CPU sync stalls
    described above.
    """

    def __init__(self, config: KPoolLoraConfig) -> None:
        self.config = config
        self._per_adapter: dict[int, Deque[Tensor]] = {
            i: deque(maxlen=config.buffer_capacity)
            for i in range(config.n_adapters)
        }

    def push(self, adapter_idx: int, grad: Tensor) -> None:
        """Append a flattened snapshot of the gradient on the gradient's
        source device. Called from backward hooks; must be cheap.

        Uses `.detach().flatten().clone()` instead of `.to("cpu", copy=
        True)`. Detach + flatten are O(0) view operations; clone is an
        async same-device memcpy on the gradient's stream and does not
        force a host sync. The resulting tensor is independent of the
        live `.grad` buffer (so subsequent in-place .zero_() in the
        HOLD path does not corrupt past snapshots).
        """
        snap = grad.detach().flatten().clone()
        self._per_adapter[adapter_idx].append(snap)

    def variance(self, adapter_idx: int) -> float:
        """Per-element-mean variance across stored snapshots for one
        adapter. Returns inf when fewer than 2 snapshots are stored.

        Variance is computed in fp32 regardless of snapshot dtype.
        Under FSDP MixedPrecision(param_dtype=bf16) the gradients
        captured by push() are bf16, which has only ~3 decimal digits
        of mantissa precision. Computing variance directly in bf16
        rounds small fluctuations to zero, causing the trigger to see
        var=0 when the actual gradient variance is small-but-nonzero;
        the trigger then FIREs unconditionally and the SDK never
        delivers communication savings. Casting to fp32 before the
        var() computation preserves precision; only the final
        `.item()` call forces a host sync to extract the scalar
        variance for the FIRE/HOLD decision (~1 sync per active
        adapter per training step).
        """
        snaps = self._per_adapter[adapter_idx]
        if len(snaps) < 2:
            return float("inf")
        sizes = {s.numel() for s in snaps}
        if len(sizes) > 1:
            # Snapshots from different LoRA params under the same adapter
            # have different sizes. Compute variance on each subset and
            # take the max as the conservative "is this adapter settled".
            grouped: dict[int, list[Tensor]] = {}
            for s in snaps:
                grouped.setdefault(s.numel(), []).append(s)
            return max(
                float(torch.stack(g).to(torch.float32).var(dim=0).mean().item())
                for g in grouped.values()
                if len(g) >= 2
            )
        stacked = torch.stack(list(snaps)).to(torch.float32)
        return float(stacked.var(dim=0).mean().item())

    def occupancy(self, adapter_idx: int) -> int:
        return len(self._per_adapter[adapter_idx])

    def converged(self, adapter_idx: int) -> bool:
        return self.variance(adapter_idx) < self.config.buffer_convergence_eps

    def clear(self, adapter_idx: int) -> None:
        self._per_adapter[adapter_idx].clear()

    def clear_all(self) -> None:
        for q in self._per_adapter.values():
            q.clear()


class BufferConvergenceAggregator:
    """Couples the elastic buffer with the sideband. `process_post_backward`
    runs every optimizer step after backward and before optimizer.step,
    and returns a list of decisions plus a side effect of zeroing the
    gradient of HOLD-decided adapters.

    The aggregator owns the buffer; the runtime owns the registry of
    LoRA params per adapter (so the aggregator can find the right .grad
    tensors when it needs to zero them)."""

    def __init__(self, config: KPoolLoraConfig) -> None:
        self.config = config
        self.buffer = ElasticAdapterBuffer(config)
        self._fire_count: dict[int, int] = {i: 0 for i in range(config.n_adapters)}
        self._hold_count: dict[int, int] = {i: 0 for i in range(config.n_adapters)}

    # Buffer-convergence aggregation trigger (relates to Infinity,
    # US App. 64/055,093). The variance-threshold trigger encoded below
    # ("FIRE when variance < buffer_convergence_eps") is a data-driven
    # aggregation policy, in contrast to time/quorum-based triggers such
    # as Decoupled DiLoCo (arXiv:2604.21428):
    #   * A time/quorum trigger fires when a time or learner-count
    #     condition is satisfied (e.g. receive from at least K learners
    #     plus an adaptive grace window); the policy is independent of
    #     the variance of the received fragments.
    #   * This SDK fires when a convergence condition is satisfied -- the
    #     variance across stored snapshots in the elastic buffer falls
    #     below an epsilon threshold and (optionally) the sideband-
    #     reported peer drift indicates the peers are within phase
    #     tolerance.
    # The variance test is computed in-place on the elastic buffer's
    # tensor stack; no grace timer, no learner count, no token weighting
    # is involved.
    def decide(self, adapter_idx: int, peer_drift_ms: float) -> AggregationDecision:
        if self.buffer.occupancy(adapter_idx) == 0:
            return AggregationDecision(adapter_idx, False, float("inf"), "empty_buffer")
        var = self.buffer.variance(adapter_idx)
        if peer_drift_ms > self.config.max_drift_ms:
            return AggregationDecision(adapter_idx, False, var, "hold_drift")
        if var < self.config.buffer_convergence_eps:
            return AggregationDecision(adapter_idx, True, var, "fire")
        return AggregationDecision(adapter_idx, False, var, "hold_variance")

    # The HOLD path below is the action-side embodiment of the buffer-
    # convergence aggregation rule: when the variance trigger does NOT
    # fire, the per-step gradient is explicitly zeroed in place so that
    # the downstream optimizer.step becomes a no-op for the affected
    # adapter, while the elastic buffer retains the snapshot for the
    # next decision. This zero-on-HOLD action is what distinguishes the
    # mechanism from a pure-time-based async optimizer (e.g., async SGD,
    # delayed all-reduce, Decoupled DiLoCo's grace window): in a
    # time-based async optimizer the gradient is always applied; only
    # the synchronization boundary moves. Here the application is gated
    # on a data-driven convergence test (the variance trigger encoded in
    # `decide` above).
    def process_post_backward(
        self,
        adapter_params: dict[int, list[torch.nn.Parameter]],
        active_adapters: tuple[int, ...],
        peer_drift_ms: float,
    ) -> list[AggregationDecision]:
        """Apply the fire/hold decision to each adapter's parameters.

        adapter_params: mapping of adapter_idx -> list of trainable LoRA
            Parameters (lora_A.weight, lora_B.weight, etc.).
        active_adapters: the K indices the router selected this step.
            Inactive adapters are skipped (their .grad is already None or
            zero because peft's set_adapter masked their forward).
        peer_drift_ms: current sideband-reported drift to the (single)
            peer node. If multi-peer, the caller passes the max across
            peers as the conservative bound.
        """
        decisions: list[AggregationDecision] = []
        for adapter_idx in active_adapters:
            decision = self.decide(adapter_idx, peer_drift_ms)
            if decision.fired:
                self._fire_count[adapter_idx] += 1
                self.buffer.clear(adapter_idx)
            else:
                self._hold_count[adapter_idx] += 1
                # Zero gradients so optimizer.step is a no-op for this
                # adapter this step; the buffer retains the snapshot for
                # next time around. detach_() is conditional on the
                # gradient having a grad_fn (mirrors torch.optim.zero_
                # grad's behavior): under FSDP with the default
                # gradient_as_bucket_view=True the .grad tensor is a
                # view into a DDP/FSDP gradient bucket and detach_() is
                # rejected with "Can't detach views in-place". Bucket-
                # view grads do not carry a grad_fn so the conditional
                # skips the detach safely. zero_() works on both
                # bucket-view and standalone grads.
                for p in adapter_params.get(adapter_idx, []):
                    if p.grad is not None:
                        if p.grad.grad_fn is not None:
                            p.grad.detach_()
                        p.grad.zero_()
            decisions.append(decision)
        return decisions

    # Reduce-scatter gating support. The variance-threshold trigger
    # operates post-backward; this `force_hold_active` path
    # is invoked when the runtime's `pre_forward_step` predicted HOLD
    # from PRIOR step decisions and wrapped the forward+backward in
    # FSDP's `no_sync()` context manager. Under no_sync(), reduce-scatter
    # is skipped, so the local non-reduced gradient must be zeroed before
    # the next step to prevent local-gradient accumulation into the next
    # reduce-scatter. This method does that zeroing and emits decisions
    # tagged with reason "gated_predict_hold" so that diagnostics
    # distinguish "trigger fired HOLD on current step's variance" from
    # "prior step's HOLD prediction gated reduce-scatter".
    def force_hold_active(
        self,
        adapter_params: dict[int, list[torch.nn.Parameter]],
        active_adapters: tuple[int, ...],
    ) -> list[AggregationDecision]:
        """Force HOLD on the given active adapters: zero their local
        gradients in place and emit gated_predict_hold decisions. Does
        NOT consult the variance trigger; the gating decision was already
        made at pre_forward_step time based on prior decisions.

        Used by the no_sync()-gated path in `pre_forward_step`. After
        forward+backward under no_sync(), the local gradients are non-
        reduced; applying them via optimizer.step would diverge ranks
        under FULL_SHARD. Zeroing them here keeps the parameter state
        consistent across ranks and makes optimizer.step a no-op.
        """
        decisions: list[AggregationDecision] = []
        for adapter_idx in active_adapters:
            self._hold_count[adapter_idx] += 1
            for p in adapter_params.get(adapter_idx, []):
                if p.grad is not None:
                    if p.grad.grad_fn is not None:
                        p.grad.detach_()
                    p.grad.zero_()
            decisions.append(
                AggregationDecision(
                    adapter_idx, False, float("nan"), "gated_predict_hold"
                )
            )
        return decisions

    @property
    def fire_count(self) -> dict[int, int]:
        return dict(self._fire_count)

    @property
    def hold_count(self) -> dict[int, int]:
        return dict(self._hold_count)
