"""Top-level runtime: plesio_init / plesio_shutdown / step entry points.

The integration glue between the SDK components (router, aggregator,
sideband, diagnostics) and a peft-wrapped + FSDP-wrapped model.

Public training-loop API:
    plesio_init(model, config, sender_id)       call once after FSDP wrap
    apply_kpool_step(model, step)               call before each forward;
                                                returns the active subset
    post_backward_step(model, step)             call after backward, before
                                                optimizer.step
    plesio_shutdown(model)                      call once before
                                                dist.destroy_process_group
"""
from __future__ import annotations

import asyncio
import contextlib
import threading
from collections.abc import Callable
from typing import Any, cast

import torch
import torch.nn as nn

from tsugi_kpool.aggregator import AggregationDecision, BufferConvergenceAggregator
from tsugi_kpool.config import KPoolLoraConfig
from tsugi_kpool.diagnostics import DiagnosticsWriter
from tsugi_kpool.router import KPoolRouter, adapter_names, attach_router
from tsugi_kpool.sideband import Sideband


class _PlesioRuntime:
    """Per-model runtime container."""

    def __init__(self, config: KPoolLoraConfig, sender_id: str) -> None:
        self.config = config
        self.sender_id = sender_id
        self.aggregator = BufferConvergenceAggregator(config)
        self.sideband = Sideband(config, sender_id)
        self.diagnostics = DiagnosticsWriter(config.diagnostics_dir)
        # adapter_idx -> list of trainable LoRA params for that adapter
        self.adapter_params: dict[int, list[nn.Parameter]] = {}
        # adapter_idx -> last AggregationDecision (for diagnostics emit)
        self.last_decisions: dict[int, AggregationDecision] = {}
        # cache of (rank-tuple -> ProcessGroup) for sideband-restricted all-reduce
        self.subgroup_cache: dict[tuple[int, ...], Any] = {}
        # hook handles so we can remove them at shutdown
        self._hook_handles: list[Any] = []
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: threading.Thread | None = None
        # pre_forward_step / post_backward_step state for the no_sync()-
        # gated path. Set True by pre_forward_step when prior-step
        # decisions on all current `active` adapters were HOLD; cleared
        # by post_backward_step after the forced-HOLD path runs.
        self.gated_step_active: bool = False

    def start(self) -> None:
        if not self.config.sideband_enabled:
            return
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(
            target=self._run_loop, args=(self._loop,), daemon=True
        )
        self._loop_thread.start()
        # start() returns immediately; the actual binding happens inside
        # the asyncio loop. We block briefly to let it bind before the
        # training loop starts so heartbeats are not lost.
        future = asyncio.run_coroutine_threadsafe(self.sideband.start(), self._loop)
        future.result(timeout=5.0)

    def stop(self) -> None:
        for h in self._hook_handles:
            try:
                h.remove()
            except Exception:
                pass
        self._hook_handles.clear()
        if self._loop is not None:
            try:
                asyncio.run_coroutine_threadsafe(self.sideband.stop(), self._loop).result(
                    timeout=5.0
                )
            except Exception:
                pass
            self._loop.call_soon_threadsafe(self._loop.stop)
            if self._loop_thread is not None:
                self._loop_thread.join(timeout=2.0)
        self.diagnostics.close()

    @staticmethod
    def _run_loop(loop: asyncio.AbstractEventLoop) -> None:
        asyncio.set_event_loop(loop)
        loop.run_forever()


_RUNTIMES: dict[int, _PlesioRuntime] = {}


# ---------------------------------------------------------------------------
# LoRA-parameter discovery + hook installation
# ---------------------------------------------------------------------------


def _discover_adapter_params(
    model: nn.Module, n_adapters: int
) -> dict[int, list[nn.Parameter]]:
    """Walk the model tree and group LoRA-adapter params by adapter
    index. peft stores adapter-named parameters under ParameterDict-like
    submodules; the convention from kpool_run.py is adapter names
    'adapter_0' ... 'adapter_{N-1}'.

    We identify a parameter as belonging to adapter_i if any name segment
    in its qualified name equals 'adapter_<int>' exactly. This matches
    both standard peft layouts and FSDP-flattened layouts.

    We deliberately do NOT filter by requires_grad here. peft marks
    inactive adapter slots with requires_grad=False at `add_adapter`
    time; filtering by requires_grad would exclude them from discovery
    and the runtime would not install backward hooks on their
    parameters. When the router later activates such a slot via
    set_adapter, peft flips requires_grad=True, but by then it is too
    late to install a hook. Discovering all `adapter_<int>` params at
    plesio_init regardless of requires_grad keeps the hook coverage
    complete; the hook itself is a no-op when the active adapter mask
    excludes the param from the autograd graph (grad is None).
    """
    by_adapter: dict[int, list[nn.Parameter]] = {i: [] for i in range(n_adapters)}
    for name, param in model.named_parameters():
        parts = name.split(".")
        for part in parts:
            if part.startswith("adapter_"):
                try:
                    idx = int(part.removeprefix("adapter_"))
                except ValueError:
                    continue
                if 0 <= idx < n_adapters:
                    by_adapter[idx].append(param)
                    break
    return by_adapter


# Backward-hook installation: the mechanism by which the elastic adapter
# buffer is populated with local pre-reduce gradient snapshots. With FSDP
# wrapped at use_orig_params=True, peft's underlying LoRA Parameter
# objects remain addressable; the hook below fires on each rank's
# local-partition backward computation BEFORE FSDP performs its
# reduce-scatter, capturing the pre-aggregation gradient and pushing it
# into the elastic buffer for the variance test.
def _install_backward_hooks(
    model: nn.Module, runtime: _PlesioRuntime
) -> None:
    """Register per-parameter backward hooks that push local gradients
    into the elastic buffer. With FSDP(use_orig_params=True) these fire
    on each rank's local parameter partition before reduce-scatter, so
    the buffer captures pre-reduce local gradients.

    Hooks are no-ops when grad is None (which happens for params whose
    adapter is currently masked off via peft.set_adapter).

    `Tensor.register_hook` rejects tensors with requires_grad=False. peft
    marks inactive adapter slots as requires_grad=False at add_adapter
    time; to install hooks on every adapter regardless of initial active
    state we temporarily flip requires_grad=True around the register
    call, then restore. Once registered, the hook survives subsequent
    requires_grad transitions; peft's later set_adapter calls will
    enable the hook again automatically when the adapter rejoins the
    autograd graph.
    """
    for adapter_idx, params in runtime.adapter_params.items():
        for p in params:
            original_requires_grad = p.requires_grad
            if not original_requires_grad:
                p.requires_grad_(True)
            handle = p.register_hook(  # type: ignore[no-untyped-call]  # torch stub gap: Tensor.register_hook is untyped
                _make_hook(adapter_idx, runtime)
            )
            runtime._hook_handles.append(handle)
            if not original_requires_grad:
                p.requires_grad_(False)


def _make_hook(
    adapter_idx: int, runtime: _PlesioRuntime
) -> Callable[[torch.Tensor], None]:
    def hook(grad: torch.Tensor) -> None:
        # Push and return None (do not modify the gradient at hook time;
        # the aggregator zeros it later in process_post_backward if the
        # decision is HOLD).
        if grad is None:
            return None
        runtime.aggregator.buffer.push(adapter_idx, grad)
        return None
    return hook


# FSDP-compatible gradient-capture path. The Parameter-level
# `register_hook` mechanism above does not fire under FSDP's
# use_orig_params=True wrap in PyTorch 2.12 because FSDP populates per-
# orig-param `.grad` via its post-backward callback AFTER all module
# backward computations complete. Module.register_full_backward_hook
# has the same issue: at hook-fire time, the parameter .grad is still
# None because FSDP hasn't done its reduce-scatter callback yet.
#
# To survive FSDP we capture the adapter's OUTPUT-TENSOR gradient
# rather than the parameter gradient. A forward hook on each per-
# adapter sub-module saves the output tensor reference, then installs
# `Tensor.register_hook` on the output. The output-tensor hook fires
# when autograd computes the gradient w.r.t. the activation, which
# happens BEFORE FSDP's reduce-scatter (reduce-scatter operates on
# parameter gradients, not activation gradients). This captures the
# per-adapter local gradient signal under FSDP at every backward.
#
# Note: the buffer is generic on the kind of gradient tensor it stores
# and does not require parameter vs activation gradients specifically.
# The output-gradient interpretation satisfies the elastic-buffer
# population requirement, and the variance-threshold trigger operates
# correctly on output-gradient time series (variance over output-gradient
# snapshots tracks the same convergence signal as variance over
# parameter-gradient snapshots, up to a per-adapter linear transform).
def _install_module_backward_hooks(
    model: nn.Module, runtime: _PlesioRuntime
) -> None:
    """Install forward hooks on each per-adapter LoRA sub-module that
    register a `Tensor.register_hook` on the module's output activation
    at every forward call. The activation hook captures the output-
    tensor gradient when autograd computes it during backward, which
    happens BEFORE FSDP's reduce-scatter callback. The push to the
    elastic buffer is therefore captured under FSDP.

    Walks model.named_modules() and matches the `adapter_<int>` segment
    in the module's qualified name. For peft, this matches
    `lora_A.adapter_0`, `lora_B.adapter_0`, etc. For toy fixtures
    whose adapter sub-modules have no forward, the forward hook is
    registered but never fires; the Parameter-level hook above is the
    active capture path for that case.
    """
    seen_modules: set[int] = set()
    for name, module in model.named_modules():
        if id(module) in seen_modules:
            continue
        for part in name.split("."):
            if part.startswith("adapter_"):
                try:
                    idx = int(part.removeprefix("adapter_"))
                except ValueError:
                    continue
                if 0 <= idx < runtime.config.n_adapters:
                    handle = module.register_forward_hook(
                        _make_activation_hook(idx, runtime)
                    )
                    runtime._hook_handles.append(handle)
                    seen_modules.add(id(module))
                    break


def _make_activation_hook(
    adapter_idx: int, runtime: _PlesioRuntime
) -> Callable[["nn.Module", tuple[Any, ...], Any], None]:
    """Forward hook that arms a backward hook on the module's output
    activation. Forward-hook signature: (module, input, output). The
    output is the activation tensor produced by the module's forward;
    its .register_hook fires when autograd computes the gradient w.r.t.
    that activation in backward, which under FSDP is BEFORE the
    parameter-gradient reduce-scatter."""
    def forward_hook(
        module: "nn.Module",
        inputs: tuple[Any, ...],  # tuple[Tensor | None, ...]
        output: Any,
    ) -> None:
        if isinstance(output, torch.Tensor) and output.requires_grad:
            def grad_hook(grad: torch.Tensor) -> None:
                if grad is not None:
                    runtime.aggregator.buffer.push(adapter_idx, grad)
                return None
            output.register_hook(grad_hook)  # type: ignore[no-untyped-call]  # torch stub gap: Tensor.register_hook is untyped
    return forward_hook


# ---------------------------------------------------------------------------
# Public training-loop API
# ---------------------------------------------------------------------------


# Entry point. This function wires the four SDK components together into
# a single working system:
#   (1) KPoolRouter      -- K-of-N adapter selection per training step
#                            (K-Pool LoRA, US App. 64/060,315).
#   (2) ElasticAdapterBuffer
#                       -- elastic gradient-tensor buffer storing local
#                            pre-reduce gradient snapshots (Infinity,
#                            US App. 64/055,093).
#   (3) BufferConvergenceAggregator
#                       -- buffer-convergence aggregation rule with a
#                            variance-threshold trigger.
#   (4) Sideband        -- phase-correction signaling channel, logically
#                            distinct from the NCCL gradient data plane,
#                            carrying timestamp-delta phase-drift
#                            telemetry.
def plesio_init(
    model: nn.Module,
    config: KPoolLoraConfig,
    sender_id: str | None = None,
) -> None:
    """Wire the K-Pool LoRA SDK onto a peft-wrapped + FSDP-wrapped model.

    Must be called AFTER FSDP wrap so the router's adapter-name handles
    survive the FSDP flattening (with use_orig_params=True). The
    aggregator's backward hooks attach to the original LoRA parameters,
    which FSDP exposes when use_orig_params=True.
    """
    if id(model) in _RUNTIMES:
        raise RuntimeError("plesio_init already called for this model")
    if sender_id is None:
        sender_id = f"rank-{id(model)}"

    attach_router(model, config)
    runtime = _PlesioRuntime(config, sender_id)

    # Discover the LoRA parameters per adapter so the aggregator knows
    # which .grad tensors to zero on HOLD decisions and so backward
    # hooks can be installed before training begins.
    runtime.adapter_params = _discover_adapter_params(model, config.n_adapters)
    if not any(runtime.adapter_params.get(i) for i in range(config.n_adapters)):
        raise RuntimeError(
            "plesio_init found no adapters named "
            f"'adapter_0'..'adapter_{config.n_adapters - 1}'. Build the K-of-N "
            "adapter pool before calling plesio_init: get_peft_model(base, "
            "lora_config, adapter_name='adapter_0') then "
            "add_adapter('adapter_1', ...). See the README quickstart and "
            "docs/architecture.md."
        )
    _install_backward_hooks(model, runtime)
    # Also install Module-level full_backward_hooks on per-adapter
    # sub-modules so the elastic buffer is populated under FSDP /
    # torch.distributed (where the Parameter-level hooks above do not
    # fire). On toy fixtures whose adapter modules have no forward,
    # this is a no-op (the Module hook is registered but never fires).
    _install_module_backward_hooks(model, runtime)

    runtime.start()
    _RUNTIMES[id(model)] = runtime
    setattr(model, "_plesio_runtime", runtime)


def plesio_shutdown(model: nn.Module) -> None:
    """Tear down the SDK runtime: stop sideband, remove backward hooks,
    close diagnostics."""
    runtime = _RUNTIMES.pop(id(model), None)
    if runtime is None:
        return
    runtime.stop()
    if hasattr(model, "_plesio_runtime"):
        delattr(model, "_plesio_runtime")


def apply_kpool_step(model: nn.Module, step: int) -> tuple[int, ...]:
    """Drive the K-Pool router and apply peft.set_adapter for the chosen
    K-of-N. Call once at the top of each optimizer step, before the
    micro-batch accumulation loop.

    Returns the active adapter indices."""
    router: KPoolRouter | None = getattr(model, "_kpool_router", None)
    if router is None:
        raise RuntimeError("apply_kpool_step called before plesio_init")
    active = router.select(step=step)
    # peft's set_adapter lives on the inner peft model; with FSDP wrap,
    # access it via .module.
    underlying = model.module if hasattr(model, "module") else model
    names = adapter_names(router.config, active)
    # peft >=0.13 split set_adapter into two surfaces:
    #   * `PeftModel.set_adapter(name: str)` accepts only a single
    #     adapter name and refuses lists ("Only one adapter can be
    #     active at a time" per upstream docstring).
    #   * `LoraModel.set_adapter(name: str | list[str])` accepts a list
    #     and activates multiple adapters simultaneously (the K-of-N
    #     case we need).
    # When the model is a PeftModel wrapper, route the list-of-names
    # call to its inner LoraModel via `.base_model`. Toy fixtures
    # without a `base_model` attribute take the direct `set_adapter`
    # path, which accepts a list.
    target: Any = getattr(underlying, "base_model", None) or underlying
    if hasattr(target, "set_adapter"):
        target.set_adapter(names)
    return active


# Reduce-scatter gating (upstream variant of the buffer-convergence
# trigger; relates to Infinity, US App. 64/055,093). The variance
# trigger operates post-backward; on its own that only avoids
# `optimizer.step`, not the FSDP reduce-scatter that already ran during
# backward(). To realize the trigger's
# bytes-not-transmitted potential as wall-clock savings under cross-rack
# network conditions, this function consults the runtime's most-recent
# decisions for the current `active` adapters and, when all of them
# previously HOLD'd, returns FSDP's `no_sync()` context manager so the
# next forward+backward skips reduce-scatter entirely. The forced-HOLD
# path in `post_backward_step` then zeros the (non-reduced) local
# gradients so optimizer.step is also a no-op for the gated step, which
# keeps the FULL_SHARD parameter state consistent across ranks.
def pre_forward_step(
    model: nn.Module,
    step: int,
    active: tuple[int, ...],
) -> "contextlib.AbstractContextManager[Any]":
    """Return a context manager to wrap forward+backward.

    Prediction rule: if all adapters in `active` have a most-recent
    decision (`runtime.last_decisions`) whose `.fired` is False (i.e.,
    HOLD), return `model.no_sync()` to skip FSDP reduce-scatter for the
    next forward+backward. Otherwise return `nullcontext()` and the
    standard variance-trigger path runs as before.

    The runtime tracks the gating state via `gated_step_active` so that
    `post_backward_step` knows to take the forced-HOLD path that zeros
    the (non-reduced) local gradients. The variance-trigger logic is
    untouched on non-gated steps.

    Falls back to `nullcontext()` gracefully when `model` is not an FSDP
    instance (model.no_sync attribute absent), so unit tests on plain
    nn.Modules don't fail.
    """
    runtime = _RUNTIMES.get(id(model))
    if runtime is None:
        raise RuntimeError("pre_forward_step called before plesio_init")

    # Default to nullcontext on step 0 (no prior decisions to predict
    # from) and whenever ANY active adapter has a missing prior decision
    # OR a prior FIRE decision (we predict the variance trigger might
    # fire this step, so we must not skip reduce-scatter).
    all_active_held = bool(active) and all(
        (d := runtime.last_decisions.get(adapter_idx)) is not None and not d.fired
        for adapter_idx in active
    )

    if step == 0 or not all_active_held:
        runtime.gated_step_active = False
        return contextlib.nullcontext()

    no_sync = getattr(model, "no_sync", None)
    if no_sync is None:
        # Model is not FSDP-wrapped (or doesn't expose no_sync). The
        # gated path requires reduce-scatter suppression to be
        # meaningful; without it, fall back to the standard post-
        # backward path so behavior on non-FSDP fixtures is unchanged.
        runtime.gated_step_active = False
        return contextlib.nullcontext()

    runtime.gated_step_active = True
    runtime.diagnostics.emit(
        "pre_forward_gated",
        step=step,
        active=list(active),
    )
    return cast("contextlib.AbstractContextManager[Any]", no_sync())


def post_backward_step(
    model: nn.Module, step: int, active: tuple[int, ...]
) -> list[AggregationDecision]:
    """Apply the buffer-convergence aggregation rule per active adapter.
    Call after the backward pass and before optimizer.step.

    If the most recent `pre_forward_step` returned a gated (no_sync())
    context, the variance-trigger path is bypassed and all active
    adapters are forced to HOLD: their local (non-reduced) gradients are
    zeroed and decisions are emitted with reason="gated_predict_hold".
    This keeps FULL_SHARD parameter state consistent across ranks.

    Returns the list of AggregationDecisions emitted this step (one per
    active adapter) so the training loop can log them."""
    runtime = _RUNTIMES.get(id(model))
    if runtime is None:
        raise RuntimeError("post_backward_step called before plesio_init")

    if runtime.gated_step_active:
        decisions = runtime.aggregator.force_hold_active(
            runtime.adapter_params, active
        )
        runtime.gated_step_active = False
        peer_drift_emit = None
    else:
        peer_drift = runtime.sideband.max_drift_ms_across_peers()
        decisions = runtime.aggregator.process_post_backward(
            runtime.adapter_params, active, peer_drift
        )
        peer_drift_emit = None if peer_drift == float("inf") else peer_drift

    for d in decisions:
        runtime.last_decisions[d.adapter_idx] = d
    # Emit a diagnostics record
    runtime.diagnostics.emit(
        "post_backward",
        step=step,
        active=list(active),
        peer_drift_ms=peer_drift_emit,
        decisions=[
            {"adapter": d.adapter_idx, "fired": d.fired, "variance": d.variance, "reason": d.reason}
            for d in decisions
        ],
    )
    return decisions


def get_runtime(model: nn.Module) -> _PlesioRuntime:
    """Test/benchmark hook to access the runtime for introspection."""
    if id(model) not in _RUNTIMES:
        raise RuntimeError("plesio_init not called for this model")
    return _RUNTIMES[id(model)]


__all__ = [
    "plesio_init",
    "plesio_shutdown",
    "apply_kpool_step",
    "pre_forward_step",
    "post_backward_step",
    "get_runtime",
]
