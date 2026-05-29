# Architecture

## Layers

```
+-----------------------------------------------------------+
| user training script (transformers / peft / accelerate)   |
+-----------------------------------------------------------+
| KPoolLoraConfig + plesio_init / plesio_shutdown           | <- public API
+-----------------------------------------------------------+
| KPoolRouter         BufferConvergenceAggregator           |
|  (K-of-N adapter     +- ElasticAdapterBuffer              |
|   selection)         +- buffer-variance trigger           |
+-----------------------------------------------------------+
| Sideband (asyncio + TCP)    backend.all_reduce_adapter_grad|
|  (phase-correction          (NCCL-backed when distributed) |
|   control plane)                                           |
+-----------------------------------------------------------+
| torch.distributed (NCCL) | OS TCP stack                    |
+-----------------------------------------------------------+
```

## Component overview

| Component | Role |
|---|---|
| `KPoolRouter` | K-out-of-N adapter selection |
| `ElasticAdapterBuffer` | per-adapter elastic gradient buffer |
| `BufferConvergenceAggregator` | buffer-convergence aggregation rule |
| `Sideband` | phase-correction sideband channel |
| `backend.all_reduce_adapter_grad` | engineering glue; NCCL gradient data plane stays vanilla |

The router and the buffer / aggregator / sideband layers are deliberately
independent: K-Pool routing works without the sideband (synchronous
aggregation), and the sideband mechanics work without K-Pool routing
(single-adapter mode). Both turn on together for the full cross-rack run.

## Why two patents map to one SDK

K-Pool LoRA describes WHAT is synchronized (which K of N adapters
participate) and at which granularity. Infinity describes HOW the
synchronization itself works (elastic buffer + drift-aware aggregation
+ control-plane / data-plane separation). The SDK is a single product
because the use case (productized LoRA fine-tuning with measurable
straggler-tax recovery) needs both.

## FSDP integration

`plesio_init(model, config)` must be called **after** the model is
FSDP-wrapped, and FSDP must be configured with **`use_orig_params=True`**. The
aggregator's backward hooks attach to the original LoRA parameters, which FSDP
only exposes under `use_orig_params=True`; under FSDP the runtime additionally
installs module-level activation-gradient hooks, because Parameter-level hooks
do not fire inside FSDP's post-backward callback.

```python
model = FSDP(get_peft_model(base, lora_config), use_orig_params=True)
plesio_init(model, config)   # after the FSDP wrap
```

End-to-end FSDP behavior is validated on CUDA hardware. The public test suite
covers the single-process and MPS paths; the multi-process / FSDP integration
test is skipped on Apple silicon (a PyTorch MPS-FSDP limitation, not an SDK
issue). For introspection (per-adapter HOLD/FIRE decisions, fire counts) in
tests and benchmarks, `get_runtime(model)` returns the live runtime and is
exported from the package root (`from tsugi_kpool import get_runtime`).

## Sideband trust boundary

The sideband is an unauthenticated, low-bandwidth TCP control plane for a
trusted cluster fabric. It binds to loopback by default, enforces a
source-address peer allow-list (from `sideband_peers`), and bounds inbound
frame size. It is not meant to be exposed to untrusted networks and does not
yet carry cryptographic message authentication (planned for a later release).
See `SECURITY.md`.

## What is intentionally not here

- Full-model gradient synchronization. The aggregator operates on adapter
  gradients only; the base model parameters flow through stock FSDP.
  Phase 2 productization expands to full-model.

- Custom C++ ProcessGroup backend. The wrapper is Python-level; a C++
  backend is a possible future enhancement.

- OpenDiLoCo or Hivemind dependencies. OpenDiLoCo may be unmaintained as
  of 2026; this SDK routes around it.

- Multi-rack 3+ node sideband. The current sideband is two-node;
  N-node sideband is a future enhancement.

## File-by-file reading order

For someone joining the codebase, read in this order:

1. `src/tsugi_kpool/config.py`. what every knob does
2. `src/tsugi_kpool/router.py`. K-Pool routing
3. `src/tsugi_kpool/aggregator.py`. Infinity-at-adapter-granularity
4. `src/tsugi_kpool/sideband.py`. phase-correction channel
5. `src/tsugi_kpool/runtime.py`. how the layers compose at runtime
6. `docs/benchmark_protocol.md`. the cross-rack benchmark protocol
