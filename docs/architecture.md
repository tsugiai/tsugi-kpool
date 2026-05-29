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
