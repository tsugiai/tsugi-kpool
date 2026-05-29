# tsugi-kpool

[![PyPI version](https://img.shields.io/pypi/v/tsugi-kpool.svg)](https://pypi.org/project/tsugi-kpool/)
[![Python versions](https://img.shields.io/pypi/pyversions/tsugi-kpool.svg)](https://pypi.org/project/tsugi-kpool/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![CI](https://github.com/tsugiai/tsugi-kpool/actions/workflows/ci.yml/badge.svg)](https://github.com/tsugiai/tsugi-kpool/actions/workflows/ci.yml)

K-Pool LoRA SDK. Software productization of TsugiCinema Inc.'s K-Pool LoRA provisional (US App. 64/060,315, filed 2026-05-07) and Infinity provisional (US App. 64/055,093, filed 2026-05-01), packaged as a drop-in extension to PyTorch + PEFT for distributed LoRA fine-tuning that targets straggler-tax recovery on cross-rack training clusters (benchmark methodology in `docs/benchmark_protocol.md`; results pending public release).

## What this is

A Python package that wraps PyTorch distributed and PEFT to implement, at LoRA adapter granularity:

1. **K-out-of-N adapter routing**. Selects K of N adapter modules per step.
2. **Adapter-gradient elastic buffer**. FIFO buffer of adapter gradients prior to aggregation.
3. **Buffer-convergence aggregation**. Triggers aggregation when buffer-variance falls below a threshold instead of on iteration count.
4. **Phase-correction sideband**. Low-bandwidth TCP channel between training nodes carrying drift telemetry, parallel to (not displacing) the NCCL gradient data plane.

**How communication is actually skipped.** The variance trigger runs after each backward pass and decides HOLD or FIRE per adapter (on HOLD it zeros the local adapter gradient, so the optimizer step is a no-op for that adapter). The reduce-scatter itself is skipped *predictively on the next step*: when all currently active adapters most recently HELD, the next forward/backward runs under `no_sync()`. So this is a two-stage design (post-backward HOLD/FIRE, then next-step gated communication), not a same-step "variance fell below the threshold, therefore skip now" rule.

The public API stays close to `peft.LoraConfig` + `accelerate.Accelerator` so adoption friction is minimal.

## What this is not

- Not a fork of OpenDiLoCo. The architectural sibling exists but Prime Intellect's open-source orchestration layer is a separate branch; this SDK goes through `torch.distributed.ProcessGroup` directly.
- Not a full-model Infinity instance. This SDK demonstrates the mechanism at adapter granularity. The transport-layer / full-model instantiation is a separate productization track.

## Install

```bash
pip install tsugi-kpool
```

Or install the unified surface that bundles this SDK with the companion cross-rack reducer:

```bash
pip install tsugi   # exposes tsugi.kpool and tsugi.mend
```

For local development:

```bash
pip install -e ".[dev]"
```

## Minimal usage

K-Pool routes over a **pool of N named adapters**, so the one load-bearing
setup step is building `adapter_0 .. adapter_{N-1}` explicitly (a single
`peft.LoraConfig` applied N times). `KPoolLoraConfig` is the SDK's own config
and is **not** a `peft.PeftConfig`, so pass a real `LoraConfig` to
`get_peft_model` and `KPoolLoraConfig` to `plesio_init`:

```python
from tsugi_kpool import (
    KPoolLoraConfig, plesio_init, apply_kpool_step, post_backward_step,
)
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

base = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")

# Build the N-adapter pool the router selects from.
n_adapters = 8
lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"])
model = get_peft_model(base, lora_config, adapter_name="adapter_0")
for i in range(1, n_adapters):
    model.add_adapter(f"adapter_{i}", lora_config)

config = KPoolLoraConfig(
    r=16, lora_alpha=32, target_modules=("q_proj", "v_proj"),
    n_adapters=n_adapters,
    k_active=2,                            # K-out-of-N per step
    sideband_enabled=True,                 # turn the plesiochronous path on
    aggregation_mode="buffer_convergence",
    buffer_convergence_eps=1e-3,
)

plesio_init(model, config)   # starts the sideband + aggregator
# For FSDP: wrap the model with use_orig_params=True BEFORE plesio_init
# (see docs/architecture.md).

# Per training step:
#   active = apply_kpool_step(model, step=step)   # select K adapters
#   ... forward / loss / backward ...
#   post_backward_step(model, step=step, active=active)
```

`plesio_init` fails fast with a clear error if the `adapter_0 .. adapter_{N-1}`
pool was not built. A fully **runnable** end-to-end version (CPU, ungated
`sshleifer/tiny-gpt2`) is in
[`examples/minimal_finetune.py`](examples/minimal_finetune.py): it builds the
pool, runs a few steps, and prints the per-step HOLD/FIRE decisions. See
[`docs/benchmark_protocol.md`](docs/benchmark_protocol.md) for the benchmark
methodology and [`docs/architecture.md`](docs/architecture.md) for the
mechanism description.

## License

**Apache License, Version 2.0** with its full automatic patent grant. TsugiCinema, Inc. is the Licensor. The Apache-2.0 patent grant in Section 3 extends to TsugiCinema's K-Pool LoRA (US App. 64/060,315) and Infinity (US App. 64/055,093) patent estates AS PRACTICED BY THE SDK CODE AS DISTRIBUTED. See `LICENSE` for the NOTICE preamble explaining the doctrine and the full Apache-2.0 license text.

The license posture reflects an open-source-first strategy: the SDK ships under Apache-2.0 with a full automatic patent grant for the embodiment as distributed, and is packaged together with the companion `tsugi-mend` SDK under the unified `pip install tsugi` product surface.

## Status

**Pre-Alpha (0.1.1).** APIs are stabilizing and may change before v1.0. Published to PyPI as `tsugi-kpool`; also reachable through the unified `tsugi` meta-package as `tsugi.kpool`.
