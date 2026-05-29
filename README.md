# tsugi-kpool

[![PyPI version](https://img.shields.io/pypi/v/tsugi-kpool.svg)](https://pypi.org/project/tsugi-kpool/)
[![Python versions](https://img.shields.io/pypi/pyversions/tsugi-kpool.svg)](https://pypi.org/project/tsugi-kpool/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![CI](https://github.com/tsugiai/tsugi-kpool/actions/workflows/ci.yml/badge.svg)](https://github.com/tsugiai/tsugi-kpool/actions/workflows/ci.yml)

K-Pool LoRA SDK. Software productization of TsugiCinema Inc.'s K-Pool LoRA provisional (US App. 64/060,315, filed 2026-05-07) and Infinity provisional (US App. 64/055,093, filed 2026-05-01), packaged as a drop-in extension to PyTorch + PEFT for distributed LoRA fine-tuning with measurable straggler-tax recovery on cross-rack training clusters.

## What this is

A Python package that wraps PyTorch distributed and PEFT to implement, at LoRA adapter granularity:

1. **K-out-of-N adapter routing**. Selects K of N adapter modules per step.
2. **Adapter-gradient elastic buffer**. FIFO buffer of adapter gradients prior to aggregation.
3. **Buffer-convergence aggregation**. Triggers aggregation when buffer-variance falls below a threshold instead of on iteration count.
4. **Phase-correction sideband**. Low-bandwidth TCP channel between training nodes carrying drift telemetry, parallel to (not displacing) the NCCL gradient data plane.

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

```python
from tsugi_kpool import KPoolLoraConfig, plesio_init
from transformers import AutoModelForCausalLM
from peft import get_peft_model

model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")

config = KPoolLoraConfig(
    r=16,                        # standard LoRA rank
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    n_adapters=8,                # N
    k_active=2,                  # K-out-of-N per step
    sideband_addr="tcp://0.0.0.0:51820",  # phase-correction sideband
    buffer_convergence_eps=1e-3, # buffer-variance trigger threshold
)

model = get_peft_model(model, config)
plesio_init(model, config)   # starts the sideband + aggregator threads

# from here, train as you would any peft+accelerate fine-tune
```

A runnable shape-demonstration of the full surface is in
[`examples/minimal_finetune.py`](examples/minimal_finetune.py). It loads a
gated base model (`meta-llama/Meta-Llama-3-8B`), so running it end-to-end
requires Hugging Face authentication and a GPU; the SDK wiring it shows
(config, `get_peft_model`, `plesio_init` / `plesio_shutdown`) imports and
constructs on CPU without either. See
[`docs/benchmark_protocol.md`](docs/benchmark_protocol.md) for the
benchmark methodology and [`docs/architecture.md`](docs/architecture.md)
for the mechanism description.

## License

**Apache License, Version 2.0** with its full automatic patent grant. TsugiCinema, Inc. is the Licensor. The Apache-2.0 patent grant in Section 3 extends to TsugiCinema's K-Pool LoRA (US App. 64/060,315) and Infinity (US App. 64/055,093) patent estates AS PRACTICED BY THE SDK CODE AS DISTRIBUTED. See `LICENSE` for the NOTICE preamble explaining the doctrine and the full Apache-2.0 license text.

The license posture reflects an open-source-first strategy: the SDK ships under Apache-2.0 with a full automatic patent grant for the embodiment as distributed, and is packaged together with the companion `tsugi-mend` SDK under the unified `pip install tsugi` product surface.

## Status

**Pre-Alpha (0.1.1).** APIs are stabilizing and may change before v1.0. Published to PyPI as `tsugi-kpool`; also reachable through the unified `tsugi` meta-package as `tsugi.kpool`.
