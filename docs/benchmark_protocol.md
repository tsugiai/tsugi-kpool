# Benchmark protocol

The reproduction contract for the K-Pool LoRA cross-rack measurement. The
config files referenced below (under `benchmarks/llama3_8b_lora/`) are not
shipped in this release; the protocol itself is the reproduction pointer and
can be re-derived on the stated cluster.

## Cluster

- 2x rented 8xH100 nodes from a cloud GPU provider
- Commodity Ethernet between nodes (sub-25 Gbps effective inter-node;
  this deliberately introduces the cross-rack synchronization penalty
  the SDK targets)
- Stock NCCL, stock PyTorch (>= 2.4), no out-of-tree kernels

## Workload

- Model: meta-llama/Meta-Llama-3-8B
- Task: instruction-tuning fine-tune on 50k examples from databricks-dolly-15k + alpaca-cleaned (mixed)
- Tokenizer: meta-llama/Meta-Llama-3-8B tokenizer, no modifications
- Sequence length: 2048
- Global batch size: 64
- Per-device micro-batch: 4
- Gradient accumulation: 4

## LoRA hyperparameters (vanilla baseline)

- r=16, alpha=32, dropout=0.0
- target_modules: q_proj, v_proj
- bias: none
- optimizer: AdamW, lr=2e-4, weight_decay=0.0
- scheduler: cosine, warmup 100 steps
- training: 2,000 steps

## K-Pool LoRA hyperparameters

- All vanilla settings PLUS:
- n_adapters=8, k_active=2
- routing_strategy=round_robin (loss_aware is optional and disclosed if used)
- sideband_enabled=True, sideband_heartbeat_ms=50, max_drift_ms=250
- aggregation_mode=buffer_convergence
- buffer_capacity=32, buffer_convergence_eps=1e-3
  (eps is tuned per workload; calibrate to the observed gradient-variance scale)

## Metrics

Primary:
- Tokens-per-second (mean across steps 100-1900, throwing out first
  100 warmup and last 100 cooldown)
- Final loss at step 2000
- Loss curve trajectory at steps 100, 250, 500, 1000, 1500, 2000

Secondary:
- Buffer variance trajectory (per-adapter)
- Sideband drift trajectory between the two nodes
- NCCL all-reduce calls fired by the aggregator (count and bytes)

## PASS / FAIL declaration

PASS, all conditions must hold:
- Tokens/sec uplift >= 15% over vanilla baseline (one-sided 95% CI)
- Final loss within +0.05 of vanilla baseline
- Loss curve does not diverge for any matched checkpoint
- No NaN / inf / NCCL hangs across 5 paired runs

FAIL, any one of these triggers:
- Tokens/sec uplift < 5%
- Final loss > 0.05 above vanilla
- Loss curve diverges at any matched checkpoint
- NCCL hangs / segfaults that cannot be recovered without restart

INCONCLUSIVE:
- Tokens/sec uplift 5-15% but loss-curve clean. Report the result
  honestly with its CI rather than reframing; extend the plan only with
  a pre-registered rationale.

## Reproducibility checklist

- All random seeds pinned in the run config
- All package versions pinned in `pyproject.toml` and a lockfile
- Cluster image pinned via a Dockerfile
- Exact command lines captured in the run script
- Data sources: HuggingFace dataset URLs with revision pins
- Run identifiers + tracking URLs logged in `diagnostics_dir`

## Common deviation cases

- Spot-instance preemption mid-run: discard run, do not retry on
  same instance pool
- NCCL timeout (NCCL_TIMEOUT default 30 minutes): bump to 60, retry
  once, log incident; if reproduces, the run is a FAIL signal
- HuggingFace dataset version drift between runs: do not pin to "latest",
  always pin to a specific commit hash
