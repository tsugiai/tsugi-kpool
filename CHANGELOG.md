# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
while APIs stabilize.

## [Unreleased]

### Added
- Added `recommend_buffer_convergence_eps`, a torch-free helper for choosing a
  buffer-convergence epsilon from observed per-adapter variance samples.

### Changed
- Tightened `KPoolLoraConfig` construction-time validation. Invalid numeric
  values that could silently disable or invert runtime behavior now raise
  `ValueError`.

## [0.1.2] - 2026-05-30

### Added
- Runnable quickstart and a CPU-only end-to-end example. The README and example
  now build the K-adapter pool correctly (a peft `LoraConfig` plus named
  `adapter_0..N-1`), enable the buffer-convergence aggregation and the sideband,
  and fail fast with a clear error if no adapters are discovered.
- Deterministic CPU proxy tests for the core mechanism: a learning-quality-delta
  proxy versus synchronous aggregation, a parametrized `eps` sweep asserting a
  monotonic non-increasing HOLD rate, and an exactly-K active-adapter assertion
  across routing strategies.
- FSDP integration recipe in the docs (the wrap order and the
  `use_orig_params=True` requirement) and `get_runtime` exported from the
  package root.
- A blocking CI content-scan gate over changed lines (guards every future PR),
  alongside the existing `mypy` gate.

### Changed
- README mechanism description corrected: the comm-skipping trigger is a
  one-step-delayed predictive gate (the post-backward step zeroes gradients; the
  pre-forward step decides whether to skip the next reduce-scatter), not a
  same-step "variance below eps then aggregate" rule. Throughput wording
  softened to forward-looking (no benchmark numbers are shipped in this release).

### Security
- Sideband control plane hardened (control-plane only): the default bind moved
  to loopback, the peer allow-list is enforced against the configured peers, and
  inbound frames are size-bounded. A trusted-fabric note was added to
  SECURITY.md and the architecture doc. Authenticated, secure-by-default
  transport remains planned for 0.2.0.

## [0.1.1] - 2026-05-27

### Changed
- Refreshed the published project description/metadata (the prior PyPI long
  description was stale).

### Added
- Governance scaffolding: SECURITY, CONTRIBUTING, CODE_OF_CONDUCT, issue/PR
  templates, and Dependabot config.
- Hardened CI: blocking `mypy`, advisory `pip-audit`, current `actions/*` majors.
- Tokenless PyPI Trusted-Publishing release workflow + RELEASING.md.

## [0.1.0] - 2026-05-27

### Added

- Initial public release.
