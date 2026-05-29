"""torch.distributed ProcessGroup wrapper.

Thin shim that lets the K-Pool LoRA SDK coexist with stock NCCL for the
gradient data plane. The sideband + aggregator run alongside; NCCL still
carries the all-reduce inside FSDP's reduce-scatter when the aggregator
fires.

This is a Python-level wrapper around the standard torch.distributed
API. A custom C++ ProcessGroup backend is a possible future enhancement
for tighter NCCL integration.
"""
from __future__ import annotations

from typing import Iterable, Optional

import torch.distributed as dist
from torch import Tensor


def is_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def world_size() -> int:
    return dist.get_world_size() if is_initialized() else 1


def rank() -> int:
    return dist.get_rank() if is_initialized() else 0


def all_reduce_adapter_grad(
    grad: Tensor, group: Optional[dist.ProcessGroup] = None
) -> Tensor:
    """All-reduce one adapter's gradient tensor with mean reduction.

    Currently used as a manual code path; the normal SDK flow relies on
    FSDP's built-in reduce-scatter happening automatically. This helper
    exists so the aggregator can drive an explicit all-reduce after
    draining the elastic buffer (Phase 2 path)."""
    if not is_initialized():
        return grad
    dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=group)
    grad /= world_size()
    return grad


def make_subgroup(ranks: Iterable[int]) -> dist.ProcessGroup:
    """Create a process group containing only the given ranks. Used by
    the sideband-restricted all-reduce path so we only reduce across
    peers whose drift is within max_drift_ms.

    Subgroup creation is collective: every rank in the world must call
    this with the same `ranks` argument. This is fine for symmetric
    plesiochronous setups but limits how dynamic the peer set can be
    in practice; Phase 2 explores a non-collective alternative."""
    if not is_initialized():
        raise RuntimeError("torch.distributed not initialized")
    group: dist.ProcessGroup = dist.new_group(ranks=sorted(ranks))
    return group


def all_reduce_subset(
    grad: Tensor, ranks: Iterable[int], cache: dict[tuple[int, ...], dist.ProcessGroup]
) -> Tensor:
    """All-reduce restricted to a subset of ranks. The caller passes a
    cache dict so we do not recreate the subgroup each step."""
    if not is_initialized():
        return grad
    key = tuple(sorted(ranks))
    group = cache.get(key)
    if group is None:
        group = make_subgroup(key)
        cache[key] = group
    return all_reduce_adapter_grad(grad, group=group)
