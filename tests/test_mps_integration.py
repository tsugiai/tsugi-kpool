"""MPS single-device integration tests.

Validates that the SDK + real peft + tiny HF model + plesio_init +
forward + backward operate correctly when the model is moved to
torch.device("mps") rather than CPU. MPS is an accelerator-class device
(analogous to CUDA in semantic role, distinct from CPU).

Note: torch.mps does NOT implement the full CUDA-style device-handle
API (current_device, is_initialized, ...). That limitation only affects
FSDP; single-device MPS autograd is fully functional in torch 2.12.
"""
from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import pytest

torch = pytest.importorskip("torch")
peft = pytest.importorskip("peft")
transformers = pytest.importorskip("transformers")

from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from tsugi_kpool.config import KPoolLoraConfig
from tsugi_kpool.runtime import (
    apply_kpool_step,
    get_runtime,
    plesio_init,
    plesio_shutdown,
    post_backward_step,
)


def _mps_available() -> bool:
    return torch.backends.mps.is_available() and torch.backends.mps.is_built()


skip_if_no_mps = pytest.mark.skipif(
    not _mps_available(),
    reason="MPS not available on this host (requires Apple silicon).",
)


@pytest.fixture(scope="module")
def mps_hf_model():
    try:
        base = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2")
        tokenizer = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return base, tokenizer
    except Exception as exc:
        pytest.skip(f"HF download failed; offline?: {exc}")


def _wrap_with_peft_multi_adapter(base_model, n_adapters: int):
    cfg = LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=["c_attn"],
        lora_dropout=0.0,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, cfg, adapter_name="adapter_0")
    for i in range(1, n_adapters):
        model.add_adapter(f"adapter_{i}", cfg)
    return model


@skip_if_no_mps
def test_mps_real_forward_backward_step(mps_hf_model):
    """End-to-end: tiny-gpt2 on torch.device('mps') + peft + plesio_init
    + tokenized forward + backward + occupancy check."""
    base, tokenizer = mps_hf_model
    device = torch.device("mps")
    n_adapters = 4

    model = _wrap_with_peft_multi_adapter(base, n_adapters)
    model = model.to(device)

    cfg = KPoolLoraConfig(
        n_adapters=n_adapters,
        k_active=2,
        routing_strategy="round_robin",
        sideband_enabled=True,
        aggregation_mode="buffer_convergence",
        sideband_addr="tcp://127.0.0.1:0",
        sideband_peers=(),
        buffer_capacity=8,
        buffer_convergence_eps=1e9,
        max_drift_ms=1_000_000,
    )
    plesio_init(model, cfg, sender_id="mps-test-node")
    try:
        torch.manual_seed(0)
        active = apply_kpool_step(model, step=0)
        assert active == (0, 1)

        inputs = tokenizer(
            "The MPS device path",
            return_tensors="pt",
            padding=True,
        )
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,
        )
        out.loss.backward()

        rt = get_runtime(model)
        # Each active adapter pool should have at least one snapshot.
        assert rt.aggregator.buffer.occupancy(0) >= 1, (
            f"adapter 0 buffer empty after backward on MPS; "
            f"occupancy = {rt.aggregator.buffer.occupancy(0)}"
        )
        assert rt.aggregator.buffer.occupancy(1) >= 1
        assert rt.aggregator.buffer.occupancy(2) == 0
        assert rt.aggregator.buffer.occupancy(3) == 0

        # Snapshots are stored on the source device (the gradient's
        # device at push time) rather than copied to CPU. This
        # eliminates the per-push GPU-to-host sync stall that otherwise
        # dominates the wall-clock under FSDP. For an MPS-resident
        # model, the buffer snapshot lives on MPS.
        snap = rt.aggregator.buffer._per_adapter[0][0]
        assert snap.device.type == "mps", (
            f"buffer snapshot should be MPS-resident (source device); "
            f"got {snap.device}"
        )

        decisions = post_backward_step(model, step=0, active=active)
        assert len(decisions) == cfg.k_active
    finally:
        plesio_shutdown(model)


@skip_if_no_mps
def test_mps_multi_step_fire_hold(mps_hf_model):
    """Three sequential real-tokenized training steps on MPS; at least
    one FIRE decision should land once the elastic buffer accumulates
    low-variance snapshots."""
    base, tokenizer = mps_hf_model
    device = torch.device("mps")
    n_adapters = 4

    model = _wrap_with_peft_multi_adapter(base, n_adapters)
    model = model.to(device)

    cfg = KPoolLoraConfig(
        n_adapters=n_adapters,
        k_active=2,
        routing_strategy="round_robin",
        sideband_enabled=True,
        aggregation_mode="buffer_convergence",
        sideband_addr="tcp://127.0.0.1:0",
        sideband_peers=(),
        buffer_capacity=8,
        buffer_convergence_eps=1e9,
        max_drift_ms=1_000_000,
    )
    plesio_init(model, cfg, sender_id="mps-test-node")
    try:
        torch.manual_seed(0)
        for step in range(3):
            active = apply_kpool_step(model, step=step)
            inputs = tokenizer(
                "validation input for the MPS accelerator path",
                return_tensors="pt",
                padding=True,
            )
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
            )
            out.loss.backward()
            post_backward_step(model, step=step, active=active)
            for _, p in model.named_parameters():
                if p.grad is not None:
                    p.grad = None
        rt = get_runtime(model)
        total_fires = sum(rt.aggregator.fire_count.values())
        assert total_fires >= 1, (
            f"expected >= 1 FIRE across 3 MPS steps; "
            f"fire_count={rt.aggregator.fire_count}, "
            f"hold_count={rt.aggregator.hold_count}"
        )
    finally:
        plesio_shutdown(model)
