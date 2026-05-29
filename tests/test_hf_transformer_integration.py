"""Real HuggingFace transformer integration tests.

The real-peft tests validate the SDK against peft on a hand-rolled
`_TinyAttn` nn.Module. These tests validate against an actual HF
transformer (`sshleifer/tiny-gpt2`, ~100K params, GPT-2 architecture),
closing the gap between the custom-Linear-stack tests and a real
transformer attention layer (Qwen2 / Llama-3 class).

Requires: torch + peft + transformers (the `dev` extras). First run will
download tiny-gpt2 (~5MB) from HuggingFace Hub.
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


@pytest.fixture(scope="module")
def hf_model_and_tokenizer():
    """Cache the tiny-gpt2 model + tokenizer at module scope so multiple
    tests share the single download."""
    try:
        model = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2")
        tokenizer = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return model, tokenizer
    except Exception as exc:
        pytest.skip(f"HF download failed; offline?: {exc}")


def _wrap_with_peft_multi_adapter(base_model, n_adapters: int):
    """Wrap a HF model with peft + add N independent LoRA adapters
    named adapter_0..adapter_{N-1}. GPT-2 attention uses combined
    `c_attn` projection (q, k, v stacked) rather than the
    q_proj/k_proj/v_proj convention used by Llama / Qwen2."""
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


def test_hf_tiny_gpt2_plesio_init_discovers_all_adapters(hf_model_and_tokenizer):
    """plesio_init on a peft-wrapped tiny-gpt2 discovers all 4 adapter
    pools. tiny-gpt2 has 2 transformer layers, each with one c_attn
    projection. Each adapter therefore has 2 layers x 2 lora matrices
    (lora_A + lora_B) = 4 LoRA params, multiplied by 1 c_attn module =
    4 params total per adapter."""
    base, _ = hf_model_and_tokenizer
    n_adapters = 4
    model = _wrap_with_peft_multi_adapter(base, n_adapters)

    cfg = KPoolLoraConfig(
        n_adapters=n_adapters,
        k_active=2,
        routing_strategy="round_robin",
        sideband_enabled=True,
        aggregation_mode="buffer_convergence",
        sideband_addr="tcp://127.0.0.1:0",
        sideband_peers=(),
        buffer_capacity=4,
        buffer_convergence_eps=1e9,
        max_drift_ms=1_000_000,
    )
    plesio_init(model, cfg, sender_id="hf-test-node")
    try:
        rt = get_runtime(model)
        # tiny-gpt2 has 2 transformer layers; target_modules=["c_attn"]
        # means 2 LoRA-wrapped Linear instances per adapter; each has
        # lora_A + lora_B = 4 LoRA params per adapter.
        for i in range(n_adapters):
            assert len(rt.adapter_params[i]) == 4, (
                f"adapter {i} should have 4 params (2 c_attn x 2 lora "
                f"matrices); got {len(rt.adapter_params[i])}"
            )
    finally:
        plesio_shutdown(model)


def test_hf_tiny_gpt2_real_forward_backward_step(hf_model_and_tokenizer):
    """End-to-end: tokenize a real input, run forward through tiny-gpt2
    with peft + K-Pool LoRA active, run backward, confirm backward hooks
    fire and elastic buffer accumulates snapshots for the K active
    adapters."""
    base, tokenizer = hf_model_and_tokenizer
    n_adapters = 4
    model = _wrap_with_peft_multi_adapter(base, n_adapters)

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
    plesio_init(model, cfg, sender_id="hf-test-node")
    try:
        torch.manual_seed(0)
        active = apply_kpool_step(model, step=0)
        assert active == (0, 1)

        # Real input: a short prompt tokenized to fixed length.
        inputs = tokenizer(
            "The patent claims a system",
            return_tensors="pt",
            padding=True,
        )
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,
        )
        out.loss.backward()

        rt = get_runtime(model)
        # Active adapters 0 and 1 should each have snapshots; inactive
        # adapters 2 and 3 should not.
        assert rt.aggregator.buffer.occupancy(0) >= 1, (
            f"adapter 0 buffer empty after backward; "
            f"occupancy = {rt.aggregator.buffer.occupancy(0)}"
        )
        assert rt.aggregator.buffer.occupancy(1) >= 1
        assert rt.aggregator.buffer.occupancy(2) == 0
        assert rt.aggregator.buffer.occupancy(3) == 0

        # post_backward_step yields per-active-adapter decisions
        decisions = post_backward_step(model, step=0, active=active)
        assert len(decisions) == cfg.k_active
    finally:
        plesio_shutdown(model)


def test_hf_tiny_gpt2_multi_step_fire_hold(hf_model_and_tokenizer):
    """Three sequential real-tokenized training steps with low buffer_
    convergence_eps; at least one FIRE decision should land once the
    elastic buffer accumulates a few low-variance snapshots."""
    base, tokenizer = hf_model_and_tokenizer
    n_adapters = 4
    model = _wrap_with_peft_multi_adapter(base, n_adapters)

    cfg = KPoolLoraConfig(
        n_adapters=n_adapters,
        k_active=2,
        routing_strategy="round_robin",
        sideband_enabled=True,
        aggregation_mode="buffer_convergence",
        sideband_addr="tcp://127.0.0.1:0",
        sideband_peers=(),
        buffer_capacity=8,
        buffer_convergence_eps=1e9,  # very generous; FIRE-eligible once buffer fills
        max_drift_ms=1_000_000,
    )
    plesio_init(model, cfg, sender_id="hf-test-node")
    try:
        torch.manual_seed(0)
        for step in range(3):
            active = apply_kpool_step(model, step=step)
            inputs = tokenizer(
                "the quick brown fox jumps over the lazy dog",
                return_tensors="pt",
                padding=True,
            )
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
            )
            out.loss.backward()
            post_backward_step(model, step=step, active=active)
            # Clear .grad between steps (we are not running optimizer.step)
            for _, p in model.named_parameters():
                if p.grad is not None:
                    p.grad = None
        rt = get_runtime(model)
        total_fires = sum(rt.aggregator.fire_count.values())
        assert total_fires >= 1, (
            f"expected >= 1 FIRE across 3 steps; "
            f"fire_count={rt.aggregator.fire_count}, "
            f"hold_count={rt.aggregator.hold_count}"
        )
    finally:
        plesio_shutdown(model)
