"""Runnable end-to-end tsugi-kpool demo on CPU.

Exercises the full K-of-N + buffer-convergence loop on `sshleifer/tiny-gpt2`
(~tens of thousands of params) so the mechanism is reproducible on a clean
install without a GPU or any gated-model access. The first run downloads the
tiny model (~5 MB) from the Hugging Face Hub.

    pip install "tsugi-kpool[dev]"   # transformers + peft come with the extras
    python examples/minimal_finetune.py

Expected output: per-step lines showing the K active adapters and their
HOLD / FIRE decisions, then a FIRE/HOLD tally.
"""
from __future__ import annotations


def main() -> None:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from tsugi_kpool import (
        KPoolLoraConfig,
        apply_kpool_step,
        get_runtime,
        plesio_init,
        plesio_shutdown,
        post_backward_step,
    )

    model_name = "sshleifer/tiny-gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(model_name)

    # K-Pool routes over a POOL of N named adapters. Build them explicitly:
    # one LoraConfig applied as adapter_0 .. adapter_{N-1}. (GPT-2 attention
    # uses the combined `c_attn` projection rather than q_proj/v_proj.)
    n_adapters = 4
    lora = LoraConfig(r=4, lora_alpha=8, target_modules=["c_attn"], task_type="CAUSAL_LM")
    model = get_peft_model(base, lora, adapter_name="adapter_0")
    for i in range(1, n_adapters):
        model.add_adapter(f"adapter_{i}", lora)

    config = KPoolLoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=("c_attn",),
        n_adapters=n_adapters,
        k_active=2,                       # K-out-of-N adapters per step
        routing_strategy="round_robin",
        sideband_enabled=True,            # turn the plesiochronous path on
        aggregation_mode="buffer_convergence",
        sideband_addr="tcp://127.0.0.1:0",  # ephemeral loopback port (single-node demo)
        buffer_capacity=8,
        buffer_convergence_eps=1e9,       # generous eps so FIRE is reachable in a short demo
        max_drift_ms=1_000_000,
    )

    plesio_init(model, config, sender_id="demo-node")
    try:
        torch.manual_seed(0)
        for step in range(5):
            active = apply_kpool_step(model, step=step)   # selects K adapters
            inputs = tokenizer(
                "the quick brown fox jumps over the lazy dog",
                return_tensors="pt",
                padding=True,
            )
            out = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=inputs["input_ids"],
            )
            out.loss.backward()
            decisions = post_backward_step(model, step=step, active=active)
            fired = [d.adapter_idx for d in decisions if d.fired]
            held = [d.adapter_idx for d in decisions if not d.fired]
            print(
                f"step {step}: active={active} FIRE={fired} HOLD={held} "
                f"loss={out.loss.item():.4f}"
            )
            # clear grads between steps (this demo does not run an optimizer)
            for _, p in model.named_parameters():
                if p.grad is not None:
                    p.grad = None

        rt = get_runtime(model)
        total_fire = sum(rt.aggregator.fire_count.values())
        total_hold = sum(rt.aggregator.hold_count.values())
        print(f"totals: FIRE={total_fire} HOLD={total_hold}")
    finally:
        plesio_shutdown(model)


if __name__ == "__main__":
    main()
