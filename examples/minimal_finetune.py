"""Smallest possible working example of the SDK API surface.

This is the snippet shown in the README. It is intentionally not a real
fine-tune run; it just demonstrates the user-facing shape so a an enterprise customer
engineer can see the surface area in 30 seconds.
"""
from __future__ import annotations


def main() -> None:
    from transformers import AutoModelForCausalLM
    from peft import get_peft_model, LoraConfig

    from tsugi_kpool import KPoolLoraConfig, plesio_init, plesio_shutdown

    # 1. Load the base model with stock transformers
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")

    # 2. Build the K-Pool LoRA config. Same shape as peft.LoraConfig plus
    #    K-Pool routing and Infinity sideband knobs.
    cfg = KPoolLoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=("q_proj", "v_proj"),
        n_adapters=8,
        k_active=2,
        sideband_enabled=False,           # set True + supply peers for real distributed run
        aggregation_mode="synchronous",   # set "buffer_convergence" for Infinity mode
    )

    # 3. Wrap with peft. The SDK keeps the standard peft surface.
    model = get_peft_model(model, LoraConfig(
        r=cfg.r,
        lora_alpha=cfg.lora_alpha,
        target_modules=list(cfg.target_modules),
        bias=cfg.bias,
    ))

    # 4. Wire up the K-Pool router + (optional) Infinity sideband + aggregator
    plesio_init(model, cfg)

    try:
        # 5. From here, train with your normal trainer / accelerate loop.
        # (omitted in this minimal example)
        pass
    finally:
        plesio_shutdown(model)


if __name__ == "__main__":
    main()
