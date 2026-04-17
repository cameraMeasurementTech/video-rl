#!/usr/bin/env python3
"""
Merge PEFT LoRA adapters into a full Hugging Face model for GRPO (full fine-tune).

VERL LoRA SFT saves checkpoints under ``trainer.default_local_dir/global_step_*``.
If the saved folder is a **PeftModel** (adapter_config.json present), merge into base weights.

Usage::

  python -m train.tools.merge_peft_adapter \\
    --base-model Qwen/Qwen2.5-7B-Instruct \\
    --adapter-path train/artifacts/sft_lora/run/global_step_500 \\
    --out train/artifacts/merged_for_grpo

If ``--adapter-path`` is already a **full** model (no adapter), copies to ``--out`` with a message.

Requires: transformers, peft, torch, accelerate (for device_map).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", required=True, help="Original base HF model id or local path used for SFT")
    ap.add_argument("--adapter-path", required=True, help="SFT checkpoint dir (global_step_*)")
    ap.add_argument("--out", required=True, help="Output directory for merged full model")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    args = ap.parse_args()

    adapter_path = Path(args.adapter_path).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]

    adapter_config = adapter_path / "adapter_config.json"
    if not adapter_config.is_file():
        print(
            f"No adapter_config.json in {adapter_path}.\n"
            "If this directory is already a **full** Hugging Face model, set:\n"
            f"  export MODEL_PATH={adapter_path}\n"
            "and skip merging. Otherwise point --adapter-path to a LoRA checkpoint."
        )
        return

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    base = args.base_model
    print(f"Loading base: {base}")
    model = AutoModelForCausalLM.from_pretrained(
        base,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map="cpu",
    )
    print(f"Loading adapter: {adapter_path}")
    model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=False)
    print("Merging and unloading adapters...")
    merged = model.merge_and_unload()
    merged.save_pretrained(str(out), safe_serialization=True)
    tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    tok.save_pretrained(str(out))
    print(f"Merged model saved to {out}")
    print(f"Use: export MODEL_PATH={out}")


if __name__ == "__main__":
    main()
