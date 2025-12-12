#!/usr/bin/env python3
"""
GPU runner for meta-llama/Llama-2-7b-chat-hf using transformers + bitsandbytes 4-bit.

Usage:
  python run_gpu_llama2.py --model-dir models/llama-2-7b-chat-hf --prompt "Create a minimal Flutter todo app" --out manifest.json

Notes:
- Either pass --model-dir for a local snapshot, or use --model-id to let transformers download from HF cache.
- This script expects models that support the standard tokenizer and causal LM interface.
"""
import argparse
import json
import os
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


def load_tokenizer_and_model(model_dir_or_id: str):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir_or_id, use_fast=True)

    print("Loading model (this can take a while)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_dir_or_id,
        device_map="auto",
        quantization_config=bnb_config,
        trust_remote_code=True,
    )
    return tokenizer, model


def generate_text(tokenizer, model, prompt: str, max_new_tokens=512, temperature=0.2):
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=False,
        )
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    return text


def extract_json_manifest(text: str):
    # Try direct parse
    try:
        return json.loads(text)
    except Exception:
        # Try to find first JSON object
        m = re.search(r'(\{[\s\S]*\})', text)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception as e:
                print("Found JSON-like block but failed to parse:", e)
        print("Raw output (first 2000 chars):\n", text[:2000])
        raise RuntimeError("Couldn't parse manifest JSON from model output")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", help="Local model dir (preferred)")
    p.add_argument("--model-id", help="HF model id (if not using a local dir)")
    p.add_argument("--prompt", required=True)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--out", default="manifest.json")
    p.add_argument("--temperature", type=float, default=0.2)
    args = p.parse_args()

    model_ref = args.model_dir if args.model_dir else args.model_id
    if not model_ref:
        raise SystemExit("Please provide --model-dir or --model-id")

    tokenizer, model = load_tokenizer_and_model(model_ref)
    print("Generating...")
    text = generate_text(tokenizer, model, args.prompt, max_new_tokens=args.max_new_tokens, temperature=args.temperature)

    manifest = extract_json_manifest(text)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print("Wrote manifest to", args.out)


if __name__ == "__main__":
    main()
