#!/usr/bin/env python3
"""
Simple GPU-based runner using Transformers + bitsandbytes (4-bit).
Adapt the model_id to a 7B model you've downloaded or that is available on HF.
"""

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


def load_model(model_id: str):
    # 4-bit config for bitsandbytes
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

    print("Loading model (this may take a while)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    return tokenizer, model


def generate(tokenizer, model, prompt: str, max_new_tokens=512, temperature=0.2):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=False,
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True, help="HF model id or local path")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()

    tokenizer, model = load_model(args.model_id)
    print("Generating...")
    text = generate(tokenizer, model, args.prompt, max_new_tokens=args.max_new_tokens)
    print("=== OUTPUT ===")
    print(text)


if __name__ == "__main__":
    main()
