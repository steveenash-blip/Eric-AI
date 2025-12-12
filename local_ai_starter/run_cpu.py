#!/usr/bin/env python3
"""
Minimal CPU runner using llama-cpp-python and a gguf quantized model.
Place a gguf model at the path you provide (models/model.gguf).
"""

import argparse
from pathlib import Path
from llama_cpp import Llama


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to gguf model")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args()

    if not Path(args.model).exists():
        print("Model not found:", args.model)
        return

    llm = Llama(model_path=args.model)

    resp = llm(prompt=args.prompt, max_tokens=args.max_tokens, temperature=0.2)
    text = resp.get("choices", [{}])[0].get("text", "")
    print("=== OUTPUT ===")
    print(text)


if __name__ == "__main__":
    main()
