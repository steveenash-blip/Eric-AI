#!/usr/bin/env python3
"""
Download a model snapshot from Hugging Face to a local directory using huggingface_hub.snapshot_download.

Usage:
  python download_model.py --repo-id meta-llama/Llama-2-7b-chat-hf --local-dir models/llama-2-7b-chat-hf
"""
import argparse
import os
from huggingface_hub import snapshot_download


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo-id", required=True, help="HF repo id (e.g. meta-llama/Llama-2-7b-chat-hf)")
    p.add_argument("--local-dir", required=True, help="Local folder to save the model")
    p.add_argument("--token", default=os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN"), help="Hugging Face token (or set HUGGINGFACE_HUB_TOKEN)")
    args = p.parse_args()

    print(f"Downloading {args.repo_id} to {args.local_dir} ...")
    path = snapshot_download(repo_id=args.repo_id, local_dir=args.local_dir, token=args.token, repo_type="model")
    print("Downloaded to:", path)
    print("Notes: ensure you accepted model license on Hugging Face web UI before download.")


if __name__ == "__main__":
    main()
