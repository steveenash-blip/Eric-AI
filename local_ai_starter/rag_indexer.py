#!/usr/bin/env python3
"""
Simple FAISS indexer using sentence-transformers.
Index plain-text files under --docs-dir into a FAISS index and save metadata.

Usage:
  python rag_indexer.py --docs-dir templates_and_docs --index-dir data/index
"""

import argparse
import os
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from tqdm import tqdm

EMBED_MODEL = "all-MiniLM-L6-v2"

def read_documents(docs_dir):
    docs = []
    for p in Path(docs_dir).rglob("*"):
        if p.is_file() and p.suffix.lower() in {".md", ".txt", ".dart", ".js", ".json", ".yaml", ".yml"}:
            text = p.read_text(encoding="utf-8", errors="ignore")
            docs.append({"path": str(p.relative_to(docs_dir)), "text": text})
    return docs


def build_index(docs, index_dir):
    model = SentenceTransformer(EMBED_MODEL)
    texts = [d["text"] for d in docs]
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    os.makedirs(index_dir, exist_ok=True)
    faiss.write_index(index, os.path.join(index_dir, "index.faiss"))
    with open(os.path.join(index_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)
    print("Saved index and meta to", index_dir)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--docs-dir", required=True)
    p.add_argument("--index-dir", required=True)
    args = p.parse_args()

    docs = read_documents(args.docs_dir)
    if not docs:
        print("No documents found in", args.docs_dir)
        return
    build_index(docs, args.index_dir)


if __name__ == "__main__":
    main()
