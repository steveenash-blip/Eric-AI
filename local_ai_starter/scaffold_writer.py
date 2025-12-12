#!/usr/bin/env python3
"""
Write a manifest JSON into files on disk.

Manifest format:
{
  "meta": { "install": [...], "run": [...] },
  "files": [
    { "path": "pubspec.yaml", "content": "..." },
    { "path": "lib/main.dart", "content": "..." }
  ]
}
"""

import argparse
import json
from pathlib import Path


def write_manifest(manifest_path, out_dir):
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = manifest.get("files", [])
    for entry in files:
        rel = entry.get("path")
        content = entry.get("content", "")
        if not rel:
            continue
        p = out_dir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        print("Wrote", p)

    # Save meta
    meta = manifest.get("meta", {})
    if meta:
        (out_dir / "META.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        print("Wrote META.json")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--out-dir", default="generated_app")
    args = p.parse_args()
    write_manifest(args.manifest, args.out_dir)


if __name__ == "__main__":
    main()
