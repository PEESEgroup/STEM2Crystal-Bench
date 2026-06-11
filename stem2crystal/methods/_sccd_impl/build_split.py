#!/usr/bin/env python3
"""Reconstruct a train/val/test split for from-scratch training.

The packaged benchmark split is test-only; the training scripts need split['train']/['val'].
This derives them from the rendered STEM images (one structure = one sid), holding out the
existing test set exactly, then writing <stem-root>/split_repro.json (train/val/test).

  python build_split.py --stem-root /path/to/all_stem_output --val-frac 0.05 --seed 42
"""
import argparse, json, os, random
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem-root", required=True, help="dir with images/<noise>/<sid>_<view>.png + split.json")
    ap.add_argument("--noise", default="low")
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="split_repro.json")
    a = ap.parse_args()
    root = Path(a.stem_root)
    all_ids = sorted({f.rsplit("_", 1)[0] for f in os.listdir(root / "images" / a.noise) if f.endswith(".png")})
    test = set(json.load(open(root / "split.json")).get("test", []))
    missing = test - set(all_ids)
    if missing:
        print(f"[warn] {len(missing)} test ids have no image (kept in test anyway)")
    pool = [s for s in all_ids if s not in test]
    random.Random(a.seed).shuffle(pool)
    nval = max(1, int(a.val_frac * len(pool)))
    split = {"train": pool[nval:], "val": pool[:nval], "test": sorted(test)}
    json.dump(split, open(root / a.out, "w"))
    print(f"wrote {root / a.out}: train={len(split['train'])} val={len(split['val'])} test={len(split['test'])}")

if __name__ == "__main__":
    main()
