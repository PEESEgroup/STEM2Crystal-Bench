#!/usr/bin/env python3
"""Download STEM2Crystal-Bench from the Hugging Face Hub into ./data (git-ignored)."""
import argparse
from stem2crystal.data import download_benchmark, DATASET_REPO

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=f"Download {DATASET_REPO} into a local data dir.")
    ap.add_argument("--config", choices=["all", "synthetic", "real5"], default="all")
    ap.add_argument("--data-root", default=None,
                    help="target dir (default ./data or $STEM2CRYSTAL_DATA). The synthetic images are "
                         "several GB; point this at a roomy filesystem, not a small root partition.")
    a = ap.parse_args()
    root = download_benchmark(a.config, a.data_root)
    print(f"downloaded '{a.config}' -> {root}")
