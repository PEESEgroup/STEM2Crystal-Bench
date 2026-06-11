#!/usr/bin/env python3
"""Drive a registered Method over a benchmark and write candidates as predictions/<method>/.../rank_*.cif.

  python scripts/generate.py --method example_random --benchmark synthetic --noise low --limit 50
  python scripts/generate.py --method sccd --benchmark real5

Add your own model by subclassing stem2crystal.methods.Method and registering it (see examples/).
Point --method-module at the file that defines+registers it if it is outside the package.
"""
import argparse
import importlib
import importlib.util
import json
import os
from pathlib import Path
import numpy as np
from PIL import Image
from pymatgen.core import Structure, Composition

from stem2crystal.data import benchmark_paths, default_data_root
from stem2crystal.methods import get_method, list_methods


def _load_module(path):
    spec = importlib.util.spec_from_file_location("user_method", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", required=True)
    ap.add_argument("--method-module", default=None, help="python file that registers a custom method")
    ap.add_argument("--benchmark", choices=["synthetic", "real5"], default="synthetic")
    ap.add_argument("--noise", default="low", choices=["low", "mid", "high"])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--out-root", default="predictions")
    ap.add_argument("--config", default=None,
                    help="method config (YAML/JSON). Default: configs/methods/<method>.yaml if present.")
    a = ap.parse_args()

    if a.method_module:
        _load_module(a.method_module)
    if a.method not in list_methods():
        raise SystemExit(f"unknown method '{a.method}'. Registered: {list_methods()} "
                         f"(use --method-module to load a custom one).")

    cfg_path = a.config or f"configs/methods/{a.method}.yaml"
    cfg = {}
    if os.path.exists(cfg_path):
        if cfg_path.endswith((".yaml", ".yml")):
            import yaml
            cfg = yaml.safe_load(open(cfg_path)) or {}
        else:
            cfg = json.load(open(cfg_path))
        print(f"[config] {cfg_path}")
    method = get_method(a.method, cfg)
    method.setup()

    paths = benchmark_paths(a.benchmark, a.data_root or default_data_root())
    ids = json.load(open(paths["split"]))
    ids = ids["test"] if isinstance(ids, dict) else ids
    if a.limit:
        ids = ids[:a.limit]
    img_dir = paths["images"]["real" if a.benchmark == "real5" else a.noise]
    sub = a.noise if a.benchmark == "synthetic" else "real"
    out_dir = Path(a.out_root) / a.method / sub

    for i, sid in enumerate(ids):
        gt = Structure.from_file(str(paths["gt_dir"] / f"{sid}.cif"))
        comp = gt.composition
        img_path = next((img_dir / f"{sid}_{v}.png" for v in (0, "")
                         if (img_dir / f"{sid}_{v}.png").exists()), img_dir / f"{sid}_0.png")
        image = np.asarray(Image.open(img_path).convert("L"), np.float32) / 255.0
        cands = method.generate(image, comp, k=a.k)
        method.write_candidates(out_dir, sid, cands)
        if (i + 1) % 50 == 0:
            print(f"[{a.method}/{sub}] {i + 1}/{len(ids)}", flush=True)
    print(f"done -> {out_dir}")


if __name__ == "__main__":
    main()
