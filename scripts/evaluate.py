#!/usr/bin/env python3
"""Evaluate a method's predictions against STEM2Crystal-Bench (all paper metrics).

Two ways to point at predictions:
  --pred <dir>                 score an explicit predictions dir (<dir>/<sid>_0/rank_*.cif)
  --method <name> [--noise X]  score ./predictions/<name>[/<noise>]  (the layout scripts/generate.py writes)

Examples:
  python scripts/evaluate.py --method sccd --benchmark synthetic --noise low
  python scripts/evaluate.py --pred /path/to/preds --benchmark real5 --name my_model
"""
import argparse
import json
import os
from pathlib import Path
from stem2crystal.data import benchmark_paths, default_data_root
from stem2crystal.eval import evaluate_predictions, format_table, BENCHMARK_MATCHERS


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", choices=list(BENCHMARK_MATCHERS), default="synthetic")
    ap.add_argument("--method", default=None, help="predictions under ./predictions/<method>")
    ap.add_argument("--pred", default=None, help="explicit predictions dir")
    ap.add_argument("--noise", default=None, choices=["low", "mid", "high"], help="synthetic noise level")
    ap.add_argument("--name", default=None)
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--n-jobs", type=int, default=None)
    ap.add_argument("--config", default="configs/eval.yaml", help="eval config (matcher tolerances, k, n_jobs)")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--pred-root", default="predictions")
    ap.add_argument("--json", default="", help="optional path to dump metrics as JSON")
    a = ap.parse_args()

    # eval config: matcher tolerances + k/symprec/n_jobs (CLI flags override the config)
    cfg = {}
    if a.config and os.path.exists(a.config):
        import yaml
        cfg = yaml.safe_load(open(a.config)) or {}
    mkw = dict(BENCHMARK_MATCHERS[a.benchmark])
    mkw.update((cfg.get("benchmarks", {}) or {}).get(a.benchmark, {}) or {})
    k = a.k if a.k is not None else cfg.get("k", 5)
    n_jobs = a.n_jobs if a.n_jobs is not None else cfg.get("n_jobs")
    symprec = cfg.get("symprec", 0.1)

    paths = benchmark_paths(a.benchmark, a.data_root or default_data_root())
    ids = json.load(open(paths["split"]))
    ids = ids["test"] if isinstance(ids, dict) else ids

    if a.pred:
        pred_dir = Path(a.pred)
    elif a.method:
        pred_dir = Path(a.pred_root) / a.method
        if a.benchmark == "synthetic":
            pred_dir = pred_dir / (a.noise or "low")
    else:
        ap.error("give either --pred or --method")

    name = a.name or a.method or "method"
    metrics = evaluate_predictions(paths["gt_dir"], pred_dir, ids, k=k, symprec=symprec, n_jobs=n_jobs, **mkw)
    print(format_table({name: metrics}))
    print("failure modes:", metrics["failure_modes"])
    if a.json:
        json.dump(metrics, open(a.json, "w"), indent=2)
        print("wrote", a.json)


if __name__ == "__main__":
    main()
