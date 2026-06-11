"""Dataset-level evaluation: turn a directory of per-sample candidate CIFs into the full paper
metric table. Model-agnostic — it only needs (1) ground-truth CIFs and (2) a predictions directory
laid out as ``<pred_dir>/<sid>_0/rank_{1..K}.cif`` (the convention every method in this repo writes).

Usage (library):
    from stem2crystal.eval import evaluate_predictions, BENCHMARK_MATCHERS
    table = evaluate_predictions(gt_dir, pred_dir, split_ids, **BENCHMARK_MATCHERS["synthetic"])

Usage (CLI): ``python -m stem2crystal.eval.evaluate --gt ... --pred ... --split ... --benchmark synthetic``
"""
from __future__ import annotations
import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence
import numpy as np
from pymatgen.core import Structure

from .metrics import (scale_matcher, abs_matcher, score_sample, wasserstein_lattice)

# Per-benchmark matcher settings (paper Section 5.2). These reproduce the paper tables exactly.
# (pymatgen's StructureMatcher defaults primitive_cell=True, which the paper used.)
BENCHMARK_MATCHERS = {
    # synthetic STEM2Crystal-Bench (Table 1)
    "synthetic": dict(stol=0.5, ltol=0.3, angle_tol=10, primitive_cell=True, attempt_supercell=False),
    # Real-5 v2 monolayers (Table 2): looser stol/angle + supercell matching for monolayer cells
    "real5":     dict(stol=0.3, ltol=0.3, angle_tol=15, primitive_cell=True, attempt_supercell=True),
}


def load_structure(path: Path) -> Optional[Structure]:
    try:
        return Structure.from_file(str(path))
    except Exception:
        return None


def load_candidates(pred_dir: Path, sid: str, k: int) -> List[Structure]:
    """Load ranked candidate CIFs ``<pred_dir>/<sid>_0/rank_{1..k}.cif``."""
    d = pred_dir / f"{sid}_0"
    if not d.exists():
        return []
    out = []
    for i in range(1, k + 1):
        s = load_structure(d / f"rank_{i}.cif")
        if s is not None:
            out.append(s)
    return out


# --- worker plumbing (module-level so it is picklable for multiprocessing) --------------
_W: Dict = {}

def _avail_bytes() -> int:
    """Best-effort available physical memory (bytes); falls back to 8 GiB."""
    try:
        return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except Exception:
        return 8 * 2**30

def _cap_address_space(cap_bytes: int) -> None:
    """Cap this process's virtual address space so a pathological StructureMatcher call raises
    MemoryError (caught by the per-match guard -> treated as no-match) instead of exhausting the
    machine. No-op on platforms without RLIMIT_AS (e.g. macOS)."""
    if not cap_bytes:
        return
    try:
        import resource
        _, hard = resource.getrlimit(resource.RLIMIT_AS)
        new = cap_bytes if hard == resource.RLIM_INFINITY else min(cap_bytes, hard)
        resource.setrlimit(resource.RLIMIT_AS, (new, hard))
    except Exception:
        pass

def _init_worker(gt_dir, pred_dir, k, symprec, mkw, mem_cap=0):
    _cap_address_space(mem_cap)
    _W.update(gt_dir=Path(gt_dir), pred_dir=Path(pred_dir), k=k, symprec=symprec,
              matcher=scale_matcher(**mkw), matcher_abs=abs_matcher(**mkw))

def _score_sid(sid):
    gt = load_structure(_W["gt_dir"] / f"{sid}.cif")
    if gt is None:
        return None
    cands = load_candidates(_W["pred_dir"], sid, _W["k"])
    r = score_sample(gt, cands, _W["k"], _W["matcher"], _W["matcher_abs"], _W["symprec"])
    return (r, (r.top1_b, r.top1_gamma, gt.lattice.b, gt.lattice.gamma) if r.parsed else None)


def evaluate_predictions(gt_dir: str | Path, pred_dir: str | Path, split_ids: Sequence[str],
                         k: int = 5, symprec: float = 0.1, n_jobs: Optional[int] = None,
                         **matcher_kwargs) -> Dict:
    """Score one method's predictions against ground truth. Returns a dict with every paper metric.

    ``matcher_kwargs`` (stol/ltol/angle_tol/primitive_cell/attempt_supercell) set the StructureMatcher;
    pass ``**BENCHMARK_MATCHERS[name]`` to reproduce a paper benchmark exactly.
    ``n_jobs`` parallelizes over samples (default: ``min(8, cpu_count)``; set 1 to force serial).
    Workers run with single-threaded BLAS and a per-worker memory cap, so scoring wrong-scale
    predictions (whose StructureMatcher calls can balloon) stays bounded instead of OOM-ing."""
    mkw = matcher_kwargs
    n = len(split_ids)
    if n_jobs is None:
        n_jobs = min(8, os.cpu_count() or 1)

    if n_jobs > 1:
        from multiprocessing import Pool
        # cap each worker at a fair share of available RAM (2-24 GiB), so a single pathological
        # match fails gracefully rather than exhausting the box when many workers hit one at once.
        share = 0.7 * _avail_bytes() / n_jobs
        mem_cap = int(min(24 * 2**30, max(2 * 2**30, share)))
        with Pool(n_jobs, initializer=_init_worker,
                  initargs=(gt_dir, pred_dir, k, symprec, mkw, mem_cap), maxtasksperchild=50) as pool:
            raw = [x for x in pool.map(_score_sid, list(split_ids), chunksize=4) if x is not None]
    else:
        _init_worker(gt_dir, pred_dir, k, symprec, mkw)  # serial: no AS cap on the caller's process
        raw = [x for x in (_score_sid(s) for s in split_ids) if x is not None]

    results = [r for r, _ in raw]
    parsed = [r for r in results if r.parsed]
    pb, pg, gb, gg = [], [], [], []
    for r, lat in raw:
        if lat is not None:
            pb.append(lat[0]); pg.append(lat[1]); gb.append(lat[2]); gg.append(lat[3])

    rms5 = [r.min_rms5 for r in parsed if r.min_rms5 is not None]
    rmsA = [r.rmsd_A for r in parsed if r.rmsd_A is not None]
    fails = Counter(r.failure for r in parsed if r.failure != "hit")
    frac = lambda key: round(fails.get(key, 0) / n, 4) if n else None

    return dict(
        N=n, parsable=len(parsed),
        Hit1=round(sum(r.hit1 for r in parsed) / n, 4) if n else None,
        Hit5=round(sum(r.hit5 for r in parsed) / n, 4) if n else None,
        MinRMS_median=round(float(np.median(rms5)), 4) if rms5 else None,
        MinRMS_mean=round(float(np.mean(rms5)), 4) if rms5 else None,
        Match_abs=round(sum(r.match_abs for r in parsed) / n, 4) if n else None,
        RMSD_A=round(float(np.median(rmsA)), 4) if rmsA else None,
        SG_Acc=round(sum(r.sg_acc for r in parsed) / n, 4) if n else None,
        W1_b=round(wasserstein_lattice(pb, gb), 4) if pb else None,
        W1_gamma=round(wasserstein_lattice(pg, gg), 4) if pg else None,
        failure_modes={key: frac(key) for key in
                       ["formula", "wrong_lattice", "lattice_ok_wrong_SG", "coord_other"]},
    )


_COLS = ["Hit1", "Hit5", "MinRMS_median", "MinRMS_mean", "Match_abs", "RMSD_A", "SG_Acc", "W1_b", "W1_gamma"]

def format_table(rows: Dict[str, Dict]) -> str:
    """rows = {method_name: metric_dict}. Returns an aligned text table."""
    cell = lambda v: " -- " if v is None else f"{v:.4f}"
    head = f"{'Method':16s} " + " ".join(f"{c:>12s}" for c in _COLS)
    lines = [head, "-" * len(head)]
    for name, m in rows.items():
        lines.append(f"{name:16s} " + " ".join(f"{cell(m.get(c)):>12s}" for c in _COLS))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Evaluate a predictions directory against STEM2Crystal-Bench ground truth.")
    ap.add_argument("--gt", required=True, help="ground-truth CIF dir (<gt>/<sid>.cif)")
    ap.add_argument("--pred", required=True, help="predictions dir (<pred>/<sid>_0/rank_*.cif)")
    ap.add_argument("--split", required=True, help="JSON with {'test': [sid, ...]} (or a flat list)")
    ap.add_argument("--benchmark", choices=list(BENCHMARK_MATCHERS), default="synthetic")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--n-jobs", type=int, default=None)
    ap.add_argument("--name", default="method")
    ap.add_argument("--json", default="")
    a = ap.parse_args()

    split = json.load(open(a.split))
    ids = split["test"] if isinstance(split, dict) else split
    metrics = evaluate_predictions(a.gt, a.pred, ids, k=a.k, n_jobs=a.n_jobs, **BENCHMARK_MATCHERS[a.benchmark])
    print(format_table({a.name: metrics}))
    print("failure modes:", metrics["failure_modes"])
    if a.json:
        json.dump(metrics, open(a.json, "w"), indent=2)
        print("wrote", a.json)


if __name__ == "__main__":
    main()
