"""All STEM2Crystal-Bench evaluation metrics (paper Section 5.2), as pure functions over
pymatgen ``Structure`` objects. This module is **fully decoupled from any model**: give it a
ground-truth structure and a ranked list of candidate structures, and it returns every metric
reported in the paper.

Metrics implemented (paper Table 1 / Table 2 / ablation / failure modes):
    Hit@1, Hit@5 ............ top-1 / best-of-K reconstruction success (scale-normalized matcher)
    MinRMS@K (median/mean) .. min scale-normalized RMS over formula-matched candidates
    Match_abs ............... scale-sensitive match rate (matcher with rescaling disabled)
    RMSD (Angstrom) ......... absolute structural RMS over formula-matched candidates
    SG-Acc .................. top-1 space-group accuracy (spglib 3D international number)
    W1(b), W1(gamma) ........ 1D Wasserstein-1 distance of in-plane lattice marginals
    failure_mode ............ formula / wrong_lattice / lattice_ok_wrong_SG / coord_other

Matcher conventions (paper Section 5.2):
    * scale-normalized: ``StructureMatcher(stol=0.5, ltol=0.3, angle_tol=10, scale=True)``
    * absolute:         same but ``scale=False`` (exposes absolute-cell errors).
The matcher is passed in so callers can reproduce the exact per-benchmark setting
(the synthetic benchmark uses the bare matcher; Real-5 adds ``primitive_cell``/``attempt_supercell``).
"""
from __future__ import annotations
import signal
from dataclasses import dataclass, field
from typing import Optional, Sequence
import numpy as np
from pymatgen.core import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

try:                                   # optional W1 dependency
    from scipy.stats import wasserstein_distance
except Exception:                      # pragma: no cover
    wasserstein_distance = None

__all__ = [
    "scale_matcher", "abs_matcher", "formula", "space_group",
    "hit_at_k", "min_rms_at_k", "match_abs", "rmsd_angstrom", "sg_accuracy",
    "wasserstein_lattice", "failure_mode", "SampleResult", "score_sample",
]

# --------------------------------------------------------------------------------------
# Matcher factories (paper defaults)
# --------------------------------------------------------------------------------------
def scale_matcher(stol=0.5, ltol=0.3, angle_tol=10, primitive_cell=False,
                  attempt_supercell=False) -> StructureMatcher:
    """Scale-normalized matcher (used for Hit@k, MinRMS@K, SG-Acc)."""
    return StructureMatcher(stol=stol, ltol=ltol, angle_tol=angle_tol, scale=True,
                            primitive_cell=primitive_cell, attempt_supercell=attempt_supercell)

def abs_matcher(stol=0.5, ltol=0.3, angle_tol=10, primitive_cell=False,
                attempt_supercell=False) -> StructureMatcher:
    """Absolute (scale-sensitive) matcher (used for Match_abs, RMSD-Angstrom)."""
    return StructureMatcher(stol=stol, ltol=ltol, angle_tol=angle_tol, scale=False,
                            primitive_cell=primitive_cell, attempt_supercell=attempt_supercell)


# --------------------------------------------------------------------------------------
# Small helpers (with a wall-clock guard, since StructureMatcher can hang on bad cells)
# --------------------------------------------------------------------------------------
class _Timeout(Exception):
    pass

def _alarm(_s, _f):  # pragma: no cover
    raise _Timeout()

signal.signal(signal.SIGALRM, _alarm)

def _guard(fn, seconds: float = 1.5):
    """Run ``fn`` but abort (return None) if it exceeds ``seconds`` (main thread only)."""
    try:
        signal.setitimer(signal.ITIMER_REAL, seconds)
        out = fn()
        signal.setitimer(signal.ITIMER_REAL, 0)
        return out
    except Exception:
        signal.setitimer(signal.ITIMER_REAL, 0)
        return None

def formula(s: Structure) -> Optional[str]:
    try:
        return s.composition.reduced_formula
    except Exception:
        return None

def space_group(s: Structure, symprec: float = 0.1) -> Optional[int]:
    """3D international space-group number (spglib via pymatgen). Not the 2D layer group."""
    try:
        return SpacegroupAnalyzer(s, symprec=symprec).get_space_group_number()
    except Exception:
        return None

def _lattice_relerr(p: Structure, g: Structure) -> float:
    try:
        return max(abs(p.lattice.a - g.lattice.a) / g.lattice.a,
                   abs(p.lattice.b - g.lattice.b) / g.lattice.b,
                   abs(p.lattice.gamma - g.lattice.gamma) / max(g.lattice.gamma, 1e-6))
    except Exception:
        return 9.9

def _formula_matched(gt, cands):
    fg = formula(gt)
    return [c for c in cands if c is not None and formula(c) == fg]


# --------------------------------------------------------------------------------------
# Per-sample metrics (each returns a scalar / category for one ground-truth + candidates)
# --------------------------------------------------------------------------------------
def hit_at_k(gt: Structure, cands: Sequence[Structure], k: int, matcher: StructureMatcher) -> int:
    """1 iff at least one of the top-k formula-matched candidates matches GT under ``matcher``."""
    for c in _formula_matched(gt, cands[:k]):
        if _guard(lambda c=c: matcher.get_rms_dist(gt, c)) is not None:
            return 1
    return 0

def min_rms_at_k(gt, cands, k, matcher) -> Optional[float]:
    """Minimum scale-normalized RMS over formula-matched top-k candidates (None if no match)."""
    best = None
    for c in _formula_matched(gt, cands[:k]):
        d = _guard(lambda c=c: matcher.get_rms_dist(gt, c))
        if d is not None:
            best = d[0] if best is None else min(best, d[0])
    return best

def match_abs(gt, cands, k, matcher_abs) -> int:
    """Scale-sensitive match: 1 iff a formula-matched candidate matches with rescaling disabled."""
    for c in _formula_matched(gt, cands[:k]):
        if _guard(lambda c=c: matcher_abs.get_rms_dist(gt, c)) is not None:
            return 1
    return 0

def rmsd_angstrom(gt, cands, k, matcher_abs) -> Optional[float]:
    """Absolute-Angstrom RMS over formula-matched candidates (min). Scales the normalized RMS by
    the per-atom length (V/N)**(1/3), following the paper's RMSD definition."""
    best = None
    vpa = (gt.volume / max(len(gt), 1)) ** (1.0 / 3.0)
    for c in _formula_matched(gt, cands[:k]):
        d = _guard(lambda c=c: matcher_abs.get_rms_dist(gt, c))
        if d is not None:
            val = d[0] * vpa
            best = val if best is None else min(best, val)
    return best

def sg_accuracy(gt, cands, symprec: float = 0.1) -> int:
    """Top-1 space-group accuracy: rank-1 candidate is formula-consistent AND shares GT space group."""
    t1 = cands[0] if cands else None
    if t1 is None or formula(t1) != formula(gt):
        return 0
    return int(space_group(t1, symprec) == space_group(gt, symprec))

def failure_mode(gt, cands, k, matcher, symprec: float = 0.1, lattice_tol: float = 0.20) -> str:
    """Categorize a sample. Returns 'hit' if Hit@k succeeds, else the top-1 failure category:
    'formula' / 'wrong_lattice' (lattice rel-err > lattice_tol) / 'lattice_ok_wrong_SG' / 'coord_other'."""
    if hit_at_k(gt, cands, k, matcher):
        return "hit"
    t1 = cands[0] if cands else None
    if t1 is None or formula(t1) != formula(gt):
        return "formula"
    if _lattice_relerr(t1, gt) > lattice_tol:
        return "wrong_lattice"
    if space_group(t1, symprec) != space_group(gt, symprec):
        return "lattice_ok_wrong_SG"
    return "coord_other"


# --------------------------------------------------------------------------------------
# Dataset-level distributional metric
# --------------------------------------------------------------------------------------
def wasserstein_lattice(pred_vals: Sequence[float], gt_vals: Sequence[float]) -> Optional[float]:
    """1D Wasserstein-1 distance between rank-1 and GT marginals of a lattice parameter (e.g. b, gamma)."""
    if wasserstein_distance is None or not pred_vals or not gt_vals:
        return None
    return float(wasserstein_distance(list(pred_vals), list(gt_vals)))


# --------------------------------------------------------------------------------------
# Convenience: score one sample across every per-sample metric at once
# --------------------------------------------------------------------------------------
@dataclass
class SampleResult:
    parsed: bool = False
    hit1: int = 0
    hit5: int = 0
    min_rms5: Optional[float] = None
    min_rms1: Optional[float] = None
    match_abs: int = 0
    rmsd_A: Optional[float] = None
    sg_acc: int = 0
    failure: str = "formula"
    top1_b: Optional[float] = None
    top1_gamma: Optional[float] = None
    n_atoms: Optional[int] = None

def score_sample(gt: Structure, cands: Sequence[Structure], k: int,
                 matcher: StructureMatcher, matcher_abs: StructureMatcher,
                 symprec: float = 0.1) -> SampleResult:
    """Compute every per-sample metric for one (GT, ranked candidates) pair."""
    cands = [c for c in cands if c is not None]
    r = SampleResult(n_atoms=len(gt))
    if not cands:
        return r
    r.parsed = True
    t1 = cands[0]
    r.top1_b = t1.lattice.b
    r.top1_gamma = t1.lattice.gamma
    r.hit1 = hit_at_k(gt, cands, 1, matcher)
    r.hit5 = hit_at_k(gt, cands, k, matcher)
    r.min_rms1 = min_rms_at_k(gt, cands, 1, matcher)
    r.min_rms5 = min_rms_at_k(gt, cands, k, matcher)
    r.match_abs = match_abs(gt, cands, k, matcher_abs)
    r.rmsd_A = rmsd_angstrom(gt, cands, k, matcher_abs)
    r.sg_acc = sg_accuracy(gt, cands, symprec)
    r.failure = failure_mode(gt, cands, k, matcher, symprec)
    return r
