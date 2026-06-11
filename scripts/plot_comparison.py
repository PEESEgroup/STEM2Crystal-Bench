#!/usr/bin/env python3
"""Render a ball-and-stick comparison of reconstructions across methods (the paper's Figure 4 style).

For each sample it draws one panel per method (the rank-1 candidate) and one for the ground truth,
and labels each prediction with its RMS to the ground truth (or "no match"). Methods are columns,
samples are rows.

  python scripts/plot_comparison.py --benchmark synthetic --noise low \
      --methods mattergen microscopygpt diffcsp sccd \
      --samples <sid1> <sid2> --out assets/figures/comparison.png

With --samples omitted, the first --n ids of the test split are used. Rendering needs ASE:
  pip install ase  (or:  pip install -e ".[viz]")
"""
import argparse
import json
from pathlib import Path

from pymatgen.core import Structure
from stem2crystal.data import benchmark_paths, default_data_root
from stem2crystal.eval.metrics import scale_matcher

# display names for the built-in methods (anything else is shown as-is)
LABELS = {"mattergen": "MatterGen", "microscopygpt": "MicroscopyGPT",
          "diffcsp": "DiffCSP", "sccd": "SCCD", "automat": "AutoMat"}


def _load(path: Path):
    try:
        return Structure.from_file(path)
    except Exception:
        return None


def _rms(pred, gt):
    """RMS to GT under the benchmark's scale-normalized matcher; None if no match."""
    if pred is None or gt is None:
        return None
    try:
        r = scale_matcher(primitive_cell=True).get_rms_dist(pred, gt)
        return r[0] if r else None
    except Exception:
        return None


def _draw(ax, struct, title, sub):
    ax.set_axis_off()
    ax.set_title(title, fontsize=11)
    if struct is None:
        ax.text(0.5, 0.5, "no match", ha="center", va="center",
                transform=ax.transAxes, color="#999", fontsize=10)
        return
    from ase.visualize.plot import plot_atoms
    from pymatgen.io.ase import AseAtomsAdaptor
    plot_atoms(AseAtomsAdaptor.get_atoms(struct), ax, radii=0.5, rotation="20x,30y,0z")
    if sub:
        ax.text(0.5, -0.06, sub, ha="center", va="top", transform=ax.transAxes, fontsize=9)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", default="synthetic", choices=["synthetic", "real5"])
    ap.add_argument("--noise", default="low", choices=["low", "mid", "high"])
    ap.add_argument("--methods", nargs="+", default=["mattergen", "microscopygpt", "diffcsp", "sccd"])
    ap.add_argument("--samples", nargs="*", default=None, help="sample ids (default: first --n of the split)")
    ap.add_argument("--n", type=int, default=2, help="number of samples when --samples is omitted")
    ap.add_argument("--pred-root", default="predictions")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--out", default="comparison.png")
    a = ap.parse_args()

    paths = benchmark_paths(a.benchmark, a.data_root or default_data_root())
    if a.samples:
        sids = a.samples
    else:
        ids = json.load(open(paths["split"]))
        ids = ids["test"] if isinstance(ids, dict) else ids
        sids = ids[:a.n]

    def pred_cif(method, sid):
        d = Path(a.pred_root) / method
        if a.benchmark == "synthetic":
            d = d / a.noise
        return d / f"{sid}_0" / "rank_1.cif"

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = list(a.methods) + ["gt"]
    fig, axes = plt.subplots(len(sids), len(cols),
                             figsize=(2.2 * len(cols), 2.4 * len(sids)), squeeze=False)
    for r, sid in enumerate(sids):
        gt = _load(paths["gt_dir"] / f"{sid}.cif")
        gt_formula = gt.composition.reduced_formula if gt is not None else sid
        for c, col in enumerate(cols):
            if col == "gt":
                _draw(axes[r][c], gt, "GT", gt_formula)
            else:
                pred = _load(pred_cif(col, sid))
                rms = _rms(pred, gt)
                sub = f"RMS={rms:.4f}" if rms is not None else "no match"
                _draw(axes[r][c], pred, LABELS.get(col, col), sub)

    fig.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=160, bbox_inches="tight")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
