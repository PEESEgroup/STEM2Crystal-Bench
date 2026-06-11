"""MatterGen baseline — large-scale *de-novo* diffusion materials generator (Zeni et al.).
It is composition-free and image-blind: it samples crystals from its prior; we keep, per target, the
samples whose reduced formula matches the requested composition (chemical-system conditioning).

Upstream / weights (NOT in this repo — fetch with `scripts/download_models.py --model mattergen`):
    code   : pip install mattergen  (or set $MATTERGEN_HOME to the cloned repo)
    weights: the official `chemical_system_energy_above_hull` checkpoint (auto-downloaded by mattergen)
Requires:  a CUDA GPU; the `mattergen` package importable.
"""
from __future__ import annotations
import os
import sys
from typing import List
import numpy as np
from pymatgen.core import Structure, Composition

from .base import Method, register_method


@register_method("mattergen")
class MatterGen(Method):
    """De-novo diffusion baseline; chemical-system conditioned, image-blind."""

    def setup(self) -> None:
        home = self.config.get("home") or os.environ.get("MATTERGEN_HOME")
        if home and home not in sys.path:
            sys.path.insert(0, home)
        try:
            from mattergen.generator import CrystalGenerator  # type: ignore
        except Exception as e:
            raise ImportError("MatterGen needs the `mattergen` package (pip install mattergen, or set "
                              "$MATTERGEN_HOME). Fetch the checkpoint: scripts/download_models.py --model mattergen.") from e
        self.checkpoint = self.config.get("checkpoint", "chemical_system_energy_above_hull")
        self.energy_above_hull = self.config.get("energy_above_hull", 0.0)
        self._CrystalGenerator = CrystalGenerator
        self._oversample = self.config.get("oversample", 8)   # draw more, keep formula-matched

    def generate(self, image: np.ndarray, composition: Composition, k: int = 5) -> List[Structure]:
        chem_sys = "-".join(sorted(e.symbol for e in composition.elements))
        gen = self._CrystalGenerator(checkpoint=self.checkpoint,
                                     chemical_system=chem_sys,
                                     energy_above_hull=self.energy_above_hull)
        target = composition.reduced_formula
        kept: List[Structure] = []
        for s in gen.generate(num_samples=k * self._oversample):   # de-novo draws
            try:
                if s.composition.reduced_formula == target:
                    kept.append(s)
                    if len(kept) >= k:
                        break
            except Exception:
                pass
        return kept[:k]
