"""Minimal example: add a new method to STEM2Crystal-Bench in ~20 lines.

This 'method' ignores the image and returns the composition packed into a guessed hexagonal
2D-slab cell — a deliberately weak baseline that shows the full plug-in contract. Copy this file,
implement ``generate``, and you are done; the standalone evaluator scores it with every paper metric.

Try it:
    python scripts/generate.py --method example_random --benchmark synthetic --limit 20
    python scripts/evaluate.py  --method example_random --benchmark synthetic
"""
from __future__ import annotations
from typing import List
import numpy as np
from pymatgen.core import Structure, Lattice, Composition

from stem2crystal.methods import Method, register_method


@register_method("example_random")
class ExampleRandomSlab(Method):
    """Image-blind baseline: place the given atoms at random fractional sites in a guessed cell."""

    def setup(self) -> None:
        self.rng = np.random.default_rng(self.config.get("seed", 0))

    def generate(self, image: np.ndarray, composition: Composition, k: int = 5) -> List[Structure]:
        species = [sp for sp, n in composition.element_composition.items() for _ in range(int(n))]
        n = len(species)
        out = []
        for _ in range(k):
            a = float(self.rng.uniform(3.0, 4.0))             # guess in-plane lattice constant
            lat = Lattice.from_parameters(a, a, 20.0, 90, 90, 120)   # 2D slab with vacuum
            frac = self.rng.uniform(0, 1, size=(n, 3))
            frac[:, 2] = 0.5                                  # flatten into the slab plane
            try:
                out.append(Structure(lat, species, frac))
            except Exception:
                pass
        return out
