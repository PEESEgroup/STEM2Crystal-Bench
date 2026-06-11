"""DiffCSP baseline — composition-conditioned diffusion crystal-structure prediction (Jiao et al.).
Image-blind: it ignores the STEM image and samples a structure from the given composition.

Upstream / weights (NOT in this repo — fetch with `scripts/download_models.py --model diffcsp`):
    code   : the DiffCSP package, importable; set $DIFFCSP_HOME to its checkpoint dir
    weights: a trained DiffCSP checkpoint (e.g. the 2D STEM2Crystal model) at $DIFFCSP_CKPT
Requires:  pip install torch torch-geometric ; DiffCSP's package on PYTHONPATH.
"""
from __future__ import annotations
import os
from typing import List
import numpy as np
from pymatgen.core import Structure, Composition

from .base import Method, register_method


@register_method("diffcsp")
class DiffCSP(Method):
    """Composition-only diffusion CSP baseline. Wraps the upstream DiffCSP sampler."""

    def setup(self) -> None:
        try:
            import torch  # noqa: F401
            from diffcsp.pl_modules.diffusion import CSPDiffusion  # type: ignore
        except Exception as e:
            raise ImportError("DiffCSP needs its package on PYTHONPATH (set $DIFFCSP_HOME) and "
                              "torch/torch-geometric. Fetch weights: scripts/download_models.py --model diffcsp.") from e
        ckpt = self.config.get("ckpt") or os.environ.get("DIFFCSP_CKPT")
        if not ckpt or not os.path.exists(ckpt):
            raise FileNotFoundError("Set $DIFFCSP_CKPT to a downloaded DiffCSP checkpoint "
                                    "(scripts/download_models.py --model diffcsp).")
        import torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CSPDiffusion.load_from_checkpoint(ckpt, map_location=self.device).to(self.device).eval()
        self.step_lr = self.config.get("step_lr", 1e-5)

    def generate(self, image: np.ndarray, composition: Composition, k: int = 5) -> List[Structure]:
        import torch
        from torch_geometric.data import Data, Batch
        from pymatgen.core import Lattice
        elems = [e for e, n in composition.element_composition.items() for _ in range(int(n))]
        z = torch.tensor([e.Z for e in elems], dtype=torch.long)
        out: List[Structure] = []
        for _ in range(k):
            d = Data(atom_types=z, num_atoms=torch.tensor([len(z)]), num_nodes=len(z))
            batch = Batch.from_data_list([d]).to(self.device)
            with torch.no_grad():
                res, _ = self.model.sample(batch, step_lr=self.step_lr)
            frac = res["frac_coords"].cpu().numpy()
            lengths = res["lengths"].cpu().numpy()[0]; angles = res["angles"].cpu().numpy()[0]
            try:
                out.append(Structure(Lattice.from_parameters(*lengths, *angles),
                                     [e.symbol for e in elems], frac))
            except Exception:
                pass
        return out
