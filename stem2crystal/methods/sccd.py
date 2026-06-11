"""SCCD (the paper model) as a registered ``Method``, so it benchmarks uniformly with the baselines.

SCCD's validated training/sampling code is the research stack under ``stem2crystal/methods/_sccd_impl/``
(it imports the DiffCSP + EMDiffuse packages). This adapter wraps that stack for the per-image
``generate`` interface. It needs:
    * the research stack on PYTHONPATH — set ``$SCCD_HOME`` to the dir holding the sccd reference
      scripts and the DiffCSP/EMDiffuse packages (i.e. the training repo);
    * the Phase-III checkpoint + the Phase-I evidence checkpoint (fetch with
      ``scripts/download_models.py --model sccd``), via ``config['checkpoints']`` or ``$SCCD_PHASE3`` /
      ``$SCCD_PHASE1``.

If the stack/weights are absent, :meth:`setup` raises a clear, actionable error. For the canonical
batched 4-GPU pipeline, run the reference scripts directly (see ``stem2crystal/methods/_sccd_impl/README.md``).
"""
from __future__ import annotations
import os
import sys
from typing import List
import numpy as np
from pymatgen.core import Structure, Composition, Lattice

from .base import Method, register_method


@register_method("sccd")
class SCCD(Method):
    """STEM2Crystal CoDiffusion — image+composition → 2D-slab crystal."""

    IMG = 256

    def setup(self) -> None:
        home = self.config.get("home") or os.environ.get("SCCD_HOME")
        if not home:
            raise EnvironmentError("Set $SCCD_HOME to the SCCD research stack (dir with the sccd reference "
                                   "scripts + the DiffCSP/EMDiffuse packages).")
        for p in (home, os.path.join(home, "scripts")):
            if os.path.isdir(p) and p not in sys.path:
                sys.path.insert(0, p)
        try:
            import torch
            import torch.nn as nn
            import gen_sccd_v5 as ref                                            # reference sampler
            from train_phase3_f2_guarded import (Phase3F2Guarded, load_f2_weights,  # type: ignore
                                                 load_evidence_weights, F2_CKPT)
        except Exception as e:
            raise ImportError("Could not import the SCCD research stack from $SCCD_HOME "
                              "(needs torch, torch-geometric, the DiffCSP + EMDiffuse packages).") from e

        ck = self.config.get("checkpoints") or {}
        phase3 = ck.get("phase3_joint") or os.environ.get("SCCD_PHASE3")
        phase1 = ck.get("phase1_evidence") or os.environ.get("SCCD_PHASE1") or str(getattr(ref, "EV_CKPT", ""))
        if not phase3 or not os.path.exists(phase3):
            raise FileNotFoundError("Set the Phase-III checkpoint (config['checkpoints']['phase3_joint'] or "
                                    "$SCCD_PHASE3). Fetch: scripts/download_models.py --model sccd.")

        self.torch, self.ref = torch, ref
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.dev = dev
        from pathlib import Path
        m = Phase3F2Guarded(g_img_dropout=0.0, inject_crys_to_img=True).to(dev)
        load_f2_weights(m, str(F2_CKPT)); load_evidence_weights(m, Path(phase1))
        base = m.crystal.base; lat_head = m.crystal.lattice_out_6d
        H = base.coord_out.in_features; nL = base.num_layers
        gimg_head = nn.Sequential(nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, 256)).to(dev)
        inj = nn.ModuleList([nn.Linear(256, H) for _ in range(nL + 1)]).to(dev)
        gate = nn.ModuleList([nn.Linear(H + 256, H) for _ in range(nL + 1)]).to(dev)
        s = torch.load(phase3, map_location="cpu", weights_only=False)
        base.load_state_dict(s["base"]); lat_head.load_state_dict(s["lat_head"])
        m.evidence_unet.load_state_dict(s["evidence_unet"]); m.g_crys_to_unet.load_state_dict(s["g_crys_to_unet"])
        gimg_head.load_state_dict(s["gimg_head"]); inj.load_state_dict(s["inj"]); gate.load_state_dict(s["gate"])
        for p in m.parameters():
            p.requires_grad_(False)
        m.eval()
        self.m, self.base, self.lat_head = m, base, lat_head
        self.gimg_head, self.inj, self.gate = gimg_head, inj, gate
        self.unk = s["unk"].to(dev)
        ab2 = m.beta_scheduler.alphas_cumprod[200]; self.ce0 = ab2.sqrt(); self.ce1 = (1 - ab2).sqrt()

    def generate(self, image: np.ndarray, composition: Composition, k: int = 5) -> List[Structure]:
        torch, ref = self.torch, self.ref
        from torch_geometric.data import Data, Batch
        from diffcsp.common.data_utils import chemical_symbols
        elems = [e for e, n in composition.element_composition.items() for _ in range(int(n))]
        z = torch.tensor([e.Z for e in elems], dtype=torch.long)
        b = Batch.from_data_list([Data(atom_types=z, num_atoms=torch.tensor([len(z)]), num_nodes=len(z))]).to(self.dev)
        from PIL import Image as _Im
        arr = np.asarray(image, np.float32)
        if arr.shape != (self.IMG, self.IMG):                       # model expects 256x256
            arr = np.asarray(_Im.fromarray((np.clip(arr, 0, 1) * 255).astype("uint8"))
                             .resize((self.IMG, self.IMG), _Im.BILINEAR), np.float32) / 255.0
        stem = torch.tensor(arr).view(1, 1, self.IMG, self.IMG).to(self.dev)
        ev_t = self.ce0 * stem + self.ce1 * torch.randn_like(stem)
        g_img = ref.gimg_of(self.m, self.gimg_head, ev_t, stem, torch.full((1,), 200, device=self.dev))
        out: List[Structure] = []
        for i in range(k):
            cfg = dict(seed=self.config.get("seed", 42) + i * 101, temp=1.0, step_lr=1.0, z_mid=0.5)
            x, zlat = ref.sample(self.m, self.base, self.lat_head, self.unk, self.inj, self.gate, g_img, b, cfg)
            from train_diffcsp6d_slabprior import slab_logvec_to_lattice_params  # type: ignore
            L, A = slab_logvec_to_lattice_params(zlat, project_ab=True)
            try:
                out.append(Structure(Lattice.from_parameters(*L[0].tolist(), *A[0].tolist()),
                                     [chemical_symbols[int(zz)] for zz in z.tolist()], x.numpy() % 1.0))
            except Exception:
                pass
        return out
