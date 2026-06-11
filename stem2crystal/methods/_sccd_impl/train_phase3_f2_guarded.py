#!/usr/bin/env python3
"""Phase III F2-Guarded image-conditioned lattice residual training.

This script keeps Phase-II F2 as a protected reference and trains only a small
image-conditioned 6D lattice residual by default. The coordinate branch is not
image-conditioned; image features can only make bounded corrections to
log(a), log(b), and gamma/180 unless the user explicitly changes
``--delta_scale``.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_scatter import scatter
from tqdm import tqdm

ROOT = Path(os.environ.get("SCCD_HOME", "/home/ubuntu/efs/KDD/EM/stem2cif-mattergen"))
PROJ = ROOT / "methods/DiffCSP"
EMDIFFUSE = ROOT / "methods/EMDiffuse"
DATA_ROOT = PROJ / "data/stem2crystal"
STEM_ROOT = ROOT / "data/all_stem_output"
F2_CKPT = (
    PROJ
    / "outputs/phase2opt_F2_E1_lr2e5_w102_phys002_z002_ab002_e40"
    / "epoch=epoch=35-val_composite=val_composite=0.2288.ckpt"
)

sys.path.insert(0, str(PROJ))
sys.path.insert(0, str(PROJ / "scripts"))
sys.path.insert(0, str(EMDIFFUSE))
os.environ["PROJECT_ROOT"] = str(PROJ)
os.environ["HYDRA_JOBS"] = str(PROJ / "outputs")
os.environ["WABDB_DIR"] = str(PROJ / "wandb")

from diffcsp.common.data_utils import add_scaled_lattice_prop, get_scaler_from_data_list  # noqa: E402
from diffcsp.pl_data.stem_crystal_dataset import STEMCrystalDataset  # noqa: E402
from diffcsp.pl_modules.cspnet import CSPNet  # noqa: E402
from diffcsp.pl_modules.diff_utils import BetaScheduler, SigmaScheduler, d_log_p_wrapped_normal  # noqa: E402
from diffcsp.pl_modules.diffusion_2d import SinusoidalTimeEmbeddings, lattice_params_to_logvec  # noqa: E402
from models.guided_diffusion_modules.unet import UNet  # noqa: E402
from scripts.train_diffcsp6d_slabprior import slab_logvec_to_lattice_params, slab_logvec_to_matrix  # noqa: E402
from scripts.train_sccd_v6_6d import FFTFeatureExtractor  # noqa: E402

LATTICE_DIM = 6
NOISE_TO_IDX = {"low": 0, "mid": 1, "high": 2}


def sorted_l1(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(torch.sort(x.flatten())[0], torch.sort(y.flatten())[0])


def moment_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if x.numel() == 0:
        return x.sum() * 0.0
    xm, ym = x.mean(dim=0), y.mean(dim=0)
    xv = x.var(dim=0, unbiased=False)
    yv = y.var(dim=0, unbiased=False)
    return F.mse_loss(xm, ym) + F.mse_loss(xv, yv)


class STEMCrystalDatasetWithNoise(STEMCrystalDataset):
    """STEM dataset variant that records the sampled noise level."""

    def __getitem__(self, index):
        d = self.data[index]
        sid = d["mp_id"]
        prop = self.scaler.transform(d[self.prop]) if self.scaler else torch.tensor(0.0)
        frac_coords, atom_types, lengths, angles, edge_indices, to_jimages, num_atoms = d["graph_arrays"]

        view = random.choice(self.views)
        noise = random.choice(self.noises)
        cache_key = (sid, noise, view)
        if cache_key in self._cache:
            stem_img = self._cache[cache_key]
        else:
            img_path = os.path.join(self.stem_root, "images", noise, f"{sid}_{view}.png")
            stem_img = self._load_image(img_path)

        mask_key = ("mask", sid, view)
        if mask_key in self._cache:
            evidence = self._cache[mask_key]
        else:
            evidence = self._load_evidence(sid, view)

        return Data(
            frac_coords=torch.Tensor(frac_coords),
            atom_types=torch.LongTensor(atom_types),
            lengths=torch.Tensor(lengths).view(1, -1),
            angles=torch.Tensor(angles).view(1, -1),
            edge_index=torch.LongTensor(edge_indices.T).contiguous(),
            to_jimages=torch.LongTensor(to_jimages),
            num_atoms=num_atoms,
            num_bonds=edge_indices.shape[0],
            num_nodes=num_atoms,
            y=prop.view(1, -1),
            stem_image=torch.FloatTensor(stem_img).unsqueeze(0),
            evidence=torch.FloatTensor(evidence).unsqueeze(0),
            stem_noise_idx=torch.LongTensor([NOISE_TO_IDX[noise]]),
        )


class CSPNetImageResidual6D(nn.Module):
    """CSPNet 6D lattice head plus bounded image residual adapter."""

    def __init__(
        self,
        global_dim=256,
        lattice_dim=6,
        delta_scale=None,
        gate_bias=-4.0,
        conf_bias=-4.0,
        init_delta_std=0.0,
    ):
        super().__init__()
        self.base = CSPNet(
            hidden_dim=512,
            latent_dim=256,
            max_atoms=100,
            num_layers=6,
            act_fn="silu",
            dis_emb="sin",
            num_freqs=128,
            edge_style="fc",
            max_neighbors=20,
            cutoff=7.0,
            ln=True,
            ip=True,
        )
        hidden_dim = 512
        self.lattice_dim = lattice_dim
        self.lattice_out_6d = nn.Sequential(
            nn.Linear(hidden_dim + lattice_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, lattice_dim, bias=False),
        )
        self.img_gate = nn.Sequential(nn.Linear(global_dim, 64), nn.SiLU(), nn.Linear(64, lattice_dim))
        self.img_delta = nn.Sequential(nn.Linear(global_dim, 64), nn.SiLU(), nn.Linear(64, lattice_dim))
        self.img_conf = nn.Sequential(nn.Linear(global_dim, 64), nn.SiLU(), nn.Linear(64, 1))
        self.g_crys_head = nn.Sequential(nn.Linear(hidden_dim, global_dim), nn.SiLU(), nn.Linear(global_dim, global_dim))
        if delta_scale is None:
            delta_scale = [0.03, 0.03, 0.0, 0.0, 0.0, 0.03]
        self.register_buffer("delta_scale", torch.tensor(delta_scale, dtype=torch.float32))
        nn.init.zeros_(self.img_gate[-1].weight)
        nn.init.constant_(self.img_gate[-1].bias, gate_bias)
        if init_delta_std > 0:
            nn.init.normal_(self.img_delta[-1].weight, mean=0.0, std=init_delta_std)
        else:
            nn.init.zeros_(self.img_delta[-1].weight)
        nn.init.zeros_(self.img_delta[-1].bias)
        nn.init.zeros_(self.img_conf[-1].weight)
        nn.init.constant_(self.img_conf[-1].bias, conf_bias)

    def forward_base(self, t, atom_types, frac_coords, lattices, num_atoms, node2graph, z_lat):
        edges, frac_diff = self.base.gen_edges(num_atoms, frac_coords, lattices, node2graph)
        edge2graph = node2graph[edges[0]]
        node_features = self.base.node_embedding(atom_types - 1)
        t_per_atom = t.repeat_interleave(num_atoms, dim=0)
        node_features = self.base.atom_latent_emb(torch.cat([node_features, t_per_atom], dim=1))
        for i in range(self.base.num_layers):
            node_features = self.base._modules[f"csp_layer_{i}"](
                node_features, frac_coords, lattices, edges, edge2graph, frac_diff=frac_diff
            )
        if self.base.ln:
            node_features = self.base.final_layer_norm(node_features)
        coord_out = self.base.coord_out(node_features)
        graph_features = scatter(node_features, node2graph, dim=0, reduce="mean")
        pred_z = self.lattice_out_6d(torch.cat([graph_features, z_lat], dim=-1))
        return pred_z, coord_out, self.g_crys_head(graph_features)

    def image_residual(self, g_img):
        if g_img is None:
            return None, None
        gate = torch.sigmoid(self.img_gate(g_img))
        conf = torch.sigmoid(self.img_conf(g_img))
        delta = torch.tanh(self.img_delta(g_img)) * self.delta_scale[None, :]
        return conf * gate * delta, conf

    def forward(self, t, atom_types, frac_coords, lattices, num_atoms, node2graph, z_lat, g_img=None):
        pred_z, coord_out, g_crys = self.forward_base(
            t, atom_types, frac_coords, lattices, num_atoms, node2graph, z_lat
        )
        residual, conf = self.image_residual(g_img)
        if residual is not None:
            pred_z = pred_z + residual
        return pred_z, coord_out, g_crys, residual, conf


class Phase3F2Guarded(nn.Module):
    def __init__(
        self,
        delta_scale=None,
        train_lattice_head=False,
        cost_lattice=10.0,
        cost_coord=1.0,
        lambda_img=0.02,
        lambda_anchor_pred=0.5,
        lambda_anchor_z0=0.5,
        lambda_z0_gt=0.05,
        lambda_w1_log=0.02,
        lambda_w1_phys=0.02,
        lambda_kl_proxy=0.01,
        lambda_sparse=0.02,
        lambda_conf=0.005,
        lambda_align=0.0,
        z0_min_alpha_bar=0.7,
        w1_min_alpha_bar=0.2,
        g_img_dropout=0.1,
        noise_weights=None,
        gate_bias=-4.0,
        conf_bias=-4.0,
        init_delta_std=0.0,
        inject_crys_to_img=False,
        detach_crys_to_img=True,
        crys_to_img_scale=1.0,
    ):
        super().__init__()
        self.train_lattice_head = train_lattice_head
        self.cost_lattice = cost_lattice
        self.cost_coord = cost_coord
        self.lambda_img = lambda_img
        self.lambda_anchor_pred = lambda_anchor_pred
        self.lambda_anchor_z0 = lambda_anchor_z0
        self.lambda_z0_gt = lambda_z0_gt
        self.lambda_w1_log = lambda_w1_log
        self.lambda_w1_phys = lambda_w1_phys
        self.lambda_kl_proxy = lambda_kl_proxy
        self.lambda_sparse = lambda_sparse
        self.lambda_conf = lambda_conf
        self.lambda_align = lambda_align
        self.z0_min_alpha_bar = z0_min_alpha_bar
        self.w1_min_alpha_bar = w1_min_alpha_bar
        self.g_img_dropout = g_img_dropout
        self.inject_crys_to_img = inject_crys_to_img
        self.detach_crys_to_img = detach_crys_to_img
        self.crys_to_img_scale = crys_to_img_scale
        self.lattice_dim = LATTICE_DIM
        self.register_buffer("noise_scale", torch.tensor([1.0, 1.0, 0.5, 0.05, 0.05, 1.0], dtype=torch.float32))
        self.register_buffer("dim_weight", torch.tensor([2.0, 2.0, 0.2, 0.02, 0.02, 2.0], dtype=torch.float32))
        if noise_weights is None:
            noise_weights = [1.0, 1.0, 1.0]
        self.register_buffer("noise_weights", torch.tensor(noise_weights, dtype=torch.float32))

        self.evidence_unet = UNet(
            in_channel=2,
            out_channel=1,
            inner_channel=32,
            channel_mults=[1, 2, 4, 8],
            attn_res=[16],
            num_head_channels=32,
            res_blocks=2,
            dropout=0.1,
            image_size=256,
        )
        self._bottleneck_feat = None
        self._inject_g_crys = None
        self.evidence_unet.middle_block.register_forward_hook(self._middle_block_hook)
        self.fft_extractor = FFTFeatureExtractor(in_channels=256, out_dim=256)
        self.g_crys_to_unet = nn.Sequential(nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, 256))
        nn.init.zeros_(self.g_crys_to_unet[-1].weight)
        nn.init.zeros_(self.g_crys_to_unet[-1].bias)

        self.crystal = CSPNetImageResidual6D(
            global_dim=256,
            lattice_dim=LATTICE_DIM,
            delta_scale=delta_scale,
            gate_bias=gate_bias,
            conf_bias=conf_bias,
            init_delta_std=init_delta_std,
        )
        self.f2_ref = CSPNetImageResidual6D(global_dim=256, lattice_dim=LATTICE_DIM, delta_scale=delta_scale)
        self.beta_scheduler = BetaScheduler(timesteps=1000, scheduler_mode="cosine")
        self.sigma_scheduler = SigmaScheduler(timesteps=1000, sigma_begin=0.005, sigma_end=0.5)
        self.time_embedding = SinusoidalTimeEmbeddings(256)
        self.set_stage(1)

    def _middle_block_hook(self, _module, _inputs, output):
        if self.inject_crys_to_img and self._inject_g_crys is not None:
            g_crys = self._inject_g_crys.detach() if self.detach_crys_to_img else self._inject_g_crys
            output = output + self.crys_to_img_scale * self.g_crys_to_unet(g_crys)[:, :, None, None]
        self._bottleneck_feat = output
        return output

    def encode_lattice(self, lengths, angles):
        return lattice_params_to_logvec(lengths, angles, dim=6)

    def set_stage(self, stage: int):
        self.f2_ref.requires_grad_(False)
        self.evidence_unet.requires_grad_(False)
        self.crystal.requires_grad_(False)
        self.fft_extractor.requires_grad_(True)
        self.g_crys_to_unet.requires_grad_(True)
        self.crystal.img_gate.requires_grad_(True)
        self.crystal.img_delta.requires_grad_(True)
        self.crystal.img_conf.requires_grad_(True)
        if stage >= 2 and self.train_lattice_head:
            self.crystal.lattice_out_6d.requires_grad_(True)
        self.stage = stage

    def evidence_forward(self, x_t, stem, t, g_crys):
        self._inject_g_crys = g_crys
        try:
            pred_noise = self.evidence_unet(torch.cat([x_t, stem], dim=1), t)
        finally:
            self._inject_g_crys = None
        g_img = None
        if self._bottleneck_feat is not None:
            bn = self._bottleneck_feat
            if g_crys is not None and not self.inject_crys_to_img:
                bn = bn + self.crys_to_img_scale * self.g_crys_to_unet(g_crys.detach())[:, :, None, None]
            g_img = F.normalize(self.fft_extractor(bn), dim=-1)
            if self.training and self.g_img_dropout > 0:
                g_img = F.dropout(g_img, p=self.g_img_dropout, training=True)
        return pred_noise, g_img

    def forward(self, batch):
        bsz = batch.num_graphs
        device = batch.frac_coords.device
        times = self.beta_scheduler.uniform_sample_t(bsz, device)
        time_emb = self.time_embedding(times)
        alpha_bar = self.beta_scheduler.alphas_cumprod[times]
        c0 = torch.sqrt(alpha_bar)
        c1 = torch.sqrt(1.0 - alpha_bar)

        z_clean = self.encode_lattice(batch.lengths, batch.angles)
        rand_z = torch.randn_like(z_clean) * self.noise_scale[None, :]
        input_z = c0[:, None] * z_clean + c1[:, None] * rand_z
        input_z[:, 3:5] = z_clean[:, 3:5] + c1[:, None] * rand_z[:, 3:5]
        input_lattice = slab_logvec_to_matrix(input_z, project_ab=True)

        frac_coords = batch.frac_coords
        rand_x = torch.randn_like(frac_coords)
        sigmas = self.sigma_scheduler.sigmas[times]
        sigmas_norm = self.sigma_scheduler.sigmas_norm[times]
        sigmas_pa = sigmas.repeat_interleave(batch.num_atoms)[:, None]
        sigmas_norm_pa = sigmas_norm.repeat_interleave(batch.num_atoms)[:, None]
        input_frac = (frac_coords + sigmas_pa * rand_x) % 1.0
        tar_x = d_log_p_wrapped_normal(sigmas_pa * rand_x, sigmas_pa) / torch.sqrt(sigmas_norm_pa)

        evidence = batch.evidence if batch.evidence.dim() == 4 else batch.evidence.unsqueeze(1)
        stem = batch.stem_image if batch.stem_image.dim() == 4 else batch.stem_image.unsqueeze(1)
        noise_img = torch.randn_like(evidence)
        noisy_ev = c0[:, None, None, None] * evidence + c1[:, None, None, None] * noise_img

        with torch.no_grad():
            f2_z, f2_x, f2_g = self.f2_ref.forward_base(
                time_emb, batch.atom_types, input_frac, input_lattice, batch.num_atoms, batch.batch, input_z
            )
        pred_noise_img, g_img = self.evidence_forward(noisy_ev, stem, times, g_crys=f2_g)

        pred_base, pred_x, g_crys_pred, _, _ = self.crystal(
            time_emb, batch.atom_types, input_frac, input_lattice, batch.num_atoms, batch.batch, input_z, g_img=None
        )
        residual, conf = self.crystal.image_residual(g_img)
        if residual is None:
            residual = torch.zeros_like(pred_base)
            conf = torch.zeros(bsz, 1, device=device)
        pred_z = pred_base + residual

        graph_w = torch.ones(bsz, 1, device=device)
        noise_idx = getattr(batch, "stem_noise_idx", None)
        if noise_idx is not None:
            graph_w = self.noise_weights[noise_idx.view(-1).long()].view(-1, 1).to(device)

        diff_z = (pred_z - rand_z) ** 2
        loss_lat = (graph_w * diff_z * self.dim_weight[None, :]).mean()
        loss_coord = F.mse_loss(pred_x, tar_x) if pred_x.requires_grad else F.mse_loss(f2_x, tar_x)
        loss_img = F.mse_loss(pred_noise_img, noise_img)
        loss_anchor_pred = (graph_w * (pred_z - f2_z.detach()).pow(2)).mean()

        z0_hat = (input_z - c1[:, None] * pred_z) / c0[:, None].clamp(min=1e-6)
        z0_f2 = (input_z - c1[:, None] * f2_z.detach()) / c0[:, None].clamp(min=1e-6)
        loss_anchor_z0 = F.smooth_l1_loss(z0_hat[:, [0, 1, 5]], z0_f2[:, [0, 1, 5]])

        z0_mask = alpha_bar > self.z0_min_alpha_bar
        if z0_mask.any():
            loss_z0_gt = F.smooth_l1_loss(z0_hat[z0_mask][:, [0, 1, 5]], z_clean[z0_mask][:, [0, 1, 5]])
        else:
            loss_z0_gt = pred_z.sum() * 0.0

        w1_mask = alpha_bar > self.w1_min_alpha_bar
        if w1_mask.any():
            z_pred = z0_hat[w1_mask]
            z_tgt = z_clean[w1_mask]
            loss_w1_log = (
                sorted_l1(z_pred[:, 0], z_tgt[:, 0])
                + sorted_l1(z_pred[:, 1], z_tgt[:, 1])
                + sorted_l1(z_pred[:, 5], z_tgt[:, 5])
            ) / 3.0
            p_len, p_ang = slab_logvec_to_lattice_params(z_pred, project_ab=True)
            t_len, t_ang = slab_logvec_to_lattice_params(z_tgt, project_ab=True)
            loss_w1_phys = (
                sorted_l1(p_len[:, 0], t_len[:, 0]) / 10.0
                + sorted_l1(p_len[:, 1], t_len[:, 1]) / 10.0
                + sorted_l1(p_ang[:, 2], t_ang[:, 2]) / 180.0
            ) / 3.0
            loss_kl_proxy = moment_loss(z_pred[:, [0, 1, 5]], z_tgt[:, [0, 1, 5]])
        else:
            loss_w1_log = pred_z.sum() * 0.0
            loss_w1_phys = pred_z.sum() * 0.0
            loss_kl_proxy = pred_z.sum() * 0.0

        delta_norm = residual / self.crystal.delta_scale.clamp(min=1e-6)[None, :]
        active_dims = self.crystal.delta_scale[None, :] > 0
        loss_sparse = delta_norm[:, active_dims.squeeze(0)].abs().mean() if active_dims.any() else residual.sum() * 0.0
        loss_conf = conf.mean()
        if g_img is not None and self.lambda_align > 0:
            loss_align = (1.0 - F.cosine_similarity(g_img, g_crys_pred.detach(), dim=-1)).mean()
        else:
            loss_align = pred_z.sum() * 0.0

        loss = (
            self.cost_lattice * loss_lat
            + self.cost_coord * loss_coord
            + self.lambda_img * loss_img
            + self.lambda_anchor_pred * loss_anchor_pred
            + self.lambda_anchor_z0 * loss_anchor_z0
            + self.lambda_z0_gt * loss_z0_gt
            + self.lambda_w1_log * loss_w1_log
            + self.lambda_w1_phys * loss_w1_phys
            + self.lambda_kl_proxy * loss_kl_proxy
            + self.lambda_sparse * loss_sparse
            + self.lambda_conf * loss_conf
            + self.lambda_align * loss_align
        )
        val_composite = (
            loss_lat
            + 0.25 * loss_coord.detach()
            + loss_anchor_pred
            + loss_anchor_z0
            + loss_z0_gt
            + loss_w1_log
            + loss_w1_phys
            + loss_kl_proxy
            + 0.1 * loss_sparse
            + self.lambda_align * loss_align
        )
        return {
            "loss": loss,
            "loss_lattice": loss_lat,
            "loss_coord": loss_coord.detach() if not pred_x.requires_grad else loss_coord,
            "loss_img": loss_img,
            "loss_anchor_pred": loss_anchor_pred,
            "loss_anchor_z0": loss_anchor_z0,
            "loss_z0_gt": loss_z0_gt,
            "loss_w1_log": loss_w1_log,
            "loss_w1_phys": loss_w1_phys,
            "loss_kl_proxy": loss_kl_proxy,
            "loss_sparse": loss_sparse,
            "loss_conf": loss_conf,
            "loss_align": loss_align,
            "val_composite": val_composite,
            "conf_mean": conf.mean().detach(),
            "residual_abs": residual.abs().mean().detach(),
        }


def load_f2_weights(model: Phase3F2Guarded, ckpt_path: str) -> int:
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = state.get("state_dict", state)
    target = model.state_dict()
    mapped = {}
    for prefix in ("crystal", "f2_ref"):
        for key, val in state.items():
            new_key = None
            if key.startswith("decoder.base."):
                new_key = f"{prefix}.base." + key[len("decoder.base.") :]
            elif key.startswith("decoder.lattice_out_2d."):
                new_key = f"{prefix}.lattice_out_6d." + key[len("decoder.lattice_out_2d.") :]
            if new_key in target and target[new_key].shape == val.shape:
                mapped[new_key] = val
    model.load_state_dict(mapped, strict=False)
    model.f2_ref.requires_grad_(False)
    return len(mapped)


def load_evidence_weights(model: Phase3F2Guarded, path: Path) -> bool:
    if not path.exists():
        return False
    model.evidence_unet.load_state_dict(torch.load(path, map_location="cpu", weights_only=False), strict=True)
    return True


def parse_float_list(text: str, n: int) -> list[float]:
    vals = [float(x.strip()) for x in text.split(",") if x.strip()]
    if len(vals) != n:
        raise ValueError(f"expected {n} comma-separated floats, got {len(vals)}: {text}")
    return vals


def mean_meter(meters, key, n):
    return meters[key] / max(n, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--stage2_epoch", type=int, default=10**9)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--stage2_lr_scale", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_name", default="phase3_f2_guarded_A_adapter")
    parser.add_argument("--phase2_ckpt", default=str(F2_CKPT))
    parser.add_argument("--train_lattice_head", action="store_true")
    parser.add_argument("--delta_scale", default="0.03,0.03,0.0,0.0,0.0,0.03")
    parser.add_argument("--noise_weights", default="1.0,1.0,1.0")
    parser.add_argument("--lambda_img", type=float, default=0.02)
    parser.add_argument("--lambda_anchor_pred", type=float, default=0.5)
    parser.add_argument("--lambda_anchor_z0", type=float, default=0.5)
    parser.add_argument("--lambda_z0_gt", type=float, default=0.05)
    parser.add_argument("--lambda_w1_log", type=float, default=0.02)
    parser.add_argument("--lambda_w1_phys", type=float, default=0.02)
    parser.add_argument("--lambda_kl_proxy", type=float, default=0.01)
    parser.add_argument("--lambda_sparse", type=float, default=0.02)
    parser.add_argument("--lambda_conf", type=float, default=0.005)
    parser.add_argument("--lambda_align", type=float, default=0.0)
    parser.add_argument("--z0_min_alpha_bar", type=float, default=0.7)
    parser.add_argument("--w1_min_alpha_bar", type=float, default=0.2)
    parser.add_argument("--g_img_dropout", type=float, default=0.1)
    parser.add_argument("--gate_bias", type=float, default=-4.0)
    parser.add_argument("--conf_bias", type=float, default=-4.0)
    parser.add_argument("--init_delta_std", type=float, default=0.0)
    parser.add_argument("--inject_crys_to_img", action="store_true")
    parser.add_argument("--no_detach_crys_to_img", action="store_true")
    parser.add_argument("--crys_to_img_scale", type=float, default=1.0)
    parser.add_argument("--init_phase3_ckpt", default="")
    parser.add_argument("--precache", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = PROJ / "outputs" / args.output_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2))

    train_cryst = torch.load(DATA_ROOT / "train.pt", weights_only=False)
    val_cryst = torch.load(DATA_ROOT / "val.pt", weights_only=False)
    add_scaled_lattice_prop(train_cryst, "scale_length")
    add_scaled_lattice_prop(val_cryst, "scale_length")
    if args.limit:
        train_cryst = train_cryst[: args.limit]
        val_cryst = val_cryst[: max(1, min(args.limit // 4, len(val_cryst)))]

    train_ds = STEMCrystalDatasetWithNoise(train_cryst, STEM_ROOT, precache=args.precache)
    val_ds = STEMCrystalDatasetWithNoise(val_cryst, STEM_ROOT, precache=False)
    train_ds.lattice_scaler = get_scaler_from_data_list(train_cryst, key="scaled_lattice")
    train_ds.scaler = get_scaler_from_data_list(train_cryst, key="formation_energy_per_atom")
    val_ds.lattice_scaler = train_ds.lattice_scaler
    val_ds.scaler = train_ds.scaler
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=min(32, args.batch_size),
        shuffle=False,
        num_workers=max(1, args.num_workers // 2),
        pin_memory=torch.cuda.is_available(),
    )

    model = Phase3F2Guarded(
        delta_scale=parse_float_list(args.delta_scale, 6),
        train_lattice_head=args.train_lattice_head,
        lambda_img=args.lambda_img,
        lambda_anchor_pred=args.lambda_anchor_pred,
        lambda_anchor_z0=args.lambda_anchor_z0,
        lambda_z0_gt=args.lambda_z0_gt,
        lambda_w1_log=args.lambda_w1_log,
        lambda_w1_phys=args.lambda_w1_phys,
        lambda_kl_proxy=args.lambda_kl_proxy,
        lambda_sparse=args.lambda_sparse,
        lambda_conf=args.lambda_conf,
        lambda_align=args.lambda_align,
        z0_min_alpha_bar=args.z0_min_alpha_bar,
        w1_min_alpha_bar=args.w1_min_alpha_bar,
        g_img_dropout=args.g_img_dropout,
        noise_weights=parse_float_list(args.noise_weights, 3),
        gate_bias=args.gate_bias,
        conf_bias=args.conf_bias,
        init_delta_std=args.init_delta_std,
        inject_crys_to_img=args.inject_crys_to_img,
        detach_crys_to_img=not args.no_detach_crys_to_img,
        crys_to_img_scale=args.crys_to_img_scale,
    ).to(device)
    n_loaded = load_f2_weights(model, args.phase2_ckpt)
    ev_path = PROJ / "outputs/phase1_evidence/best_evidence_unet.pt"
    ev_loaded = load_evidence_weights(model, ev_path)
    if args.init_phase3_ckpt:
        state = torch.load(args.init_phase3_ckpt, map_location="cpu", weights_only=False)
        loaded = model.load_state_dict(state.get("state_dict", state), strict=False)
        print(
            f"[INFO] initialized Phase III state={args.init_phase3_ckpt} "
            f"missing={len(loaded.missing_keys)} unexpected={len(loaded.unexpected_keys)}",
            flush=True,
        )
    model.set_stage(1)
    print(f"[INFO] device={device} loaded F2 tensors={n_loaded} evidence={ev_loaded}", flush=True)
    print(f"[INFO] trainable parameters={sum(p.numel() for p in model.parameters() if p.requires_grad):,}", flush=True)

    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    best = float("inf")
    best_epoch = 0
    meter_keys = [
        "loss",
        "loss_lattice",
        "loss_coord",
        "loss_img",
        "loss_anchor_pred",
        "loss_anchor_z0",
        "loss_z0_gt",
        "loss_w1_log",
        "loss_w1_phys",
        "loss_kl_proxy",
        "loss_sparse",
        "loss_conf",
        "loss_align",
        "conf_mean",
        "residual_abs",
    ]

    for epoch in range(1, args.epochs + 1):
        if epoch == args.stage2_epoch:
            model.set_stage(2)
            opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr * args.stage2_lr_scale)
            print(
                f"[INFO] switched to stage 2 at epoch {epoch}; "
                f"trainable={sum(p.numel() for p in model.parameters() if p.requires_grad):,}",
                flush=True,
            )
        model.train()
        meters = {k: 0.0 for k in meter_keys}
        n = 0
        for batch in tqdm(train_loader, desc=f"epoch {epoch}", disable=True):
            batch = batch.to(device)
            out = model(batch)
            opt.zero_grad(set_to_none=True)
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            for k in meter_keys:
                meters[k] += float(out[k].detach().cpu())
            n += 1

        model.eval()
        vloss = 0.0
        vcomp = 0.0
        vn = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch)
                vloss += float(out["loss"].detach().cpu()) * batch.num_graphs
                vcomp += float(out["val_composite"].detach().cpu()) * batch.num_graphs
                vn += batch.num_graphs
        vloss /= max(vn, 1)
        vcomp /= max(vn, 1)
        if vcomp < best:
            best = vcomp
            best_epoch = epoch
            torch.save(model.state_dict(), out_dir / "best.pt")
            tag = " *BEST*"
        else:
            tag = ""
        torch.save(model.state_dict(), out_dir / "latest.pt")
        print(
            f"epoch {epoch:03d} train={mean_meter(meters,'loss',n):.4f} "
            f"lat={mean_meter(meters,'loss_lattice',n):.4f} coord={mean_meter(meters,'loss_coord',n):.4f} "
            f"img={mean_meter(meters,'loss_img',n):.4f} a_pred={mean_meter(meters,'loss_anchor_pred',n):.5f} "
            f"a_z0={mean_meter(meters,'loss_anchor_z0',n):.5f} z0={mean_meter(meters,'loss_z0_gt',n):.5f} "
            f"w1={mean_meter(meters,'loss_w1_log',n):.5f}/{mean_meter(meters,'loss_w1_phys',n):.5f} "
            f"klp={mean_meter(meters,'loss_kl_proxy',n):.5f} sparse={mean_meter(meters,'loss_sparse',n):.5f} "
            f"align={mean_meter(meters,'loss_align',n):.5f} "
            f"conf={mean_meter(meters,'conf_mean',n):.5f} res={mean_meter(meters,'residual_abs',n):.6f} "
            f"val={vloss:.4f} comp={vcomp:.4f}{tag}",
            flush=True,
        )
    print(f"[INFO] done best_composite={best:.4f} epoch={best_epoch}", flush=True)


if __name__ == "__main__":
    main()
