#!/usr/bin/env python3
"""
Phase I: Pretrain Evidence Denoiser using EMDiffuse U-Net architecture.

Trains D_img to denoise STEM images: noisy STEM (y) → clean evidence map (x₀).
Uses DDPM noise-prediction objective + auxiliary ℓ₁ reconstruction loss + gradient loss.

Features:
- EMDiffuse pretrained weight initialization
- Early stopping on val_loss with patience
- Per-epoch denoising visualization (original STEM / denoised / GT evidence)

Usage:
    cd /home/ubuntu/efs/KDD/stem2cif-mattergen/methods/DiffCSP
    python scripts/train_phase1_evidence.py --epochs 100 --batch_size 16

Multi-GPU:  torchrun --nproc_per_node=4 train_phase1_evidence.py --epochs 100
Single-GPU: python train_phase1_evidence.py --epochs 100
Paths via env: EMDIFFUSE_ROOT, STEM_ROOT, PHASE1_OUT, EMDIFFUSE_INIT, SPLIT_FILE.
"""
import os, sys, argparse, json, time, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add EMDiffuse to path for its U-Net
EMDIFFUSE_ROOT = os.environ.get("EMDIFFUSE_ROOT", "/home/ubuntu/efs/KDD/stem2cif-mattergen/methods/EMDiffuse")
sys.path.insert(0, EMDIFFUSE_ROOT)
from models.guided_diffusion_modules.unet import UNet

STEM_ROOT = os.environ.get("STEM_ROOT", "/home/ubuntu/efs/KDD/stem2cif-mattergen/data/all_stem_output")
OUTPUT_DIR = os.environ.get("PHASE1_OUT", "/home/ubuntu/efs/KDD/stem2cif-mattergen/methods/DiffCSP/outputs/phase1_evidence")

IMG_SIZE = 256


# ── Dataset ──────────────────────────────────────────────────────────────

class STEMEvidenceDataset(Dataset):
    def __init__(self, structure_ids, stem_root, img_size=256,
                 noises=('low', 'mid', 'high'), views=(0, 1, 2)):
        self.sids = structure_ids
        self.stem_root = Path(stem_root)
        self.img_size = img_size
        self.noises = list(noises)
        self.views = list(views)

    def __len__(self):
        return len(self.sids)

    def _load_img(self, path):
        try:
            im = Image.open(path).convert('L')
            im = im.resize((self.img_size, self.img_size), Image.BILINEAR)
            return np.array(im, dtype=np.float32) / 255.0
        except:
            return np.zeros((self.img_size, self.img_size), dtype=np.float32)

    def __getitem__(self, idx):
        sid = self.sids[idx]
        view = random.choice(self.views)
        noise = random.choice(self.noises)
        stem_path = self.stem_root / 'images' / noise / f'{sid}_{view}.png'
        stem_img = self._load_img(str(stem_path))
        mask_path = self.stem_root / 'masks' / f'{sid}_{view}_mask.png'
        evidence = self._load_img(str(mask_path))
        return {
            'stem': torch.FloatTensor(stem_img).unsqueeze(0),
            'evidence': torch.FloatTensor(evidence).unsqueeze(0),
            'sid': sid, 'noise': noise, 'view': view,
        }


# ── DDPM Beta Schedule ───────────────────────────────────────────────────

def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def gradient_loss(pred, target):
    dx_pred = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    dx_target = target[:, :, :, 1:] - target[:, :, :, :-1]
    dy_pred = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    dy_target = target[:, :, 1:, :] - target[:, :, :-1, :]
    return F.l1_loss(dx_pred, dx_target) + F.l1_loss(dy_pred, dy_target)


# ── Visualization ────────────────────────────────────────────────────────

@torch.no_grad()
def one_step_denoise(model, evidence, stem, alphas_cumprod, t_val=50):
    """Quick single-step denoising for visualization (not full DDPM reverse)."""
    B = evidence.size(0)
    t = torch.full((B,), t_val, device=evidence.device)
    noise = torch.randn_like(evidence)
    sqrt_ac = alphas_cumprod[t].sqrt()[:, None, None, None]
    sqrt_om = (1 - alphas_cumprod[t]).sqrt()[:, None, None, None]
    x_t = sqrt_ac * evidence + sqrt_om * noise
    model_input = torch.cat([x_t, stem], dim=1)
    pred_noise = model(model_input, t)
    x0_pred = (x_t - sqrt_om * pred_noise) / sqrt_ac.clamp(min=1e-6)
    return x0_pred.clamp(0, 1)


def save_epoch_visualization(model, val_ds, alphas_cumprod, epoch, output_dir,
                              n_samples=6):
    """Save denoising examples: STEM (noisy) | Denoised | GT evidence."""
    model.eval()
    viz_dir = Path(output_dir) / 'viz'
    viz_dir.mkdir(parents=True, exist_ok=True)

    indices = random.sample(range(len(val_ds)), min(n_samples, len(val_ds)))

    fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4 * n_samples))
    if n_samples == 1:
        axes = axes[None, :]

    for i, idx in enumerate(indices):
        sample = val_ds[idx]
        stem = sample['stem'].unsqueeze(0).cuda()
        evidence = sample['evidence'].unsqueeze(0).cuda()
        sid = sample['sid']

        denoised = one_step_denoise(model, evidence, stem, alphas_cumprod, t_val=100)

        stem_np = stem[0, 0].cpu().numpy()
        gt_np = evidence[0, 0].cpu().numpy()
        pred_np = denoised[0, 0].cpu().numpy()

        axes[i, 0].imshow(stem_np, cmap='gray', vmin=0, vmax=1)
        axes[i, 0].set_title('Noisy STEM' if i == 0 else '')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(pred_np, cmap='gray', vmin=0, vmax=1)
        axes[i, 1].set_title('Denoised (predicted x₀)' if i == 0 else '')
        axes[i, 1].axis('off')

        axes[i, 2].imshow(gt_np, cmap='gray', vmin=0, vmax=1)
        axes[i, 2].set_title('GT Evidence (mask)' if i == 0 else '')
        axes[i, 2].axis('off')

        formula = sid.split('__')[1].split('-')[0] if '__' in sid else sid[:15]
        axes[i, 0].set_ylabel(formula, fontsize=9, rotation=0, labelpad=50, va='center')

    plt.suptitle(f'Phase I Epoch {epoch}', fontsize=14)
    plt.tight_layout()
    fig.savefig(viz_dir / f'epoch_{epoch:03d}.png', dpi=120, bbox_inches='tight')
    plt.close(fig)


# ── Training (DDP) ──

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--lambda_recon", type=float, default=1.0)
    parser.add_argument("--lambda_grad", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # ── DDP setup ──
    ddp = "LOCAL_RANK" in os.environ
    if ddp:
        dist.init_process_group("nccl")
        rank = dist.get_rank(); world = dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        rank, world, local_rank = 0, 1, 0
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    is_main = (rank == 0)

    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)

    output_dir = Path(OUTPUT_DIR)
    if is_main: output_dir.mkdir(parents=True, exist_ok=True)

    with open(Path(STEM_ROOT) / os.environ.get('SPLIT_FILE', 'split.json')) as f:
        split = json.load(f)
    train_ds = STEMEvidenceDataset(split['train'], STEM_ROOT)
    val_ds = STEMEvidenceDataset(split['val'], STEM_ROOT)

    train_sampler = DistributedSampler(train_ds, num_replicas=world, rank=rank, shuffle=True) if ddp else None
    val_sampler = DistributedSampler(val_ds, num_replicas=world, rank=rank, shuffle=False) if ddp else None
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=(train_sampler is None),
                              sampler=train_sampler, num_workers=6, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            sampler=val_sampler, num_workers=4, pin_memory=True)
    if is_main: print(f"[INFO] Train: {len(train_ds)}, Val: {len(val_ds)}, world={world}", flush=True)

    model = UNet(in_channel=2, out_channel=1, inner_channel=32, channel_mults=[1, 2, 4, 8],
                 attn_res=[16], num_head_channels=32, res_blocks=2, dropout=0.2, image_size=IMG_SIZE).to(device)
    if is_main: print(f"[INFO] EMDiffuse U-Net: {sum(p.numel() for p in model.parameters()):,} params", flush=True)

    PRETRAINED_PATH = os.environ.get("EMDIFFUSE_INIT", "/home/ubuntu/efs/KDD/EMDiffuse_model_weight/EMDiffuse-n/best_Network_ema.pth")
    if os.path.exists(PRETRAINED_PATH):
        ckpt = torch.load(PRETRAINED_PATH, map_location='cpu', weights_only=False)
        state = {k.replace('denoise_fn.', ''): v for k, v in ckpt.items()}
        model.load_state_dict(state, strict=False)
        if is_main: print("[INFO] Loaded EMDiffuse pretrained weights", flush=True)
    elif is_main:
        print("[WARN] No pretrained weights, training from scratch", flush=True)

    if ddp:
        model = DDP(model, device_ids=[local_rank])
    raw_model = model.module if ddp else model

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    betas = cosine_beta_schedule(args.timesteps).to(device)
    alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)

    best_val_loss = float('inf'); patience_counter = 0
    for epoch in range(1, args.epochs + 1):
        if ddp: train_sampler.set_epoch(epoch)
        model.train(); epoch_loss = epoch_ddpm = 0.0; n_batches = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", disable=not is_main)
        for batch in pbar:
            evidence = batch['evidence'].to(device, non_blocking=True)
            stem = batch['stem'].to(device, non_blocking=True)
            B = evidence.size(0)
            t = torch.randint(0, args.timesteps, (B,), device=device)
            noise = torch.randn_like(evidence)
            sqrt_ac = alphas_cumprod[t].sqrt()[:, None, None, None]
            sqrt_om = (1 - alphas_cumprod[t]).sqrt()[:, None, None, None]
            x_t = sqrt_ac * evidence + sqrt_om * noise
            pred_noise = model(torch.cat([x_t, stem], dim=1), t)
            loss_ddpm = F.mse_loss(pred_noise, noise)
            x0_pred = (x_t - sqrt_om * pred_noise) / sqrt_ac.clamp(min=1e-6)
            loss = loss_ddpm + args.lambda_recon * F.l1_loss(x0_pred, evidence) + args.lambda_grad * gradient_loss(x0_pred, evidence)
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            epoch_loss += loss.item(); epoch_ddpm += loss_ddpm.item(); n_batches += 1
            if is_main: pbar.set_postfix(ddpm=f"{loss_ddpm.item():.4f}", loss=f"{loss.item():.4f}")
        scheduler.step()
        avg_train = epoch_loss / max(n_batches, 1); avg_ddpm = epoch_ddpm / max(n_batches, 1)

        model.eval(); val_loss = 0.0; val_n = 0
        with torch.no_grad():
            for batch in val_loader:
                evidence = batch['evidence'].to(device); stem = batch['stem'].to(device); B = evidence.size(0)
                t = torch.randint(0, args.timesteps, (B,), device=device)
                noise = torch.randn_like(evidence)
                sqrt_ac = alphas_cumprod[t].sqrt()[:, None, None, None]
                sqrt_om = (1 - alphas_cumprod[t]).sqrt()[:, None, None, None]
                x_t = sqrt_ac * evidence + sqrt_om * noise
                pred_noise = model(torch.cat([x_t, stem], dim=1), t)
                val_loss += F.mse_loss(pred_noise, noise).item() * B; val_n += B
        vt = torch.tensor([val_loss, float(val_n)], device=device)
        if ddp: dist.all_reduce(vt)
        avg_val = (vt[0] / vt[1].clamp(min=1)).item()

        improved = ""
        if avg_val < best_val_loss:
            best_val_loss = avg_val; patience_counter = 0
            if is_main: torch.save(raw_model.state_dict(), output_dir / 'best_evidence_unet.pt')
            improved = " *BEST*"
        else:
            patience_counter += 1
        if is_main:
            print(f"  Epoch {epoch}: train={avg_train:.4f} ddpm={avg_ddpm:.4f} val={avg_val:.6f} "
                  f"patience={patience_counter}/{args.patience}{improved}", flush=True)
            torch.save(raw_model.state_dict(), output_dir / 'latest_evidence_unet.pt')
        if patience_counter >= args.patience:
            if is_main: print(f"\n[INFO] Early stopping at epoch {epoch}", flush=True)
            break

    if is_main: print(f"\n[INFO] Phase I complete. Best val_loss: {best_val_loss:.6f}", flush=True)
    if ddp: dist.destroy_process_group()


if __name__ == "__main__":
    train()
