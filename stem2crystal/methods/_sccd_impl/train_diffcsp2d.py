#!/usr/bin/env python3
"""
Train DiffCSP-2D on STEM2Crystal-Bench dataset.

Usage:
    cd /home/ubuntu/efs/KDD/stem2cif-mattergen/methods/DiffCSP
    python scripts/train_diffcsp2d.py --max_epochs 600
"""
import os, sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ["PROJECT_ROOT"] = str(PROJECT_ROOT)
os.environ["HYDRA_JOBS"] = str(PROJECT_ROOT / "outputs")
os.environ["WABDB_DIR"] = str(PROJECT_ROOT / "wandb")
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import torch

import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

import hydra
from omegaconf import OmegaConf

DATA_ROOT = PROJECT_ROOT / "data/stem2crystal"


def build_config(args):
    data_cfg = OmegaConf.create({
        "root_path": str(DATA_ROOT),
        "prop": "formation_energy_per_atom",
        "num_targets": 1,
        "niggli": True, "primitive": False,
        "graph_method": "crystalnn",
        "lattice_scale_method": "scale_length",
        "preprocess_workers": 30, "readout": "mean",
        "max_atoms": 100, "otf_graph": False,
        "eval_model_name": "stem2crystal",
        "tolerance": 0.1,
        "use_space_group": False, "use_pos_index": False,
        "train_max_epochs": args.max_epochs,
        "early_stopping_patience": 100000,
        "teacher_forcing_max_epoch": 300,
        "datamodule": {
            "_target_": "diffcsp.pl_data.datamodule.CrystDataModule",
            "datasets": {
                "train": {
                    "_target_": "diffcsp.pl_data.dataset.CrystDataset",
                    "name": "STEM2Crystal train", "path": str(DATA_ROOT / "train.csv"),
                    "save_path": str(DATA_ROOT / "train.pt"),
                    "prop": "formation_energy_per_atom",
                    "niggli": True, "primitive": False,
                    "graph_method": "crystalnn", "tolerance": 0.1,
                    "use_space_group": False, "use_pos_index": False,
                    "lattice_scale_method": "scale_length",
                    "preprocess_workers": 30,
                },
                "val": [{"_target_": "diffcsp.pl_data.dataset.CrystDataset",
                    "name": "STEM2Crystal val", "path": str(DATA_ROOT / "val.csv"),
                    "save_path": str(DATA_ROOT / "val.pt"),
                    "prop": "formation_energy_per_atom",
                    "niggli": True, "primitive": False,
                    "graph_method": "crystalnn", "tolerance": 0.1,
                    "use_space_group": False, "use_pos_index": False,
                    "lattice_scale_method": "scale_length",
                    "preprocess_workers": 30,
                }],
                "test": [{"_target_": "diffcsp.pl_data.dataset.CrystDataset",
                    "name": "STEM2Crystal test", "path": str(DATA_ROOT / "test.csv"),
                    "save_path": str(DATA_ROOT / "test.pt"),
                    "prop": "formation_energy_per_atom",
                    "niggli": True, "primitive": False,
                    "graph_method": "crystalnn", "tolerance": 0.1,
                    "use_space_group": False, "use_pos_index": False,
                    "lattice_scale_method": "scale_length",
                    "preprocess_workers": 30,
                }],
            },
            "num_workers": {"train": 0, "val": 0, "test": 0},
            "batch_size": {
                "train": args.batch_size,
                "val": min(64, args.batch_size),
                "test": min(64, args.batch_size),
            },
        },
    })

    # DiffCSP-2D model config
    model_cfg = OmegaConf.create({
        "_target_": "diffcsp.pl_modules.diffusion_2d.CSPDiffusion2D",
        "time_dim": 256, "latent_dim": 0,
        "cost_coord": 1.0,
        "lattice_dim": args.lattice_dim,
        "cost_lattice": 10.0,   # Higher weight for lattice (was 1.0)
        "max_neighbors": 20, "radius": 7.0, "timesteps": 1000,
        "decoder": {
            "_target_": "diffcsp.pl_modules.cspnet.CSPNet",
            "hidden_dim": 512, "latent_dim": 0,
            "max_atoms": 100, "num_layers": 6,
            "act_fn": "silu", "dis_emb": "sin", "num_freqs": 128,
            "edge_style": "fc", "max_neighbors": 20, "cutoff": 7.0,
            "ln": True, "ip": True,
        },
        "beta_scheduler": {
            "_target_": "diffcsp.pl_modules.diff_utils.BetaScheduler",
            "timesteps": 1000, "scheduler_mode": "cosine",
        },
        "sigma_scheduler": {
            "_target_": "diffcsp.pl_modules.diff_utils.SigmaScheduler",
            "timesteps": 1000, "sigma_begin": 0.005, "sigma_end": 0.5,
        },
    })

    optim_cfg = OmegaConf.create({
        "optimizer": {
            "_target_": "torch.optim.Adam",
            "lr": args.lr, "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0,
        },
        "use_lr_scheduler": True,
        "lr_scheduler": {
            "_target_": "torch.optim.lr_scheduler.ReduceLROnPlateau",
            "factor": 0.6, "patience": 30, "min_lr": 1e-4,
            "monitor": "train_loss_epoch",
        },
    })

    return OmegaConf.create({
        "data": data_cfg, "model": model_cfg,
        "optim": optim_cfg, "logging": {"val_check_interval": 2},
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_epochs", type=int, default=600)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--resume_ckpt", type=str, default=None)
    parser.add_argument("--lattice_dim", type=int, default=4, choices=[4, 6],
                        help="Lattice parameterization: 4=2D slab, 6=general 3D")
    parser.add_argument("--output_name", type=str, default=None,
                        help="Optional output directory name under outputs/")
    args = parser.parse_args()

    seed_everything(args.seed)

    suffix = "2d" if args.lattice_dim == 4 else "6d"
    output_name = args.output_name or f"stem2crystal_{suffix}"
    output_dir = PROJECT_ROOT / "outputs" / output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = build_config(args)

    print("[INFO] Instantiating CrystDataModule...")
    datamodule = hydra.utils.instantiate(cfg.data.datamodule, _recursive_=False)

    print("[INFO] Instantiating CSPDiffusion2D model...")
    model = hydra.utils.instantiate(
        cfg.model, optim=cfg.optim, data=cfg.data,
        logging=cfg.logging, _recursive_=False)

    if datamodule.scaler is not None:
        model.lattice_scaler = datamodule.lattice_scaler.copy()
        model.scaler = datamodule.scaler.copy()
    torch.save(datamodule.lattice_scaler, output_dir / "lattice_scaler.pt")
    torch.save(datamodule.scaler, output_dir / "prop_scaler.pt")
    OmegaConf.save(cfg, output_dir / "hparams.yaml")

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Model parameters: {num_params:,}")
    print(f"[INFO] Lattice loss weight: {cfg.model.cost_lattice}")

    callbacks = [
        ModelCheckpoint(
            dirpath=output_dir, monitor="val_loss", mode="min",
            save_top_k=1, save_last=True, verbose=True,
            filename="epoch={epoch}-val_loss={val_loss:.4f}"),
        EarlyStopping(monitor="val_loss", mode="min", patience=100, verbose=True),
    ]

    trainer = pl.Trainer(
        default_root_dir=output_dir,
        accelerator="gpu" if args.gpus > 0 else "cpu",
        devices=args.gpus if args.gpus > 0 else "auto",
        strategy=("ddp_find_unused_parameters_true" if (args.gpus and args.gpus > 1) else "auto"),
        max_epochs=args.max_epochs,
        callbacks=callbacks,
        gradient_clip_val=0.5, gradient_clip_algorithm="value",
        accumulate_grad_batches=1, check_val_every_n_epoch=2,
        deterministic=True, precision=32,
        logger=False, enable_progress_bar=True,
    )

    print(f"[INFO] Starting DiffCSP-2D training for {args.max_epochs} epochs...")
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=args.resume_ckpt)

    print(f"[INFO] Training complete. Best: {trainer.checkpoint_callback.best_model_path}")
    print(f"[INFO] Outputs: {output_dir}")


if __name__ == "__main__":
    main()
