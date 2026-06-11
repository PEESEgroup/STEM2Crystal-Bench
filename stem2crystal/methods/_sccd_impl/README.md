# SCCD reference implementation (research code)

The exact scripts that train and sample the paper's SCCD model. These are **research code**: they run
against the full training stack (the `DiffCSP` and `EMDiffuse` packages) and expect the benchmark on
disk. They are included for transparency and reproduction, not as a relocatable library.

| File | Role |
|------|------|
| `build_split.py` | Reconstruct a train/val/test split from the rendered images (the packaged split is test-only) |
| `train_phase1_evidence.py` | Phase I — evidence denoiser (EMDiffuse U-Net). **Multi-GPU via `torchrun`** |
| `train_diffcsp2d.py`       | Phase II — structure branch (composition-only warm start). Multi-GPU via `--gpus N` |
| `train_sccd_v5_ddp.py`     | Phase III — joint bidirectional co-diffusion (= paper SCCD), 4-GPU DDP |
| `train_sccd_v5_film.py`    | Phase III, FiLM image→structure coupling (equivalent variant) |
| `train_phase3_f2_guarded.py` | model definition + dataset + checkpoint loaders (imported by the above) |
| `gen_sccd_v5.py`           | synthetic-benchmark sampler (writes rank_*.cif) |
| `gen_real5v2_diffusion.py` | Real-5 sampler |

End-to-end from scratch (from the training environment):
```bash
# 0) reconstruct the train/val/test split the trainers need
python build_split.py --stem-root $STEM_ROOT/all_stem_output
# 1) Phase I evidence (multi-GPU). Paths via env: EMDIFFUSE_ROOT, STEM_ROOT, PHASE1_OUT, EMDIFFUSE_INIT, SPLIT_FILE
SPLIT_FILE=split_repro.json torchrun --nproc_per_node=4 train_phase1_evidence.py --epochs 100
# 2) Phase III joint co-diffusion (loads Phase I via PHASE1_CKPT + the Phase II F2 warm start)
PHASE1_CKPT=$PROJ/outputs/phase1_evidence/best_evidence_unet.pt \
  torchrun --nproc_per_node=4 train_sccd_v5_ddp.py --epochs 60 --bs 16 --out sccd_v5
# 3) sample + score
python gen_sccd_v5.py --ckpt sccd_v5/best.pt --noise low --tag sccd --K 5
```
All scripts take their research-stack root from `$SCCD_HOME` (default the original path).

### Known reproduction caveats (see top-level `docs/REPRODUCTION_NOTES.md`)
- **Phase II F2 warm start** (`phase2opt_F2_*`) is loaded by Phase III but its exact training entry is
  **not shipped** here; reproduce Phase I + Phase III from scratch and reuse a Phase II warm start.
- **DiffCSP generation**: `generate_diffcsp2d_predictions.py` applies 2D-slab lattice priors that change
  absolute-scale metrics (W₁/Match_abs); pin one generation config for apples-to-apples comparison.

To benchmark a **new** model instead, use the clean `stem2crystal.methods.Method` interface
(see `docs/add_a_method.md`) — you do not need this research stack.
