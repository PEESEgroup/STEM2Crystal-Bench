# MicroscopyGPT reference fine-tune (reconstructed recipe)

The upstream repo ships only inference for MicroscopyGPT (Llama-3.2-11B-Vision). These scripts
reconstruct a LoRA fine-tune so the method can be trained from scratch and reproduced end to end.

| File | Role |
|------|------|
| `finetune.py` | Build (STEM image, instruction, target-text) pairs from `train.csv` + unsloth LoRA SFT |
| `generate.py` | Inference with the fine-tuned adapter -> rank_1.cif per sample |

Target text matches the parser (`a b c` / `alpha beta gamma` / `El fx fy fz` per atom).

```bash
pip install unsloth trl
N_TRAIN=5000 EPOCHS=3 NOISE=low python finetune.py        # -> methods/models/repro_microscopygpt_lora
NOISE=low LIMIT=0 python generate.py                       # -> predictions/repro_microscopygpt_live/low
```
Env: `SCCD_HOME` (research stack root), `N_TRAIN`, `EPOCHS`, `NOISE`, `LIMIT`, `TAG`.
Note: a reconstructed recipe — not bit-identical to the released adapter (see REPRO_ISSUES #8/#10).
