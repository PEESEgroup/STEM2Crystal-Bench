#!/usr/bin/env python3
"""Reconstructed LoRA fine-tune of MicroscopyGPT (Llama-3.2-11B-Vision) for STEM->structure text.
Target text format matches eval5v2_microscopygpt.text2structure: 'a b c' / 'alpha beta gamma' / 'El fx fy fz'."""
import os, json, random, sys
os.environ.setdefault("UNSLOTH_RETURN_LOGITS", "1")
from pathlib import Path
from PIL import Image
from pymatgen.core import Structure
import pandas as pd

ROOT = Path(os.environ.get("SCCD_HOME", "/home/ubuntu/efs/KDD/EM/stem2cif-mattergen"))
DATA = ROOT / "data/all_stem_output"
DIFFCSP_DATA = ROOT / "methods/DiffCSP/data/stem2crystal"
OUT  = ROOT / "methods/models/repro_microscopygpt_lora"
N_TRAIN = int(os.environ.get("N_TRAIN", "3000"))
EPOCHS  = float(os.environ.get("EPOCHS", "1"))
NOISE   = os.environ.get("NOISE", "low")

def target_text(st):
    L = st.lattice
    s = f"{L.a:.3f} {L.b:.3f} {L.c:.3f}\n{L.alpha:.2f} {L.beta:.2f} {L.gamma:.2f}\n"
    for site in st:
        f = site.frac_coords
        s += f"{site.specie} {f[0]:.4f} {f[1]:.4f} {f[2]:.4f}\n"
    return s.strip()

def build_dataset():
    df = pd.read_csv(DIFFCSP_DATA / "train.csv")
    cif_by_id = dict(zip(df["material_id"].astype(str), df["cif"]))
    split = json.load(open(DATA / "split_repro.json"))
    ids = [s for s in split["train"] if s in cif_by_id][:N_TRAIN]
    rows = []
    for sid in ids:
        img_p = DATA / "images" / NOISE / f"{sid}_0.png"
        if not img_p.exists():
            continue
        try:
            st = Structure.from_str(cif_by_id[sid], fmt="cif")
        except Exception:
            continue
        formula = st.composition.reduced_formula
        instr = (f"The chemical formula is {formula}. Generate atomic structure description with "
                 "lattice lengths, angles, coordinates, and atom types. Also predict the Miller index.")
        img = Image.open(img_p).convert("L").resize((256, 256)).convert("RGB")
        rows.append({"messages": [
            {"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": instr}]},
            {"role": "assistant", "content": [{"type": "text", "text": target_text(st)}]},
        ]})
    return rows

def main():
    print(f"[mgpt] building dataset (N_TRAIN={N_TRAIN}, noise={NOISE})...", flush=True)
    rows = build_dataset()
    print(f"[mgpt] dataset size = {len(rows)}", flush=True)

    from unsloth import FastVisionModel
    import torch
    model, tokenizer = FastVisionModel.from_pretrained(
        "unsloth/Llama-3.2-11B-Vision-Instruct", load_in_4bit=True, use_gradient_checkpointing="unsloth")
    model = FastVisionModel.get_peft_model(
        model, finetune_vision_layers=False, finetune_language_layers=True,
        finetune_attention_modules=True, finetune_mlp_modules=True,
        r=16, lora_alpha=16, lora_dropout=0, bias="none", random_state=42)

    FastVisionModel.for_training(model)
    from unsloth.trainer import UnslothVisionDataCollator
    from trl import SFTTrainer, SFTConfig
    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        train_dataset=rows,
        args=SFTConfig(
            per_device_train_batch_size=2, gradient_accumulation_steps=4,
            warmup_steps=10, num_train_epochs=EPOCHS, learning_rate=2e-4,
            logging_steps=20, save_strategy="no", optim="adamw_8bit",
            weight_decay=0.01, lr_scheduler_type="linear", seed=42,
            output_dir=str(OUT / "_trainer"), report_to="none",
            remove_unused_columns=False, dataset_text_field="", dataset_kwargs={"skip_prepare_dataset": True},
            max_seq_length=2048),
    )
    print("[mgpt] starting LoRA fine-tune...", flush=True)
    trainer.train()
    OUT.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(OUT)); tokenizer.save_pretrained(str(OUT))
    print(f"[mgpt] DONE. adapter saved to {OUT}", flush=True)

if __name__ == "__main__":
    main()
