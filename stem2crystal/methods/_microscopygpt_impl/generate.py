#!/usr/bin/env python3
"""Generate test predictions with the reconstructed MicroscopyGPT LoRA adapter.
Writes predictions/<tag>/<noise>/<sid>_0/rank_1.cif (K=1, greedy), matching the harness layout."""
import os, json, time, sys
from pathlib import Path
from PIL import Image
import torch
from pymatgen.core import Structure, Lattice

ROOT = Path(os.environ.get("SCCD_HOME", "/home/ubuntu/efs/KDD/EM/stem2cif-mattergen"))
ADAPTER = ROOT / "methods/models/repro_microscopygpt_lora"
GT = ROOT / "data/all_stem_output"
OUTROOT = ROOT / "run_eval/predictions"
NOISE = os.environ.get("NOISE", "low")
LIMIT = int(os.environ.get("LIMIT", "0"))
TAG = os.environ.get("TAG", "repro_microscopygpt_live")

def text2structure(response):
    try:
        tail = response.split("assistant<|end_header_id|>")[-1] if "assistant<|end_header_id|>" in response else response
        tail = tail.split(". The")[0].strip("</s>").strip("<|eot_id|>").strip()
        lines = [l for l in tail.split("\n") if l.strip()]
        start = None
        for i, l in enumerate(lines):
            p = l.split()
            if len(p) == 3:
                try: [float(x) for x in p]; start = i; break
                except: pass
        if start is None: return None
        a, b, c = [float(x) for x in lines[start].split()]
        al, be, ga = [float(x) for x in lines[start+1].split()]
        atoms = []
        for l in lines[start+2:]:
            p = l.split()
            if len(p) < 4: continue
            try: atoms.append((p[0], [float(x) for x in p[1:4]]))
            except: pass
        if not atoms: return None
        return Structure(Lattice.from_parameters(a, b, c, al, be, ga), [e for e,_ in atoms], [c for _,c in atoms])
    except Exception:
        return None

def main():
    from unsloth import FastVisionModel
    model, tokenizer = FastVisionModel.from_pretrained(str(ADAPTER), load_in_4bit=True)
    FastVisionModel.for_inference(model)

    ids = json.load(open(GT / "split_repro.json"))["test"]
    if LIMIT: ids = ids[:LIMIT]
    ok = tot = 0
    for sid in ids:
        img_p = GT / "images" / NOISE / f"{sid}_0.png"
        cif_p = GT / "cifs" / f"{sid}.cif"
        if not img_p.exists() or not cif_p.exists(): continue
        tot += 1
        st_gt = Structure.from_file(str(cif_p))
        formula = st_gt.composition.reduced_formula
        img = Image.open(img_p).convert("L").resize((256, 256)).convert("RGB")
        instr = (f"The chemical formula is {formula}. Generate atomic structure description with "
                 "lattice lengths, angles, coordinates, and atom types. Also predict the Miller index.")
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": instr}]}]
        input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        inputs = tokenizer(img, input_text, add_special_tokens=False, return_tensors="pt").to("cuda")
        inputs.pop("token_type_ids", None)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=512, do_sample=False, use_cache=True)
        gen = tokenizer.batch_decode(out)[0]
        st = text2structure(gen)
        d = OUTROOT / TAG / NOISE / f"{sid}_0"; d.mkdir(parents=True, exist_ok=True)
        if st is not None:
            st.to(filename=str(d / "rank_1.cif")); ok += 1
        if tot % 25 == 0: print(f"[{NOISE}] {tot}/{len(ids)} parsed_ok={ok}", flush=True)
    print(f"[{NOISE}] DONE {ok}/{tot} parsed", flush=True)

if __name__ == "__main__":
    main()
