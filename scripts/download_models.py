#!/usr/bin/env python3
"""Download model weights into a local, git-ignored ``outputs/`` directory.

No weights are committed to this repo. SCCD + DiffCSP-2D checkpoints are hosted on the Hugging Face
model hub; MatterGen and MicroscopyGPT weights come from their upstream sources.

    python scripts/download_models.py --model sccd          # SCCD 3-phase checkpoints
    python scripts/download_models.py --model diffcsp        # DiffCSP-2D checkpoint
    python scripts/download_models.py --model microscopygpt  # MicroscopyGPT LoRA adapter
    python scripts/download_models.py --model mattergen      # prints upstream instructions
    python scripts/download_models.py --model all
"""
import argparse
import os
from pathlib import Path

# Hugging Face model repos that host the weights we can redistribute.
HF_MODELS = {
    "sccd":    "gary23ai/STEM2Crystal-SCCD",     # phase1_evidence / phase2_structure / phase3_joint
    "diffcsp": "gary23ai/STEM2Crystal-DiffCSP2D",
    # MicroscopyGPT redistributes its LoRA adapter on its own card:
    "microscopygpt": "knc6/microscopy_gpt_llama3.2_vision_11b",
}

UPSTREAM_NOTE = {
    "mattergen": ("MatterGen weights are fetched automatically by the `mattergen` package "
                  "(checkpoint 'chemical_system_energy_above_hull'). Install with `pip install mattergen` "
                  "or clone https://github.com/microsoft/mattergen and set $MATTERGEN_HOME."),
    "microscopygpt": ("Also needs the base VLM `unsloth/llama-3.2-11b-vision-instruct-unsloth-bnb-4bit` "
                      "(auto-downloaded by unsloth). Set $MICROSCOPYGPT_ADAPTER to the downloaded LoRA dir."),
}


def out_root() -> Path:
    return Path(os.environ.get("STEM2CRYSTAL_MODELS", "outputs")).resolve()


def fetch(repo: str, dest: Path):
    from huggingface_hub import snapshot_download
    dest.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=repo, local_dir=str(dest))
    return dest


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=["all", "sccd", "diffcsp", "microscopygpt", "mattergen"], default="all")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    root = Path(a.out) if a.out else out_root()

    targets = ["sccd", "diffcsp", "microscopygpt", "mattergen"] if a.model == "all" else [a.model]
    for m in targets:
        if m in HF_MODELS:
            dest = root / m
            print(f"[{m}] downloading {HF_MODELS[m]} -> {dest}")
            try:
                fetch(HF_MODELS[m], dest)
            except Exception as e:
                print(f"[{m}] download failed ({e}). If the HF repo is private/not yet published, "
                      f"see docs/baselines.md for the upstream source.")
        if m in UPSTREAM_NOTE:
            print(f"[{m}] {UPSTREAM_NOTE[m]}")
    print(f"done -> {root}")


if __name__ == "__main__":
    main()
