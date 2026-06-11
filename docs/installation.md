# Installation

## Requirements
- Python ≥ 3.9
- For **evaluation + data + the method interface**: `numpy`, `scipy`, `pymatgen`, `huggingface_hub`, `Pillow`.
- For **training / running SCCD**: a CUDA GPU + `torch`, `torch-geometric`, `torch-scatter`.

## Install

```bash
git clone <this-repo> stem2crystal && cd stem2crystal

# evaluation + benchmark + method interface (no GPU needed)
pip install -e .

# add model training / generation (PyTorch + PyG)
pip install -e ".[model]"
```

`torch-geometric` / `torch-scatter` wheels must match your CUDA/torch version; install them per the
[PyG instructions](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) if the
default pins fail.

## Sanity check
```bash
python -c "import stem2crystal, stem2crystal.eval; print('ok', stem2crystal.__version__)"
python scripts/download_benchmark.py --config real5      # small (5 samples) — quick smoke test
python -c "from stem2crystal.methods import list_methods; print(list_methods())"
```

## Where things go (all git-ignored)
| Dir | Contents | Created by |
|---|---|---|
| `data/` | downloaded benchmark | `scripts/download_benchmark.py` |
| `outputs/` | training checkpoints | `scripts/train_*.py` |
| `predictions/` | generated candidate CIFs | `scripts/generate.py` |
