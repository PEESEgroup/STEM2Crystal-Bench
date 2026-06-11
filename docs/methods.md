# Methods

Every method — SCCD and the baselines — is a registered
`stem2crystal.methods.Method`, so they generate and evaluate through the same harness.

```python
from stem2crystal.methods import list_methods
list_methods()      # ['diffcsp', 'mattergen', 'microscopygpt', 'sccd']  (+ any you register)
```

| Key | Method | Conditioning | Upstream stack (not in this repo) | Weights |
|---|---|---|---|---|
| `sccd` | STEM2Crystal CoDiffusion (ours) | image + composition | DiffCSP + EMDiffuse (set `$SCCD_HOME`) | `download_models.py --model sccd` |
| `diffcsp` | DiffCSP (Jiao et al.) | composition only | `diffcsp` package (`$DIFFCSP_HOME`) | `--model diffcsp` |
| `microscopygpt` | MicroscopyGPT (Choudhary et al.) | image + composition | `unsloth` + 11B VLM | `--model microscopygpt` |
| `mattergen` | MatterGen (Zeni et al.) | de-novo (chemical system) | `mattergen` package (`$MATTERGEN_HOME`) | auto by `mattergen` |

> **No weights or outputs are committed.** Fetch weights with
> [`scripts/download_models.py`](../scripts/download_models.py) into a git-ignored `outputs/`.
> Baselines also need their upstream package installed (they are large, independent codebases);
> the adapter raises a clear, actionable error if the package or checkpoint is missing.

## Run any method
```bash
python scripts/download_models.py --model diffcsp
python scripts/generate.py --method diffcsp --benchmark synthetic --noise low
python scripts/evaluate.py  --method diffcsp --benchmark synthetic --noise low
```
The composition-only and de-novo baselines ignore the `image` argument; image-conditioned methods
(`sccd`, `microscopygpt`) use it.

## Setup per baseline

**DiffCSP** — `pip install` its repo (or set `$DIFFCSP_HOME`); `export DIFFCSP_CKPT=outputs/diffcsp/...ckpt`.

**MicroscopyGPT** — `pip install unsloth`; `download_models.py --model microscopygpt` fetches the LoRA
adapter; `export MICROSCOPYGPT_ADAPTER=outputs/microscopygpt`. The base 11B VLM is auto-downloaded by
unsloth (≈12 GB GPU).

**MatterGen** — `pip install mattergen` (or set `$MATTERGEN_HOME`); the
`chemical_system_energy_above_hull` checkpoint is fetched by the package on first use.

**SCCD** — `export SCCD_HOME=<training-repo>` (the dir with the sccd reference scripts +
DiffCSP/EMDiffuse); `download_models.py --model sccd`; `export SCCD_PHASE3=outputs/sccd/phase3_joint/best.pt`.
For the canonical batched 4-GPU run use the reference scripts in `stem2crystal/methods/_sccd_impl/`.

## Add your own
See [add_a_method.md](add_a_method.md) — implement `generate(image, composition, k)`, register, done.
The standalone evaluator scores it with every paper metric, exactly like the methods above.
