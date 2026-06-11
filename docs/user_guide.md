# User guide

End-to-end: install → get data → (train) → generate → evaluate. If you only want to **score existing
predictions** or **benchmark your own model**, skip training.

## 0. Install & download
```bash
pip install -e ".[model]"
python scripts/download_benchmark.py --config all      # -> ./data
```

## 1. Evaluate predictions (no model needed)
The evaluator is fully decoupled from models — it reads ground-truth CIFs and a predictions directory.

```bash
# explicit predictions dir
python scripts/evaluate.py --pred predictions/sccd/low --benchmark synthetic --name SCCD

# or the conventional layout predictions/<method>/<noise>
python scripts/evaluate.py --method sccd --benchmark synthetic --noise low
python scripts/evaluate.py --method sccd --benchmark real5            # Real-5 (Table 2)
```
It prints every paper metric and the failure-mode mix; add `--json out.json` to save.

Library use:
```python
from stem2crystal.data import benchmark_paths
from stem2crystal.eval import evaluate_predictions, BENCHMARK_MATCHERS
import json
p = benchmark_paths("synthetic")
ids = json.load(open(p["split"]))["test"]
m = evaluate_predictions(p["gt_dir"], "predictions/sccd/low", ids, **BENCHMARK_MATCHERS["synthetic"])
print(m["Hit5"], m["SG_Acc"], m["failure_modes"])
```

## 2. Generate predictions with a registered method
```bash
python scripts/generate.py --method sccd --benchmark synthetic --noise low
python scripts/generate.py --method example_random --method-module examples/example_method.py \
       --benchmark synthetic --limit 50
```
Writes `predictions/<method>/<noise>/<sid>_0/rank_{1..K}.cif`.

## 3. Train SCCD (3-phase curriculum)
SCCD is trained in three stages; each phase's script is under `stem2crystal/methods/_sccd_impl/`.
The reference scripts use 4-GPU `torchrun`.
The phase scripts are in `stem2crystal/methods/_sccd_impl/`.

| Phase | Script | What |
|---|---|---|
| I | `train_phase1_evidence.py` | evidence denoiser (EMDiffuse U-Net) |
| II | `train_diffcsp2d.py` | structure branch (DiffCSP-style, composition-only warm start) |
| III | `train_sccd_v5_ddp.py` | joint bidirectional co-diffusion (= paper SCCD) |

```bash
# Phase III (canonical), 4-GPU
torchrun --nproc_per_node=4 stem2crystal/methods/_sccd_impl/train_sccd_v5_ddp.py --epochs 60 --bs 16 --out outputs/sccd
```
A FiLM-coupling variant (`train_sccd_v5_film.py`) is provided and is empirically equivalent.

## 4. Benchmark your own model
1. Implement a `Method` (see [add_a_method.md](add_a_method.md)).
2. `python scripts/generate.py --method <name> --method-module <file> --benchmark synthetic`
3. `python scripts/evaluate.py --method <name> --benchmark synthetic` for each noise level.

## Tips
- Synthetic test set = 1021 structures × K candidates × 3 noise levels; generation is the slow part
  (diffusion sampling). Use `--limit` for quick smoke tests.
- `STEM2CRYSTAL_DATA=/big/disk` to keep data off the repo volume.
- Metrics use a wall-clock guard around `StructureMatcher` (it can hang on degenerate cells); results
  are deterministic given fixed predictions.
