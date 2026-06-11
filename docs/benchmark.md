# STEM2Crystal-Bench

The benchmark is published on the Hugging Face Hub and is **not** committed to this repository.

🤗 **[gary23ai/STEM2Crystal-Bench](https://huggingface.co/datasets/gary23ai/STEM2Crystal-Bench)**

## What it contains

| Config | #samples | Description | Paper |
|---|---:|---|---|
| `synthetic` | 1021 (test) | Physics-inspired forward pipeline applied to curated 2D-slab prototypes from **C2DB + MC2D**; three controlled noise regimes (low / mid / high). Paired GT CIFs + STEM images + dense masks + split. | Table 1 |
| `real_stem_eval5` | 5 | Real, open-licensed atomic-resolution STEM images of **2D monolayer materials** (MoTe₂, WSe₂, WS₂, MoS₂, graphene), paired with **single-layer** ground-truth CIFs. | Table 2 |

Synthetic forward model: probe/PSF blur → dose-controlled Poisson counting → scan-line jitter →
low-frequency background → readout noise → intensity renormalization to `[0,1]`
(paper Appendix A.3). Images are 256×256.

## Download

```bash
python scripts/download_benchmark.py --config all          # everything -> ./data
python scripts/download_benchmark.py --config synthetic     # just Table 1
python scripts/download_benchmark.py --config real5         # just Table 2
```
Override the target dir with `--data-root <path>` or the `STEM2CRYSTAL_DATA` environment variable.
Programmatic access:
```python
from stem2crystal.data import download_benchmark, benchmark_paths
download_benchmark("synthetic")
paths = benchmark_paths("synthetic")     # {'gt_dir':..., 'images':{'low':...}, 'split':...}
```

You can also use 🤗 `datasets` directly:
```python
from datasets import load_dataset
synth = load_dataset("gary23ai/STEM2Crystal-Bench", "synthetic", split="test")
real  = load_dataset("gary23ai/STEM2Crystal-Bench", "real_stem_eval5", split="test")
```

## On-disk layout (after download)
```
data/
├── synthetic/
│   ├── cifs/            # 1021 ground-truth CIFs  (<sid>.cif)
│   ├── images/{low,mid,high}/   # <sid>_0.png  (256×256)
│   ├── masks/
│   ├── split.json       # {"test": [sid, ...]}
│   └── metadata.jsonl
└── real_stem_eval5/
    ├── cifs/            # 5 monolayer GT CIFs
    ├── images/          # 5 real STEM PNGs
    └── metadata.jsonl
```

## Prediction format (what methods write, what the evaluator reads)
One directory per sample, ranked candidate CIFs (best first):
```
predictions/<method>/<noise-or-"real">/<sid>_0/rank_1.cif … rank_K.cif
```

## Licensing
Synthetic GT CIFs: C2DB & MC2D are CC BY 4.0. Real-5 images: CC BY 4.0 (four TMD monolayers) and
CC BY 3.0 (graphene) — attribute the original deposits listed on the dataset card.
