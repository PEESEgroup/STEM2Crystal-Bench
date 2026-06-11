# Add a method

The harness is built so you can plug in any reconstruction model and have it scored with **every paper
metric**, without touching evaluation code. You implement one method, `generate`, and register it.

## The contract
```python
from stem2crystal.methods import Method, register_method

@register_method("my_model")           # unique key used on the CLI
class MyModel(Method):

    def setup(self) -> None:
        # called once. Load checkpoints / build the model. self.config is your YAML/JSON dict.
        self.net = load_my_checkpoint(self.config["ckpt"])

    def generate(self, image, composition, k=5):
        """
        image:        np.ndarray [H, W] float in [0, 1]   (the STEM observation)
        composition:  pymatgen.core.Composition           (target stoichiometry, given)
        returns:      list of up to k pymatgen Structure   (best first)
        """
        return my_sampler(self.net, image, composition, k)
```
That's the whole interface (`stem2crystal/methods/base.py`). A complete, runnable example that needs no
GPU is in [`examples/example_method.py`](../examples/example_method.py).

## Run it
```bash
# if your file lives outside the package, point --method-module at it
python scripts/generate.py --method my_model --method-module path/to/my_model.py \
       --benchmark synthetic --noise low
python scripts/evaluate.py  --method my_model --benchmark synthetic --noise low
```
To register permanently, drop the module under `stem2crystal/methods/` and import it from
`stem2crystal/methods/__init__.py` (next to `sccd_method`).

## Notes
- **Composition is given** — you do *not* predict the element set; focus model capacity on geometry.
- Return candidates **best-first**; Hit@1 / SG-Acc use rank-1, Hit@5 / MinRMS@K use the top-K.
- Output is plain CIF, so any framework works (PyTorch, JAX, a retrieval system, an LLM …). The harness
  writes `rank_{1..K}.cif`; the evaluator reads them. Nothing else is assumed about your model.
- For non-diffusion / non-image methods (e.g. composition-only baselines) just ignore `image`.
