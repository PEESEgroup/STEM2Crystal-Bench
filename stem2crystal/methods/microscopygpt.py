"""MicroscopyGPT baseline — a fine-tuned vision LLM that reads the STEM image + a composition prompt
and emits a crystal structure as text. Fits the per-image Method interface directly.

Upstream / weights (NOT in this repo — fetch with `scripts/download_models.py --model microscopygpt`):
    base model : unsloth/llama-3.2-11b-vision-instruct-unsloth-bnb-4bit  (auto-downloaded by unsloth)
    LoRA       : the MicroscopyGPT adapter, placed at $MICROSCOPYGPT_ADAPTER
Requires:  pip install unsloth ; a CUDA GPU with ~12 GB.
"""
from __future__ import annotations
import os
import re
from typing import List
import numpy as np
from PIL import Image
from pymatgen.core import Structure, Composition, Lattice

from .base import Method, register_method


def _text_to_structure(text: str):
    """Parse the model's text output (lattice line + 'Elem x y z' lines) into a Structure."""
    nums = re.findall(r"[-+]?\d*\.\d+|\d+", text)
    try:
        # first 6 numbers = a b c alpha beta gamma; then repeating (Z?, x, y, z) — kept permissive
        a, b, c, al, be, ga = (float(x) for x in nums[:6])
        lat = Lattice.from_parameters(a, b, c, al, be, ga)
        sp, fr = [], []
        for line in text.splitlines():
            m = re.match(r"\s*([A-Z][a-z]?)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)", line)
            if m:
                sp.append(m.group(1)); fr.append([float(m.group(2)), float(m.group(3)), float(m.group(4))])
        return Structure(lat, sp, fr) if sp else None
    except Exception:
        return None


@register_method("microscopygpt")
class MicroscopyGPT(Method):
    """Vision-LLM baseline (Choudhary et al.). Image-conditioned, composition-prompted."""

    def setup(self) -> None:
        try:
            from unsloth import FastVisionModel
        except Exception as e:
            raise ImportError("MicroscopyGPT needs `pip install unsloth` and a CUDA GPU. "
                              "Fetch weights with scripts/download_models.py --model microscopygpt.") from e
        adapter = self.config.get("adapter") or os.environ.get("MICROSCOPYGPT_ADAPTER")
        if not adapter or not os.path.exists(adapter):
            raise FileNotFoundError("Set $MICROSCOPYGPT_ADAPTER to the downloaded LoRA adapter dir "
                                    "(scripts/download_models.py --model microscopygpt).")
        self.model, self.tokenizer = FastVisionModel.from_pretrained(adapter, load_in_4bit=True)
        FastVisionModel.for_inference(self.model)
        self.max_new_tokens = self.config.get("max_new_tokens", 1554)

    def generate(self, image: np.ndarray, composition: Composition, k: int = 5) -> List[Structure]:
        pil = Image.fromarray((np.clip(image, 0, 1) * 255).astype("uint8")).convert("RGB").resize((256, 256))
        prompt = (f"Reconstruct the crystal structure with composition "
                  f"{composition.reduced_formula} from this STEM image.")
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        input_text = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        out = []
        for i in range(k):                              # sample k times (greedy + sampled variants)
            inputs = self.tokenizer(pil, input_text, add_special_tokens=False, return_tensors="pt").to("cuda")
            gen = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens,
                                      do_sample=(i > 0), temperature=0.7 if i > 0 else None)
            txt = self.tokenizer.decode(gen[0], skip_special_tokens=True)
            s = _text_to_structure(txt)
            if s is not None:
                out.append(s)
        return out
