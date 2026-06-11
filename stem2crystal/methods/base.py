"""Pluggable method interface for STEM2Crystal-Bench.

To add your own model you implement a single method, ``generate``, and register it. The benchmark
harness then drives your model and the standalone evaluator scores it — you never touch metric code.

    from stem2crystal.methods import Method, register_method

    @register_method("my_model")
    class MyModel(Method):
        def setup(self):
            self.net = load_my_checkpoint(self.config["ckpt"])

        def generate(self, image, composition, k=5):
            # image: np.ndarray [H, W] in [0, 1];  composition: pymatgen Composition
            # return up to k pymatgen Structure candidates, best first
            return my_sampler(self.net, image, composition, k)

Run it:  ``python scripts/generate.py --method my_model --benchmark synthetic``
then:    ``python scripts/evaluate.py  --method my_model --benchmark synthetic``
"""
from __future__ import annotations
import abc
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
from pymatgen.core import Structure, Composition


class Method(abc.ABC):
    """Base class every benchmark method implements.

    The harness instantiates you with a ``config`` dict (from a YAML), calls :meth:`setup` once,
    then calls :meth:`generate` per sample and writes the returned candidates as
    ``rank_1.cif … rank_k.cif`` for the evaluator to score.
    """

    #: human-readable name; defaults to the registry key
    name: str = "method"

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.name = self.config.get("name", self.name)

    def setup(self) -> None:
        """Load checkpoints / build the model once. Override if needed."""

    @abc.abstractmethod
    def generate(self, image: np.ndarray, composition: Composition, k: int = 5) -> List[Structure]:
        """Reconstruct up to ``k`` candidate crystal structures, **best first**.

        Args:
            image: normalized STEM observation, ``np.ndarray`` of shape ``[H, W]`` in ``[0, 1]``.
            composition: target composition (element set / stoichiometry) as a pymatgen ``Composition``.
            k: number of candidates to return.
        Returns:
            list of up to ``k`` pymatgen ``Structure`` objects, ordered best-first.
        """
        raise NotImplementedError

    # -- convenience used by scripts/generate.py -------------------------------------
    def write_candidates(self, out_dir: str | Path, sid: str, cands: List[Structure]) -> None:
        d = Path(out_dir) / f"{sid}_0"
        d.mkdir(parents=True, exist_ok=True)
        for i, c in enumerate(cands, 1):
            try:
                c.to(filename=str(d / f"rank_{i}.cif"))
            except Exception:
                pass


# -------------------------------------------------------------------------------------
# Registry
# -------------------------------------------------------------------------------------
_REGISTRY: Dict[str, type] = {}

def register_method(key: str):
    """Class decorator that registers a :class:`Method` subclass under ``key``."""
    def deco(cls):
        if not issubclass(cls, Method):
            raise TypeError(f"{cls.__name__} must subclass Method")
        if key in _REGISTRY:
            raise KeyError(f"method '{key}' already registered")
        cls.name = key
        _REGISTRY[key] = cls
        return cls
    return deco

def get_method(key: str, config: Optional[Dict] = None) -> Method:
    if key not in _REGISTRY:
        raise KeyError(f"unknown method '{key}'. Registered: {sorted(_REGISTRY)}")
    return _REGISTRY[key](config)

def list_methods() -> List[str]:
    return sorted(_REGISTRY)
