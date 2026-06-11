"""Method registry for STEM2Crystal-Bench.

All methods are equal here — the benchmark's job is to score any reconstruction model uniformly.
The built-in methods (SCCD, DiffCSP, MicroscopyGPT, MatterGen) self-register on import; each wraps an
upstream model stack + downloaded weights (never committed — see ``scripts/download_models.py`` and
``docs/methods.md``). Heavy dependencies (torch, unsloth, mattergen, …) are imported lazily inside each
method's ``setup()``, so importing this package stays light and works even when a method's upstream
stack is absent — you only need it when you actually run that method.

Add your own model by subclassing :class:`Method` and decorating it with :func:`register_method`
(see ``examples/example_method.py`` and ``docs/add_a_method.md``).
"""
from .base import Method, register_method, get_method, list_methods

#: built-in method modules (each registers one method on import)
_BUILTINS = ("sccd", "diffcsp", "microscopygpt", "mattergen")


def _register_builtins():
    for name in _BUILTINS:
        try:
            __import__(f"stem2crystal.methods.{name}")
        except Exception:
            pass   # upstream deps absent — method simply won't be available until installed


_register_builtins()

__all__ = ["Method", "register_method", "get_method", "list_methods"]
