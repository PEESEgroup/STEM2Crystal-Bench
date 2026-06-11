"""Standalone, model-agnostic evaluation for STEM2Crystal-Bench (all paper metrics)."""
# Keep BLAS single-threaded: evaluation parallelizes over samples with a process Pool, so
# multi-threaded BLAS inside each worker would oversubscribe cores and inflate memory. Set this
# before numpy is first imported (below) so it takes effect. Override by exporting the vars yourself.
import os as _os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")

from .metrics import (scale_matcher, abs_matcher, hit_at_k, min_rms_at_k, match_abs,
                      rmsd_angstrom, sg_accuracy, wasserstein_lattice, failure_mode,
                      score_sample, SampleResult)
from .evaluate import (evaluate_predictions, format_table, load_candidates,
                       load_structure, BENCHMARK_MATCHERS)

__all__ = [
    "evaluate_predictions", "format_table", "BENCHMARK_MATCHERS",
    "load_candidates", "load_structure",
    "scale_matcher", "abs_matcher", "hit_at_k", "min_rms_at_k", "match_abs",
    "rmsd_angstrom", "sg_accuracy", "wasserstein_lattice", "failure_mode",
    "score_sample", "SampleResult",
]
