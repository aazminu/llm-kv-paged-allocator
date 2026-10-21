"""Deterministic runtime diagnostics and invariant verification for `kv_paged`."""

from __future__ import annotations

import time
from typing import Any

from .allocator import PagedKVAllocator, compute_block_prefix_hash

def _domain_probe(seed: int) -> dict:
    alloc = PagedKVAllocator(num_blocks=16, block_size=4)
    seq = alloc.allocate_sequence("req-1", prompt_tokens=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    h1 = compute_block_prefix_hash([1, 2, 3, 4], parent_hash=None, algo="fnv1a")
    return {
        "num_tokens": seq.num_tokens,
        "logical_blocks": len(seq.logical_blocks),
        "prefix_hash_prefix": h1.split(":")[0],
        "invariants_ok": (seq.num_tokens == 10 and len(seq.logical_blocks) == 3 and h1.startswith("fnv1a:")),
    }


def run_diagnostic_suite(seed: int = 42) -> dict[str, Any]:
    """Execute deterministic domain probe for `llm-kv-paged-allocator` and return structured metrics."""
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    t0 = time.perf_counter()
    metrics = _domain_probe(seed)
    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 3)
    return {
        "repository": "llm-kv-paged-allocator",
        "package": "kv_paged",
        "seed": seed,
        "elapsed_ms": elapsed_ms,
        "metrics": metrics,
        "status": "ok" if metrics.get("invariants_ok", False) else "failed",
    }


def validate_runtime_invariants(seed: int = 42) -> dict[str, Any]:
    """Validate deterministic reproducibility and domain invariants for `kv_paged`."""
    run_a = run_diagnostic_suite(seed=seed)
    run_b = run_diagnostic_suite(seed=seed)
    reproducible = run_a["metrics"] == run_b["metrics"]
    passed = bool(run_a["metrics"].get("invariants_ok", False)) and reproducible
    return {
        "repository": "llm-kv-paged-allocator",
        "package": "kv_paged",
        "seed": seed,
        "reproducible": reproducible,
        "passed": passed,
        "metrics": run_a["metrics"],
    }
