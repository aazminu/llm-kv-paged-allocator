"""PagedAttention-style KV-cache block allocator with prefix caching and Copy-on-Write."""

from .allocator import (
    AllocationError,
    AllocatorStats,
    Block,
    OutOfMemoryError,
    PagedKVAllocator,
    RadixPrefixCache,
    SequenceNotFoundError,
    SequenceStatus,
    SequenceTable,
    compute_block_prefix_hash,
)

__all__ = [
    "AllocationError",
    "AllocatorStats",
    "Block",
    "OutOfMemoryError",
    "PagedKVAllocator",
    "RadixPrefixCache",
    "SequenceNotFoundError",
    "SequenceStatus",
    "SequenceTable",
    "compute_block_prefix_hash",
]

__version__ = "0.1.0"

from .diagnostics import run_diagnostic_suite, validate_runtime_invariants
__all__ = list(__all__) + ["run_diagnostic_suite", "validate_runtime_invariants"]
