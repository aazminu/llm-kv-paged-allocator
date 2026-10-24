# llm-kv-paged-allocator

A clean-room reference implementation of a **PagedAttention-style block-table KV-cache memory manager** with **chained prefix-hash radix sharing**, **zero-copy Copy-on-Write (CoW) sequence forking**, and **watermark-driven LRU eviction and preemption**.

Designed to model the core memory management primitives used in high-throughput LLM serving engines (vLLM, SGLang, TensorRT-LLM) where device HBM is partitioned into fixed-size token blocks mapped through per-request logical block tables.

## Key Capabilities

- **Non-Contiguous Logical-to-Physical Block Tables**: Eliminates external fragmentation and pre-allocation waste by slicing sequences into fixed-size physical `Block` descriptors (`block_size=16` tokens by default) indexed via per-request `SequenceTable` maps.
- **Chained Prefix-Hash & Radix Cache (`RadixPrefixCache`)**: Computes deterministic 64-bit FNV-1a or SHA-256 rolling block hashes chained over `(parent_prefix_hash, block_tokens)`. Multi-turn chat histories, shared system prompts, and few-shot templates automatically attach to existing physical KV blocks with $O(1)$ block lookup and reference counting.
- **Zero-Copy Sequence Forking & Lazy Copy-on-Write (`fork_sequence`)**: Enables parallel tool-use rollouts, beam search, and speculative decoding verification without duplicating physical KV blocks. Shared blocks with `ref_count > 1` are split lazily via `_cow_block` only when a branch mutates or appends into a partial block.
- **Convergent Block Deduplication**: When an active sequence finishes filling a partial block whose chained prefix hash matches an existing canonical block in the prefix cache, the allocator automatically deduplicates the newly filled block into the canonical block and reclaims the duplicate.
- **Watermark Eviction & Sequence Preemption (`enforce_watermark`)**: Maintains a configurable free-block watermark reserve. Under memory pressure, unreferenced cached prefix blocks are evicted in strict LRU order first, followed by swap-out preemption (`preempt_sequence` / `resume_sequence`) of least-recently-stepped active sequences.

## Architecture

```text
 Sequence "user-1" Logical Table       Physical KV Block Pool (HBM)
 +---------+---------+---------+       +--------------------------+
 | Blk #0  | Blk #1  | Blk #2  | ----> | Phys Block 0 (ref=2)     | <-- Shared System Prompt
 +---------+---------+---------+       | hash: fnv1a:8c3f...      |
      |         |                      +--------------------------+
      |         +--------------------> | Phys Block 1 (ref=2)     | <-- Shared Few-Shot Prefix
      |         |                      | hash: fnv1a:e41a...      |
 +---------+---------+---------+       +--------------------------+
 | Blk #0  | Blk #1  | Blk #3  | ----> | Phys Block 3 (ref=1)     | <-- Divergent Tail (CoW)
 +---------+---------+---------+       | tokens: [999, ...]       |
 Sequence "user-2" Logical Table       +--------------------------+
```

## Quickstart

```python
from kv_paged import PagedKVAllocator

# Initialize 64 physical KV blocks of 4 tokens each with a 15% watermark reserve
alloc = PagedKVAllocator(num_blocks=64, block_size=4, watermark_ratio=0.15)

# 1. Allocate two sequences sharing an 8-token system prompt prefix
system_prompt = [101, 102, 103, 104, 201, 202, 203, 204]
s1 = alloc.allocate_sequence("req-1", prompt_tokens=system_prompt + [11, 12])
s2 = alloc.allocate_sequence("req-2", prompt_tokens=system_prompt + [99])

assert s1.logical_blocks[:2] == s2.logical_blocks[:2]  # Zero-copy prefix cache hit

# 2. Fork a sequence for parallel beam / speculative decoding
beam_b = alloc.fork_sequence("req-1", "req-1-beam-b")
alloc.append_tokens("req-1", [13])        # Triggers lazy Copy-on-Write on tail block
alloc.append_tokens("req-1-beam-b", [77]) # Independent tail continuation

# 3. Inspect telemetry
stats = alloc.stats()
print(f"Active blocks: {stats.active_blocks}, Cache hits: {stats.prefix_cache_hits}, CoW splits: {stats.cow_splits}")
```

## Running Tests

```bash
pip install -e ".[dev]"
pytest -v
```

## Author

**Abdullah Azmi** — [abdullahazmi.com](https://abdullahazmi.com) · [@aazminu](https://github.com/aazminu)


## CLI Diagnostics & Invariant Verification

Run the built-in deterministic diagnostic probe and invariant self-check directly via the module CLI:

```bash
PYTHONPATH=src python -m kv_paged --self-test --json
```

Run the complete regression test suite:

```bash
PYTHONPATH=src pytest -q
```
