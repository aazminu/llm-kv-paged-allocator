"""PagedAttention-style KV-cache block allocator with prefix caching and Copy-on-Write.

Implements non-contiguous physical KV block management, logical-to-physical sequence block
tables, reference counting, Copy-on-Write (CoW) sequence forking for beam search and
speculative decoding, rolling FNV-1a / SHA-256 radix prefix sharing, and watermark-driven
LRU eviction and sequence preemption.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import math
from typing import Iterable, Literal, Optional, Sequence


class AllocationError(Exception):
    """Base exception for KV block allocator failures."""


class OutOfMemoryError(AllocationError):
    """Raised when physical KV blocks are exhausted and cannot be reclaimed."""


class SequenceNotFoundError(AllocationError):
    """Raised when an operation references an unknown sequence ID."""


class SequenceStatus(str, Enum):
    """Lifecycle state of a sequence managed by the allocator."""

    RUNNING = "running"
    SWAPPED = "swapped"
    FINISHED = "finished"


HashAlgorithm = Literal["fnv1a", "sha256"]

_FNV64_OFFSET_BASIS = 0xCBF29CE484222325
_FNV64_PRIME = 0x100000001B3
_UINT64_MASK = 0xFFFFFFFFFFFFFFFF


def compute_block_prefix_hash(
    tokens: Sequence[int],
    parent_hash: Optional[str] = None,
    algo: HashAlgorithm = "fnv1a",
) -> str:
    """Compute a chained prefix hash over a full block's tokens and its parent block hash.

    Chaining `parent_hash` guarantees that two blocks with identical local tokens only
    share the same prefix hash if their entire preceding token prefix is also identical.
    """
    if algo == "fnv1a":
        h = _FNV64_OFFSET_BASIS
        if parent_hash is not None:
            for b in parent_hash.encode("ascii"):
                h ^= b
                h = (h * _FNV64_PRIME) & _UINT64_MASK
        for tok in tokens:
            # Mix 64-bit signed/unsigned token integer in 8 little-endian bytes
            val = tok & _UINT64_MASK
            for shift in range(0, 64, 8):
                byte_val = (val >> shift) & 0xFF
                h ^= byte_val
                h = (h * _FNV64_PRIME) & _UINT64_MASK
        return f"fnv1a:{h:016x}"
    elif algo == "sha256":
        hasher = hashlib.sha256()
        if parent_hash is not None:
            hasher.update(parent_hash.encode("ascii"))
            hasher.update(b"|")
        for tok in tokens:
            hasher.update(int(tok).to_bytes(8, byteorder="little", signed=True))
        return f"sha256:{hasher.hexdigest()[:32]}"
    else:
        raise ValueError(f"Unsupported hash algorithm: {algo!r}")


@dataclass
class Block:
    """Represents a fixed-size physical KV-cache block in device memory."""

    block_id: int
    block_size: int
    tokens: list[int] = field(default_factory=list)
    ref_count: int = 0
    prefix_hash: Optional[str] = None
    parent_prefix_hash: Optional[str] = None
    last_accessed: int = 0
    pinned: bool = False

    @property
    def is_full(self) -> bool:
        """Return True if the physical block has reached its token capacity."""
        return len(self.tokens) == self.block_size

    @property
    def free_slots(self) -> int:
        """Return the number of remaining token slots in this block."""
        return self.block_size - len(self.tokens)

    def reset(self) -> None:
        """Clear token state and metadata before fresh allocation."""
        self.tokens.clear()
        self.ref_count = 0
        self.prefix_hash = None
        self.parent_prefix_hash = None
        self.pinned = False


@dataclass
class SequenceTable:
    """Logical-to-physical block table for a single request/sequence."""

    seq_id: str
    logical_blocks: list[int] = field(default_factory=list)
    num_tokens: int = 0
    status: SequenceStatus = SequenceStatus.RUNNING
    swapped_tokens: list[int] = field(default_factory=list)
    parent_seq_id: Optional[str] = None
    last_step: int = 0


@dataclass(frozen=True)
class AllocatorStats:
    """Point-in-time telemetry snapshot of the PagedKVAllocator."""

    total_blocks: int
    block_size: int
    active_blocks: int
    cached_evictable_blocks: int
    free_blocks: int
    active_sequences: int
    swapped_sequences: int
    prefix_cache_hits: int
    prefix_cache_misses: int
    cow_splits: int
    evictions: int
    preemptions: int
    internal_fragmentation_ratio: float


@dataclass
class _RadixNode:
    """Internal node in the block-granularity RadixPrefixCache."""

    token_tuple: tuple[int, ...]
    prefix_hash: str
    block_id: Optional[int] = None
    children: dict[tuple[int, ...], _RadixNode] = field(default_factory=dict)


class RadixPrefixCache:
    """Radix tree + prefix-hash index over full KV blocks for prefix reuse."""

    def __init__(self, block_size: int, hash_algo: HashAlgorithm = "fnv1a") -> None:
        self.block_size = block_size
        self.hash_algo: HashAlgorithm = hash_algo
        self._root = _RadixNode(token_tuple=(), prefix_hash="root")
        self._hash_to_block: dict[str, int] = {}
        self._block_to_hash: dict[int, str] = {}

    def lookup_by_hash(self, prefix_hash: str) -> Optional[int]:
        """Return the physical block_id associated with a chained prefix hash, if present."""
        return self._hash_to_block.get(prefix_hash)

    def insert_block(
        self,
        block_id: int,
        block_tokens: Sequence[int],
        parent_hash: Optional[str],
        prefix_hash: str,
    ) -> None:
        """Index a completed physical block by its prefix hash."""
        if len(block_tokens) != self.block_size:
            raise ValueError("Only full blocks can be inserted into the prefix cache.")
        self._hash_to_block[prefix_hash] = block_id
        self._block_to_hash[block_id] = prefix_hash

    def insert_sequence_path(
        self,
        block_ids: Sequence[int],
        blocks: Sequence[Block],
    ) -> None:
        """Insert a contiguous prefix chain of full blocks into both the radix trie and hash map."""
        node = self._root
        for blk_id in block_ids:
            blk = blocks[blk_id]
            if not blk.is_full or blk.prefix_hash is None:
                break
            key = tuple(blk.tokens)
            child = node.children.get(key)
            if child is None:
                child = _RadixNode(token_tuple=key, prefix_hash=blk.prefix_hash, block_id=blk_id)
                node.children[key] = child
            else:
                child.block_id = blk_id
                child.prefix_hash = blk.prefix_hash
            self._hash_to_block[blk.prefix_hash] = blk_id
            self._block_to_hash[blk_id] = blk.prefix_hash
            node = child

    def invalidate_block(self, block_id: int) -> None:
        """Remove a physical block from the prefix index when it is overwritten or evicted."""
        prefix_hash = self._block_to_hash.pop(block_id, None)
        if prefix_hash is not None and self._hash_to_block.get(prefix_hash) == block_id:
            del self._hash_to_block[prefix_hash]

    def match_prefix_blocks(
        self,
        tokens: Sequence[int],
        blocks: Sequence[Block],
    ) -> tuple[list[int], Optional[str]]:
        """Find the longest prefix of full blocks matching `tokens`.

        Returns:
            Tuple of `(matched_block_ids, last_prefix_hash)`.
        """
        num_full_blocks = len(tokens) // self.block_size
        matched_ids: list[int] = []
        parent_hash: Optional[str] = None

        for i in range(num_full_blocks):
            chunk = tokens[i * self.block_size : (i + 1) * self.block_size]
            candidate_hash = compute_block_prefix_hash(chunk, parent_hash, self.hash_algo)
            blk_id = self._hash_to_block.get(candidate_hash)
            if blk_id is None:
                break
            blk = blocks[blk_id]
            if not blk.is_full or blk.prefix_hash != candidate_hash or list(chunk) != blk.tokens:
                # Defensive guard against hash collisions
                break
            matched_ids.append(blk_id)
            parent_hash = candidate_hash

        return matched_ids, parent_hash

    def __len__(self) -> int:
        return len(self._hash_to_block)


class PagedKVAllocator:
    """PagedAttention-style KV-cache block table manager.

    Features:
    - Fixed-size physical KV blocks mapped to per-sequence logical block tables.
    - Chained prefix-hash radix caching across active and recently finished sequences.
    - Zero-copy `fork_sequence` with lazy Copy-on-Write (CoW) when appending or mutating shared blocks.
    - Watermark-based LRU cache eviction and sequence swap-out preemption.
    """

    def __init__(
        self,
        num_blocks: int,
        block_size: int = 16,
        watermark_ratio: float = 0.10,
        hash_algo: HashAlgorithm = "fnv1a",
    ) -> None:
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive")
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        if not (0.0 <= watermark_ratio < 1.0):
            raise ValueError("watermark_ratio must be in [0.0, 1.0)")

        self.num_blocks = num_blocks
        self.block_size = block_size
        self.watermark_ratio = watermark_ratio
        self.watermark_blocks = math.ceil(num_blocks * watermark_ratio)
        self.hash_algo: HashAlgorithm = hash_algo

        self._clock: int = 0
        self._blocks: list[Block] = [
            Block(block_id=i, block_size=block_size) for i in range(num_blocks)
        ]
        # Free uninitialized/reclaimed blocks (FIFO/LIFO order)
        self._free_list: OrderedDict[int, None] = OrderedDict(
            (i, None) for i in range(num_blocks)
        )
        # Evictable cached blocks (ref_count == 0, full blocks retained in prefix cache, LRU order)
        self._evictable_lru: OrderedDict[int, None] = OrderedDict()

        self.prefix_cache = RadixPrefixCache(block_size=block_size, hash_algo=hash_algo)
        self._sequences: dict[str, SequenceTable] = {}

        # Telemetry counters
        self.prefix_cache_hits: int = 0
        self.prefix_cache_misses: int = 0
        self.cow_splits: int = 0
        self.evictions: int = 0
        self.preemptions: int = 0

    def _tick(self) -> int:
        self._clock += 1
        return self._clock

    @property
    def free_block_count(self) -> int:
        """Number of immediately free (uncached) physical blocks."""
        return len(self._free_list)

    @property
    def evictable_block_count(self) -> int:
        """Number of unreferenced cached blocks that can be evicted on demand."""
        return len(self._evictable_lru)

    @property
    def available_block_count(self) -> int:
        """Total blocks available for new allocations (free + evictable cached blocks)."""
        return len(self._free_list) + len(self._evictable_lru)

    @property
    def active_block_count(self) -> int:
        """Number of physical blocks currently referenced by at least one active sequence."""
        return self.num_blocks - self.available_block_count

    def is_below_watermark(self) -> bool:
        """Return True if available blocks have dropped below the configured watermark."""
        return self.available_block_count < self.watermark_blocks

    def get_block(self, block_id: int) -> Block:
        """Return the physical Block descriptor for `block_id`."""
        if not (0 <= block_id < self.num_blocks):
            raise IndexError(f"block_id {block_id} out of range [0, {self.num_blocks})")
        return self._blocks[block_id]

    def get_sequence(self, seq_id: str) -> SequenceTable:
        """Return the SequenceTable for `seq_id` or raise SequenceNotFoundError."""
        seq = self._sequences.get(seq_id)
        if seq is None:
            raise SequenceNotFoundError(f"Sequence {seq_id!r} not found")
        return seq

    def get_sequence_tokens(self, seq_id: str) -> list[int]:
        """Reconstruct the full ordered token list for an active or swapped sequence."""
        seq = self.get_sequence(seq_id)
        if seq.status == SequenceStatus.SWAPPED:
            return list(seq.swapped_tokens)
        tokens: list[int] = []
        for blk_id in seq.logical_blocks:
            tokens.extend(self._blocks[blk_id].tokens)
        return tokens

    def _acquire_raw_block(self) -> Block:
        """Allocate a single physical block from the free list or by evicting the oldest cached block."""
        now = self._tick()
        if self._free_list:
            blk_id, _ = self._free_list.popitem(last=False)
            blk = self._blocks[blk_id]
            blk.reset()
            blk.ref_count = 1
            blk.last_accessed = now
            return blk

        # Evict the least-recently-used unreferenced block from the prefix cache
        for candidate_id in list(self._evictable_lru.keys()):
            candidate = self._blocks[candidate_id]
            if not candidate.pinned:
                del self._evictable_lru[candidate_id]
                self.prefix_cache.invalidate_block(candidate_id)
                candidate.reset()
                candidate.ref_count = 1
                candidate.last_accessed = now
                self.evictions += 1
                return candidate

        raise OutOfMemoryError(
            f"KV cache exhausted: 0 free blocks and 0 unpinned evictable blocks out of {self.num_blocks}."
        )

    def _increment_ref(self, block_id: int) -> Block:
        """Increment reference count on a physical block, rescuing it from the LRU if needed."""
        blk = self._blocks[block_id]
        if blk.ref_count == 0:
            self._evictable_lru.pop(block_id, None)
            self._free_list.pop(block_id, None)
        blk.ref_count += 1
        blk.last_accessed = self._tick()
        return blk

    def _decrement_ref(self, block_id: int, keep_in_cache: bool = True) -> None:
        """Decrement reference count on a physical block and transition it if ref_count hits 0."""
        blk = self._blocks[block_id]
        if blk.ref_count <= 0:
            raise AllocationError(f"Block {block_id} already has ref_count={blk.ref_count}")
        blk.ref_count -= 1
        blk.last_accessed = self._tick()
        if blk.ref_count == 0:
            if keep_in_cache and blk.is_full and blk.prefix_hash is not None:
                self._evictable_lru[block_id] = None
                self._evictable_lru.move_to_end(block_id, last=True)
            else:
                self.prefix_cache.invalidate_block(block_id)
                blk.reset()
                self._free_list[block_id] = None

    def _finalize_full_block(
        self,
        blk: Block,
        parent_hash: Optional[str],
        register_in_cache: bool = True,
    ) -> Optional[Block]:
        """Compute prefix hash for a newly filled block and deduplicate if an identical block exists."""
        if not blk.is_full:
            return None

        prefix_hash = compute_block_prefix_hash(blk.tokens, parent_hash, self.hash_algo)
        blk.parent_prefix_hash = parent_hash
        blk.prefix_hash = prefix_hash

        if not register_in_cache:
            return blk

        existing_id = self.prefix_cache.lookup_by_hash(prefix_hash)
        if existing_id is not None and existing_id != blk.block_id:
            existing_blk = self._blocks[existing_id]
            if existing_blk.is_full and existing_blk.tokens == blk.tokens:
                # Deduplicate into existing canonical cached block
                self._increment_ref(existing_id)
                self._decrement_ref(blk.block_id, keep_in_cache=False)
                return existing_blk

        self.prefix_cache.insert_block(
            block_id=blk.block_id,
            block_tokens=blk.tokens,
            parent_hash=parent_hash,
            prefix_hash=prefix_hash,
        )
        return blk

    def allocate_sequence(
        self,
        seq_id: str,
        prompt_tokens: Sequence[int] = (),
        use_prefix_cache: bool = True,
    ) -> SequenceTable:
        """Create a new sequence table and populate its prompt tokens.

        When `use_prefix_cache` is True, full prompt blocks matching cached prefixes
        are attached via zero-copy reference counting before allocating remaining blocks.
        """
        if seq_id in self._sequences and self._sequences[seq_id].status != SequenceStatus.FINISHED:
            raise AllocationError(f"Sequence {seq_id!r} already exists")

        total_blocks_needed = math.ceil(len(prompt_tokens) / self.block_size) if prompt_tokens else 0
        matched_ids: list[int] = []
        if use_prefix_cache and prompt_tokens:
            matched_ids, _ = self.prefix_cache.match_prefix_blocks(prompt_tokens, self._blocks)

        num_full_blocks = len(prompt_tokens) // self.block_size
        unmatched_full_blocks = num_full_blocks - len(matched_ids)
        new_blocks_needed = total_blocks_needed - len(matched_ids)

        # Count how many matched blocks are currently in _evictable_lru (reclaiming them reduces available_block_count)
        evictable_reclaimed = sum(1 for bid in matched_ids if bid in self._evictable_lru)
        if new_blocks_needed + evictable_reclaimed > self.available_block_count:
            raise OutOfMemoryError(
                f"Cannot allocate sequence {seq_id!r}: requires {new_blocks_needed} new blocks "
                f"(+{evictable_reclaimed} reclaimed cache blocks), only {self.available_block_count} available."
            )

        self.prefix_cache_hits += len(matched_ids)
        self.prefix_cache_misses += max(0, unmatched_full_blocks)

        seq = SequenceTable(
            seq_id=seq_id,
            logical_blocks=[],
            num_tokens=0,
            status=SequenceStatus.RUNNING,
            last_step=self._tick(),
        )

        for bid in matched_ids:
            self._increment_ref(bid)
            seq.logical_blocks.append(bid)
            seq.num_tokens += self.block_size

        self._sequences[seq_id] = seq

        remaining_tokens = prompt_tokens[len(matched_ids) * self.block_size :]
        if remaining_tokens:
            self.append_tokens(seq_id, remaining_tokens, register_in_cache=use_prefix_cache)

        if use_prefix_cache and seq.logical_blocks:
            self.prefix_cache.insert_sequence_path(seq.logical_blocks, self._blocks)

        return seq

    def _get_parent_prefix_hash(self, seq: SequenceTable, logical_idx: int) -> Optional[str]:
        """Return the prefix hash of `logical_idx - 1` in `seq`, or None if `logical_idx == 0`."""
        if logical_idx <= 0:
            return None
        prev_blk_id = seq.logical_blocks[logical_idx - 1]
        return self._blocks[prev_blk_id].prefix_hash

    def _cow_block(self, seq: SequenceTable, logical_idx: int) -> Block:
        """Perform a Copy-on-Write split on `seq.logical_blocks[logical_idx]` if shared."""
        blk_id = seq.logical_blocks[logical_idx]
        blk = self._blocks[blk_id]
        if blk.ref_count <= 1:
            return blk

        new_blk = self._acquire_raw_block()
        new_blk.tokens = list(blk.tokens)
        new_blk.parent_prefix_hash = blk.parent_prefix_hash
        new_blk.prefix_hash = None  # Being mutated, no longer matches old full-block hash
        self._decrement_ref(blk_id, keep_in_cache=True)
        seq.logical_blocks[logical_idx] = new_blk.block_id
        self.cow_splits += 1
        return new_blk

    def append_tokens(
        self,
        seq_id: str,
        tokens: Sequence[int],
        register_in_cache: bool = True,
    ) -> list[int]:
        """Append generated or prompt continuation tokens to an active sequence.

        Automatically handles:
        - Copy-on-Write if the tail physical block is shared (`ref_count > 1`).
        - Allocating new physical blocks across block boundaries.
        - Computing chained prefix hashes and deduplicating full blocks into the prefix cache.
        """
        seq = self.get_sequence(seq_id)
        if seq.status != SequenceStatus.RUNNING:
            raise AllocationError(
                f"Cannot append tokens to sequence {seq_id!r} with status {seq.status.value!r}"
            )
        if not tokens:
            return list(seq.logical_blocks)

        seq.last_step = self._tick()
        idx = 0
        n = len(tokens)

        # Check if we need a CoW split + how many new blocks we will need up front
        tail_free = 0
        need_cow_block = 0
        if seq.logical_blocks:
            tail_blk = self._blocks[seq.logical_blocks[-1]]
            tail_free = tail_blk.free_slots
            if tail_free > 0 and tail_blk.ref_count > 1:
                need_cow_block = 1

        overflow_tokens = max(0, n - tail_free)
        extra_blocks = math.ceil(overflow_tokens / self.block_size) if overflow_tokens else 0
        if need_cow_block + extra_blocks > self.available_block_count:
            raise OutOfMemoryError(
                f"Insufficient KV blocks to append {n} tokens to {seq_id!r}: "
                f"need {need_cow_block + extra_blocks}, available {self.available_block_count}."
            )

        while idx < n:
            if not seq.logical_blocks or self._blocks[seq.logical_blocks[-1]].is_full:
                new_blk = self._acquire_raw_block()
                seq.logical_blocks.append(new_blk.block_id)

            logical_idx = len(seq.logical_blocks) - 1
            tail_blk = self._cow_block(seq, logical_idx)
            take = min(tail_blk.free_slots, n - idx)
            tail_blk.tokens.extend(tokens[idx : idx + take])
            tail_blk.last_accessed = self._tick()
            seq.num_tokens += take
            idx += take

            if tail_blk.is_full:
                parent_hash = self._get_parent_prefix_hash(seq, logical_idx)
                canonical = self._finalize_full_block(
                    tail_blk, parent_hash=parent_hash, register_in_cache=register_in_cache
                )
                if canonical is not None:
                    seq.logical_blocks[logical_idx] = canonical.block_id

        if register_in_cache and seq.logical_blocks:
            self.prefix_cache.insert_sequence_path(seq.logical_blocks, self._blocks)

        return list(seq.logical_blocks)

    def overwrite_token(self, seq_id: str, token_index: int, new_token: int) -> int:
        """Mutate a single token at `token_index` in `seq_id`, triggering CoW if shared.

        Returns the physical block_id holding the modified token after any CoW split.
        """
        seq = self.get_sequence(seq_id)
        if seq.status != SequenceStatus.RUNNING:
            raise AllocationError(f"Sequence {seq_id!r} is not RUNNING")
        if not (0 <= token_index < seq.num_tokens):
            raise IndexError(f"token_index {token_index} out of bounds [0, {seq.num_tokens})")

        logical_idx = token_index // self.block_size
        offset = token_index % self.block_size
        blk = self._cow_block(seq, logical_idx)
        # If this block was previously indexed in the prefix cache and exclusively owned, invalidate old hash
        if blk.prefix_hash is not None:
            self.prefix_cache.invalidate_block(blk.block_id)
            blk.prefix_hash = None

        blk.tokens[offset] = new_token
        blk.last_accessed = self._tick()

        # Re-hash chain from logical_idx onward for any full blocks
        for i in range(logical_idx, len(seq.logical_blocks)):
            cur_blk = self._cow_block(seq, i)
            if cur_blk.prefix_hash is not None:
                self.prefix_cache.invalidate_block(cur_blk.block_id)
                cur_blk.prefix_hash = None
            if cur_blk.is_full:
                parent_hash = self._get_parent_prefix_hash(seq, i)
                canonical = self._finalize_full_block(
                    cur_blk, parent_hash=parent_hash, register_in_cache=True
                )
                if canonical is not None:
                    seq.logical_blocks[i] = canonical.block_id

        return seq.logical_blocks[logical_idx]

    def fork_sequence(self, parent_seq_id: str, child_seq_id: str) -> SequenceTable:
        """Zero-copy fork of `parent_seq_id` into `child_seq_id` for beam search or speculative decoding.

        All logical blocks in `parent_seq_id` are shared with `child_seq_id` and their physical
        reference counts are incremented. Physical blocks are only duplicated lazily when either
        branch mutates or appends into a shared partial block.
        """
        parent = self.get_sequence(parent_seq_id)
        if parent.status != SequenceStatus.RUNNING:
            raise AllocationError(f"Cannot fork non-running sequence {parent_seq_id!r}")
        if child_seq_id in self._sequences and self._sequences[child_seq_id].status != SequenceStatus.FINISHED:
            raise AllocationError(f"Child sequence {child_seq_id!r} already exists")

        for blk_id in parent.logical_blocks:
            self._increment_ref(blk_id)

        child = SequenceTable(
            seq_id=child_seq_id,
            logical_blocks=list(parent.logical_blocks),
            num_tokens=parent.num_tokens,
            status=SequenceStatus.RUNNING,
            parent_seq_id=parent_seq_id,
            last_step=self._tick(),
        )
        self._sequences[child_seq_id] = child
        return child

    def free_sequence(
        self,
        seq_id: str,
        keep_in_prefix_cache: bool = True,
        remove_record: bool = True,
    ) -> None:
        """Release all physical blocks held by `seq_id`."""
        seq = self.get_sequence(seq_id)
        for blk_id in seq.logical_blocks:
            self._decrement_ref(blk_id, keep_in_cache=keep_in_prefix_cache)
        seq.logical_blocks.clear()
        seq.status = SequenceStatus.FINISHED
        if remove_record:
            del self._sequences[seq_id]

    def preempt_sequence(self, seq_id: str) -> list[int]:
        """Swap out an active sequence's KV blocks to host token storage to relieve memory pressure."""
        seq = self.get_sequence(seq_id)
        if seq.status != SequenceStatus.RUNNING:
            raise AllocationError(f"Cannot preempt sequence {seq_id!r} in status {seq.status.value!r}")

        seq.swapped_tokens = self.get_sequence_tokens(seq_id)
        for blk_id in seq.logical_blocks:
            # Release physical blocks without keeping unshared ones occupying evictable cache if we are under pressure
            self._decrement_ref(blk_id, keep_in_cache=True)
        seq.logical_blocks.clear()
        seq.status = SequenceStatus.SWAPPED
        seq.last_step = self._tick()
        self.preemptions += 1
        return list(seq.swapped_tokens)

    def resume_sequence(self, seq_id: str, use_prefix_cache: bool = True) -> SequenceTable:
        """Restore a previously preempted (`SWAPPED`) sequence back into physical KV blocks."""
        seq = self.get_sequence(seq_id)
        if seq.status != SequenceStatus.SWAPPED:
            raise AllocationError(f"Sequence {seq_id!r} is not SWAPPED (status={seq.status.value!r})")

        saved_tokens = list(seq.swapped_tokens)
        parent_id = seq.parent_seq_id
        del self._sequences[seq_id]

        restored = self.allocate_sequence(
            seq_id=seq_id,
            prompt_tokens=saved_tokens,
            use_prefix_cache=use_prefix_cache,
        )
        restored.parent_seq_id = parent_id
        return restored

    def enforce_watermark(
        self,
        victim_order: Optional[Sequence[str]] = None,
    ) -> list[str]:
        """Ensure at least `watermark_blocks` are immediately free in `_free_list`.

        Step 1: Evicts unreferenced cached prefix blocks from `_evictable_lru` into `_free_list`.
        Step 2: If `_free_list` is still below `watermark_blocks`, preempts active running sequences
                (ordered by `victim_order` if given, otherwise least-recently-updated first)
                and evicts any newly unreferenced blocks until the watermark is met.

        Returns:
            List of preempted `seq_id`s.
        """
        preempted: list[str] = []

        def _evict_cached_until_target() -> None:
            for bid in list(self._evictable_lru.keys()):
                if len(self._free_list) >= self.watermark_blocks:
                    break
                blk = self._blocks[bid]
                if not blk.pinned:
                    del self._evictable_lru[bid]
                    self.prefix_cache.invalidate_block(bid)
                    blk.reset()
                    self._free_list[bid] = None
                    self.evictions += 1

        _evict_cached_until_target()
        if len(self._free_list) >= self.watermark_blocks:
            return preempted

        if victim_order is not None:
            candidates = [
                sid
                for sid in victim_order
                if sid in self._sequences and self._sequences[sid].status == SequenceStatus.RUNNING
            ]
        else:
            running = [
                s for s in self._sequences.values() if s.status == SequenceStatus.RUNNING
            ]
            # Preempt least-recently-stepped sequences first
            running.sort(key=lambda s: s.last_step)
            candidates = [s.seq_id for s in running]

        for sid in candidates:
            if len(self._free_list) >= self.watermark_blocks:
                break
            self.preempt_sequence(sid)
            preempted.append(sid)
            _evict_cached_until_target()

        return preempted

    def internal_fragmentation_ratio(self) -> float:
        """Compute internal fragmentation across active sequences.

        Defined as `unused_token_slots_in_allocated_blocks / total_allocated_token_slots`
        across all running sequences' logical block tables.
        """
        total_slots = 0
        wasted_slots = 0
        for seq in self._sequences.values():
            if seq.status != SequenceStatus.RUNNING or not seq.logical_blocks:
                continue
            seq_slots = len(seq.logical_blocks) * self.block_size
            total_slots += seq_slots
            wasted_slots += seq_slots - seq.num_tokens
        if total_slots == 0:
            return 0.0
        return wasted_slots / total_slots

    def stats(self) -> AllocatorStats:
        """Return a complete telemetry snapshot of the allocator."""
        running_seqs = sum(
            1 for s in self._sequences.values() if s.status == SequenceStatus.RUNNING
        )
        swapped_seqs = sum(
            1 for s in self._sequences.values() if s.status == SequenceStatus.SWAPPED
        )
        return AllocatorStats(
            total_blocks=self.num_blocks,
            block_size=self.block_size,
            active_blocks=self.active_block_count,
            cached_evictable_blocks=self.evictable_block_count,
            free_blocks=self.free_block_count,
            active_sequences=running_seqs,
            swapped_sequences=swapped_seqs,
            prefix_cache_hits=self.prefix_cache_hits,
            prefix_cache_misses=self.prefix_cache_misses,
            cow_splits=self.cow_splits,
            evictions=self.evictions,
            preemptions=self.preemptions,
            internal_fragmentation_ratio=self.internal_fragmentation_ratio(),
        )
