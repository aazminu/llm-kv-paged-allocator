"""Comprehensive test suite for llm-kv-paged-allocator."""

from __future__ import annotations

import pytest

from kv_paged import (
    AllocationError,
    OutOfMemoryError,
    PagedKVAllocator,
    SequenceNotFoundError,
    SequenceStatus,
    compute_block_prefix_hash,
)


def test_prefix_hash_chaining_and_determinism() -> None:
    tokens_a = [10, 20, 30, 40]
    tokens_b = [50, 60, 70, 80]

    h1_fnv = compute_block_prefix_hash(tokens_a, parent_hash=None, algo="fnv1a")
    h1_fnv_again = compute_block_prefix_hash(tokens_a, parent_hash=None, algo="fnv1a")
    assert h1_fnv == h1_fnv_again
    assert h1_fnv.startswith("fnv1a:")

    # Same local block tokens under different parent hashes must produce distinct chained hashes
    c1 = compute_block_prefix_hash(tokens_b, parent_hash=h1_fnv, algo="fnv1a")
    c2 = compute_block_prefix_hash(tokens_b, parent_hash="fnv1a:0000000000000001", algo="fnv1a")
    assert c1 != c2

    # SHA-256 backend works identically
    s1 = compute_block_prefix_hash(tokens_a, parent_hash=None, algo="sha256")
    s2 = compute_block_prefix_hash(tokens_b, parent_hash=s1, algo="sha256")
    assert s1.startswith("sha256:")
    assert s2.startswith("sha256:")
    assert s1 != s2


def test_basic_sequence_allocation_and_token_reconstruction() -> None:
    alloc = PagedKVAllocator(num_blocks=16, block_size=4)
    prompt = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    seq = alloc.allocate_sequence("req-1", prompt_tokens=prompt)

    assert seq.num_tokens == 10
    assert len(seq.logical_blocks) == 3  # 4 + 4 + 2 tokens
    assert alloc.active_block_count == 3
    assert alloc.free_block_count == 13
    assert alloc.get_sequence_tokens("req-1") == prompt


def test_append_tokens_across_block_boundaries() -> None:
    alloc = PagedKVAllocator(num_blocks=10, block_size=4)
    alloc.allocate_sequence("req-1", prompt_tokens=[1, 2])
    assert alloc.active_block_count == 1

    # Append 2 tokens to finish block 0, plus 5 more tokens (1 full block + 1 partial block)
    blocks = alloc.append_tokens("req-1", [3, 4, 5, 6, 7, 8, 9])
    assert len(blocks) == 3
    assert alloc.active_block_count == 3
    assert alloc.get_sequence_tokens("req-1") == [1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert alloc.get_block(blocks[0]).is_full
    assert alloc.get_block(blocks[1]).is_full
    assert not alloc.get_block(blocks[2]).is_full
    assert alloc.get_block(blocks[2]).free_slots == 3


def test_prefix_cache_sharing_between_concurrent_sequences() -> None:
    alloc = PagedKVAllocator(num_blocks=16, block_size=4)
    system_prefix = [101, 102, 103, 104, 201, 202, 203, 204]
    seq1 = alloc.allocate_sequence("user-1", prompt_tokens=system_prefix + [901, 902])
    seq2 = alloc.allocate_sequence("user-2", prompt_tokens=system_prefix + [999])

    # First two full blocks must map to the exact same physical blocks
    assert seq1.logical_blocks[:2] == seq2.logical_blocks[:2]
    assert seq1.logical_blocks[2] != seq2.logical_blocks[2]

    # Reference counts on shared prefix blocks must be 2
    assert alloc.get_block(seq1.logical_blocks[0]).ref_count == 2
    assert alloc.get_block(seq1.logical_blocks[1]).ref_count == 2
    assert alloc.get_block(seq1.logical_blocks[2]).ref_count == 1
    assert alloc.get_block(seq2.logical_blocks[2]).ref_count == 1

    # Total active physical blocks = 2 shared + 2 distinct tails = 4 (instead of 6)
    assert alloc.active_block_count == 4
    assert alloc.prefix_cache_hits == 2


def test_prefix_cache_resurrection_after_sequence_free() -> None:
    alloc = PagedKVAllocator(num_blocks=8, block_size=4)
    prompt = [10, 11, 12, 13, 14, 15, 16, 17, 99]
    seq1 = alloc.allocate_sequence("seq-1", prompt_tokens=prompt)
    first_two_blocks = list(seq1.logical_blocks[:2])

    alloc.free_sequence("seq-1", keep_in_prefix_cache=True)
    # Full blocks move to evictable LRU; partial tail block returns directly to free list
    assert alloc.active_block_count == 0
    assert alloc.evictable_block_count == 2
    assert alloc.free_block_count == 6

    # Allocate a new sequence with the same 8-token prefix: should resurrect both blocks
    seq2 = alloc.allocate_sequence("seq-2", prompt_tokens=[10, 11, 12, 13, 14, 15, 16, 17, 42])
    assert seq2.logical_blocks[:2] == first_two_blocks
    assert alloc.evictable_block_count == 0
    assert alloc.active_block_count == 3


def test_fork_sequence_zero_copy_and_copy_on_write_on_append() -> None:
    alloc = PagedKVAllocator(num_blocks=12, block_size=4)
    # 1 full block + 1 partial block (2 tokens)
    parent = alloc.allocate_sequence("beam-0", prompt_tokens=[1, 2, 3, 4, 5, 6])
    orig_blocks = list(parent.logical_blocks)
    assert alloc.active_block_count == 2

    child = alloc.fork_sequence("beam-0", "beam-1")
    assert child.logical_blocks == orig_blocks
    assert alloc.active_block_count == 2  # Zero new blocks allocated on fork!
    assert alloc.get_block(orig_blocks[0]).ref_count == 2
    assert alloc.get_block(orig_blocks[1]).ref_count == 2

    # Appending to child must trigger Copy-on-Write on the shared partial block (logical index 1)
    alloc.append_tokens("beam-1", [70])
    assert alloc.cow_splits == 1
    assert alloc.active_block_count == 3

    # Full block 0 remains shared; tail block 1 diverged
    assert parent.logical_blocks[0] == child.logical_blocks[0]
    assert parent.logical_blocks[1] != child.logical_blocks[1]
    assert alloc.get_block(parent.logical_blocks[0]).ref_count == 2
    assert alloc.get_block(parent.logical_blocks[1]).ref_count == 1
    assert alloc.get_block(child.logical_blocks[1]).ref_count == 1

    # Appending to parent now requires NO CoW split because parent's tail ref_count is back to 1
    alloc.append_tokens("beam-0", [60])
    assert alloc.cow_splits == 1
    assert alloc.get_sequence_tokens("beam-0") == [1, 2, 3, 4, 5, 6, 60]
    assert alloc.get_sequence_tokens("beam-1") == [1, 2, 3, 4, 5, 6, 70]


def test_fork_sequence_full_block_append_does_not_cow_full_block() -> None:
    alloc = PagedKVAllocator(num_blocks=10, block_size=4)
    # Exactly 1 full block (4 tokens)
    parent = alloc.allocate_sequence("parent", prompt_tokens=[1, 2, 3, 4])
    child = alloc.fork_sequence("parent", "child")

    # Appending to child allocates a new block #2 without copying full block #1
    alloc.append_tokens("child", [5, 6])
    assert alloc.cow_splits == 0
    assert child.logical_blocks[0] == parent.logical_blocks[0]
    assert alloc.get_block(parent.logical_blocks[0]).ref_count == 2
    assert len(child.logical_blocks) == 2


def test_overwrite_token_triggers_cow_and_rehashes_chain() -> None:
    alloc = PagedKVAllocator(num_blocks=12, block_size=4)
    s1 = alloc.allocate_sequence("s1", prompt_tokens=[1, 2, 3, 4, 5, 6, 7, 8])
    s2 = alloc.fork_sequence("s1", "s2")
    assert s1.logical_blocks == s2.logical_blocks

    # Mutate token index 1 in s2 (inside first block)
    alloc.overwrite_token("s2", token_index=1, new_token=99)
    assert alloc.get_sequence_tokens("s1") == [1, 2, 3, 4, 5, 6, 7, 8]
    assert alloc.get_sequence_tokens("s2") == [1, 99, 3, 4, 5, 6, 7, 8]
    assert s1.logical_blocks[0] != s2.logical_blocks[0]
    # Because block 0 changed, block 1's chained prefix hash in s2 must also differ from s1's block 1
    assert s1.logical_blocks[1] != s2.logical_blocks[1]


def test_convergent_append_deduplicates_full_block_into_prefix_cache() -> None:
    alloc = PagedKVAllocator(num_blocks=10, block_size=4)
    s1 = alloc.allocate_sequence("s1", prompt_tokens=[1, 2, 3, 4])
    s2 = alloc.allocate_sequence("s2", prompt_tokens=[1, 2], use_prefix_cache=True)
    assert len(s2.logical_blocks) == 1
    assert s2.logical_blocks[0] != s1.logical_blocks[0]

    # Once s2 appends [3, 4] completing [1, 2, 3, 4], it should deduplicate into s1's full block
    alloc.append_tokens("s2", [3, 4], register_in_cache=True)
    assert s2.logical_blocks[0] == s1.logical_blocks[0]
    assert alloc.get_block(s1.logical_blocks[0]).ref_count == 2
    assert alloc.active_block_count == 1


def test_lru_eviction_order_of_cached_blocks() -> None:
    alloc = PagedKVAllocator(num_blocks=4, block_size=4)
    # Fill 4 full blocks across two sequences
    s1 = alloc.allocate_sequence("s1", prompt_tokens=[1, 2, 3, 4, 5, 6, 7, 8])
    s2 = alloc.allocate_sequence("s2", prompt_tokens=[10, 20, 30, 40, 50, 60, 70, 80])
    b_s1 = list(s1.logical_blocks)
    b_s2 = list(s2.logical_blocks)

    # Free s1 first, then s2 -> s1's blocks are older in the evictable LRU
    alloc.free_sequence("s1", keep_in_prefix_cache=True)
    alloc.free_sequence("s2", keep_in_prefix_cache=True)
    assert alloc.evictable_block_count == 4
    assert alloc.free_block_count == 0

    # Allocating a brand-new 2-block sequence must evict s1's blocks and preserve s2's blocks
    alloc.allocate_sequence("s3", prompt_tokens=[100, 101, 102, 103, 104, 105, 106, 107])
    assert alloc.evictions == 2

    # s2's prefix should still be a cache hit!
    s2_again = alloc.allocate_sequence("s2-Cached", prompt_tokens=[10, 20, 30, 40, 50, 60, 70, 80])
    assert s2_again.logical_blocks == b_s2
    assert b_s1[0] != b_s2[0]


def test_watermark_enforcement_evicts_cache_then_preempts_lru_sequence() -> None:
    # 10 blocks total, watermark_ratio=0.3 -> watermark = 3 free blocks required
    alloc = PagedKVAllocator(num_blocks=10, block_size=4, watermark_ratio=0.30)
    assert alloc.watermark_blocks == 3

    # Allocate 3 sequences of 3 blocks each = 9 blocks used, 1 free block left (< 3 watermark)
    alloc.allocate_sequence("old-seq", prompt_tokens=list(range(1, 13)))
    alloc.allocate_sequence("mid-seq", prompt_tokens=list(range(101, 113)))
    alloc.allocate_sequence("new-seq", prompt_tokens=list(range(201, 213)))

    # Touch mid-seq so old-seq remains the oldest
    alloc.append_tokens("mid-seq", [])
    assert alloc.is_below_watermark()

    preempted = alloc.enforce_watermark()
    assert preempted == ["old-seq"]
    assert alloc.get_sequence("old-seq").status == SequenceStatus.SWAPPED
    assert alloc.free_block_count >= 3
    assert not alloc.is_below_watermark()


def test_preempt_and_resume_sequence_roundtrip() -> None:
    alloc = PagedKVAllocator(num_blocks=6, block_size=4)
    tokens = [11, 22, 33, 44, 55, 66, 77]
    alloc.allocate_sequence("req-swap", prompt_tokens=tokens)
    assert alloc.active_block_count == 2

    swapped = alloc.preempt_sequence("req-swap")
    assert swapped == tokens
    assert alloc.active_block_count == 0
    assert alloc.get_sequence("req-swap").status == SequenceStatus.SWAPPED
    assert alloc.get_sequence_tokens("req-swap") == tokens

    resumed = alloc.resume_sequence("req-swap")
    assert resumed.status == SequenceStatus.RUNNING
    assert alloc.get_sequence_tokens("req-swap") == tokens
    assert alloc.active_block_count == 2


def test_out_of_memory_error_when_blocks_exhausted() -> None:
    alloc = PagedKVAllocator(num_blocks=2, block_size=4)
    alloc.allocate_sequence("s1", prompt_tokens=[1, 2, 3, 4, 5, 6, 7, 8])
    assert alloc.available_block_count == 0

    with pytest.raises(OutOfMemoryError):
        alloc.allocate_sequence("s2", prompt_tokens=[9])

    with pytest.raises(OutOfMemoryError):
        alloc.append_tokens("s1", [9])


def test_internal_fragmentation_and_stats_telemetry() -> None:
    alloc = PagedKVAllocator(num_blocks=8, block_size=4, watermark_ratio=0.25)
    # 5 tokens -> 2 blocks (8 slots), 3 unused slots -> fragmentation = 3/8 = 0.375
    alloc.allocate_sequence("s1", prompt_tokens=[1, 2, 3, 4, 5])
    stats = alloc.stats()

    assert stats.total_blocks == 8
    assert stats.block_size == 4
    assert stats.active_blocks == 2
    assert stats.free_blocks == 6
    assert stats.active_sequences == 1
    assert pytest.approx(stats.internal_fragmentation_ratio, rel=1e-6) == 3.0 / 8.0


def test_sha256_allocator_mode_and_collision_safety() -> None:
    alloc = PagedKVAllocator(num_blocks=8, block_size=4, hash_algo="sha256")
    s1 = alloc.allocate_sequence("a", prompt_tokens=[1, 2, 3, 4, 5])
    s2 = alloc.allocate_sequence("b", prompt_tokens=[1, 2, 3, 4, 6])
    assert s1.logical_blocks[0] == s2.logical_blocks[0]
    blk0 = alloc.get_block(s1.logical_blocks[0])
    assert blk0.prefix_hash is not None and blk0.prefix_hash.startswith("sha256:")


def test_error_handling_and_validation_guards() -> None:
    with pytest.raises(ValueError):
        PagedKVAllocator(num_blocks=0)
    with pytest.raises(ValueError):
        PagedKVAllocator(num_blocks=4, block_size=0)
    with pytest.raises(ValueError):
        PagedKVAllocator(num_blocks=4, watermark_ratio=1.0)

    alloc = PagedKVAllocator(num_blocks=4, block_size=4)
    with pytest.raises(SequenceNotFoundError):
        alloc.get_sequence("missing")

    alloc.allocate_sequence("s1", prompt_tokens=[1, 2])
    with pytest.raises(AllocationError):
        alloc.allocate_sequence("s1", prompt_tokens=[3, 4])
    with pytest.raises(IndexError):
        alloc.get_block(99)
    with pytest.raises(IndexError):
        alloc.overwrite_token("s1", token_index=5, new_token=10)


def test_pinned_cached_block_is_skipped_during_eviction() -> None:
    alloc = PagedKVAllocator(num_blocks=2, block_size=4)
    s1 = alloc.allocate_sequence("s1", prompt_tokens=[1, 2, 3, 4])
    s2 = alloc.allocate_sequence("s2", prompt_tokens=[5, 6, 7, 8])
    b1 = s1.logical_blocks[0]
    b2 = s2.logical_blocks[0]

    alloc.free_sequence("s1", keep_in_prefix_cache=True)
    alloc.free_sequence("s2", keep_in_prefix_cache=True)

    # Pin the older block b1 so eviction must skip b1 and evict b2
    alloc.get_block(b1).pinned = True
    s3 = alloc.allocate_sequence("s3", prompt_tokens=[9, 10, 11, 12])
    assert s3.logical_blocks[0] == b2
    assert alloc.get_block(b1).tokens == [1, 2, 3, 4]


def test_enforce_watermark_with_explicit_victim_order() -> None:
    alloc = PagedKVAllocator(num_blocks=8, block_size=4, watermark_ratio=0.25)
    alloc.allocate_sequence("high-prio", prompt_tokens=[1, 2, 3, 4, 5, 6, 7, 8])
    alloc.allocate_sequence("med-prio", prompt_tokens=[10, 11, 12, 13, 14, 15, 16, 17])
    alloc.allocate_sequence("low-prio", prompt_tokens=[20, 21, 22, 23, 24, 25, 26, 27])
    # 6 blocks used, 2 free -> allocate 1 more block so only 1 free (< watermark=2)
    alloc.allocate_sequence("tiny-prio", prompt_tokens=[99])
    assert alloc.is_below_watermark()

    preempted = alloc.enforce_watermark(victim_order=["low-prio", "med-prio", "high-prio"])
    assert preempted == ["low-prio"]
    assert alloc.get_sequence("low-prio").status == SequenceStatus.SWAPPED
    assert alloc.get_sequence("high-prio").status == SequenceStatus.RUNNING


def test_multi_branch_beam_search_tree_forking_and_partial_free() -> None:
    alloc = PagedKVAllocator(num_blocks=16, block_size=4)
    root = alloc.allocate_sequence("root", prompt_tokens=[1, 2, 3, 4, 5, 6])
    b1 = alloc.fork_sequence("root", "beam-1")
    b2 = alloc.fork_sequence("root", "beam-2")
    assert alloc.get_block(root.logical_blocks[0]).ref_count == 3
    assert alloc.get_block(root.logical_blocks[1]).ref_count == 3

    # Free root while beams continue
    alloc.free_sequence("root")
    assert alloc.get_block(b1.logical_blocks[0]).ref_count == 2
    assert alloc.get_block(b1.logical_blocks[1]).ref_count == 2

    alloc.append_tokens("beam-1", [10, 11, 12])
    alloc.append_tokens("beam-2", [20, 21, 22])
    assert alloc.get_sequence_tokens("beam-1") == [1, 2, 3, 4, 5, 6, 10, 11, 12]
    assert alloc.get_sequence_tokens("beam-2") == [1, 2, 3, 4, 5, 6, 20, 21, 22]


def test_disable_prefix_cache_flag_isolates_blocks() -> None:
    alloc = PagedKVAllocator(num_blocks=8, block_size=4)
    s1 = alloc.allocate_sequence("s1", prompt_tokens=[1, 2, 3, 4], use_prefix_cache=True)
    s2 = alloc.allocate_sequence("s2", prompt_tokens=[1, 2, 3, 4], use_prefix_cache=False)
    assert s1.logical_blocks[0] != s2.logical_blocks[0]
    assert alloc.active_block_count == 2
