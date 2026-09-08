# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure GLM-5.3-Flash KV spec math (engine-side, model-free).

The model package's ``Glm5NextIndexerCache`` / ``Glm5NextTailCache`` adapt
these builders onto the layer objects; tests (and the deployment sizing pins)
import them directly, which keeps engine KV-cache logic importable without
``vllm.models.*`` (and therefore runnable on CPU-only hosts).
"""

import torch

from vllm.v1.glm5next.kv_specs import KpoolTailSpec
from vllm.v1.kv_cache_interface import MLAAttentionSpec

# fp8 naive cache: the per-<quant_block_size> scale is folded into head_dim
# (one uint8 payload + one fp32 scale per 128 elems -> +4 B per 128).
INDEXER_QUANT_BLOCK_SIZE = 128


def indexer_head_dim(index_head_dim: int, quant_block_size: int = 128) -> int:
    """Stored indexer head size after folding the fp8 per-128 scale in.

    head_dim = index_head_dim + index_head_dim // quant_block_size * 4
    (128 -> 132 for the deployment's index_head_dim=128).
    """
    return index_head_dim + index_head_dim // quant_block_size * 4


def build_indexer_spec(
    cache_block_size: int,
    head_dim: int,
    dtype: torch.dtype,
    index_kpool: int,
) -> MLAAttentionSpec:
    """KV spec for the kpool-compressed indexer K cache.

    Presents the model-wide scheduler block (e.g. 2304) to the KV manager so
    the indexer's pool accounting is uniform with the co-located MLA.
    ``compress_ratio = index_kpool`` is the kpool pooling ratio (4 tokens ->
    1 state); it is load-bearing for the pooling math downstream
    (seq_lens // compress_ratio), NOT merely page sizing. The physical
    DeepGEMM page (64 states) is derived at the worker from the kernel block
    split, not from block_size // compress_ratio.
    """
    assert index_kpool > 1, "Glm5NextIndexerCache expects index_kpool > 1"
    # Keep chunked-prefill boundaries aligned to complete pools.
    assert cache_block_size % index_kpool == 0, (
        "Glm5NextIndexerCache: cache_config.block_size "
        f"({cache_block_size}) must be a multiple of index_kpool "
        f"({index_kpool}) so chunked-prefill boundaries stay pool-aligned."
    )
    return MLAAttentionSpec(
        block_size=cache_block_size,
        num_kv_heads=1,
        head_size=head_dim,
        dtype=dtype,
        compress_ratio=index_kpool,
        tokens_per_state=index_kpool,
    )


def build_tail_spec(head_dim: int, index_kpool: int) -> KpoolTailSpec:
    """KV spec for the kpool indexer's in-progress (tail) pool.

    Stores raw bf16 K (``head_dim``) as the "K" half of each block and the
    bf16 gate score (``head_dim``) as the "V" half — not the fp8-compressed
    entry, which lives in the indexer cache. The two head slots form
    [K, gate score] in the generic [block, head, state, content] cache view.
    """
    assert index_kpool > 1, "Glm5NextTailCache expects index_kpool > 1"
    return KpoolTailSpec(
        block_size=index_kpool,
        num_kv_heads=2,
        head_size=head_dim,
        head_size_v=0,
        dtype=torch.bfloat16,
        sliding_window=index_kpool,
    )
