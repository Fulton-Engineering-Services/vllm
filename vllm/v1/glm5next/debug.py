# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Env-gated GLM-5.3-Flash diagnostics.

All dumps fire only when ``VLLM_GLM5NEXT_DEBUG=1``. They re-create the
high-value boot-debugging dumps that were removed from the hot paths during
the modular refactor (per-group backend resolution, indexer reshape inputs,
per-group allocation demand).
"""

import os

from vllm.logger import init_logger

logger = init_logger(__name__)

_ENABLED: bool | None = None


def debug_enabled() -> bool:
    global _ENABLED
    if _ENABLED is None:
        _ENABLED = os.environ.get("VLLM_GLM5NEXT_DEBUG", "0") == "1"
    return _ENABLED


def dump_attn_group_resolution(
    kv_cache_group_id: int,
    layer_names: list[str],
    num_resolved: int,
    spec: object,
) -> None:
    """Which layers resolved to backends per KV group (init_attn_backend)."""
    if not debug_enabled():
        return
    logger.warning(
        "GLM5NEXT V1 GROUP gid=%d n_layer_names=%d n_resolved=%d spec=%s layer0=%s",
        kv_cache_group_id,
        len(layer_names),
        num_resolved,
        type(spec).__name__,
        layer_names[0] if layer_names else "-",
    )


def dump_indexer_reshape(
    layer_name: str,
    kv_cache_spec,
    kernel_block_size: int,
    num_blocks: int,
    num_blocks_per_kv_block: int,
    kernel_num_blocks: int,
    shape_block_size: int,
    raw_numel: int,
) -> None:
    """GLM-5.3 kpool indexer reshape inputs (worker KV-cache reshape)."""
    if not debug_enabled():
        return
    if "indexer" not in layer_name or ".tail" in layer_name:
        return
    logger.warning(
        "GLM5NEXT IDX RESHAPE: layer=%s spec.block_size=%d "
        "storage_block_size=%d compress_ratio=%s kernel_block_size=%d "
        "num_blocks=%d num_blocks_per_kv_block=%d kernel_num_blocks=%d "
        "shape_block_size=%d raw.numel=%d page=%d",
        layer_name,
        kv_cache_spec.block_size,
        kv_cache_spec.storage_block_size,
        getattr(kv_cache_spec, "compress_ratio", "-"),
        kernel_block_size,
        num_blocks,
        num_blocks_per_kv_block,
        kernel_num_blocks,
        shape_block_size,
        raw_numel,
        kv_cache_spec.page_size_bytes,
    )


def dump_group_allocation(
    request_id: str,
    num_tokens: int,
    total_computed_tokens: int,
    apply_admission_cap: bool,
    groups: list,
    per_group: list[int],
    total: int,
    num_free_blocks: int,
) -> None:
    """Per-group block demand for one allocation request (coordinator)."""
    if not debug_enabled():
        return
    logger.warning(
        "GLM5NEXT ALLOC req=%s num_tokens=%d computed=%d cap=%s "
        "per-group=%s total=%d free=%d",
        request_id,
        num_tokens,
        total_computed_tokens,
        apply_admission_cap,
        [
            (i, type(g.kv_cache_spec).__name__, g.kv_cache_spec.block_size, d)
            for (i, g), d in zip(enumerate(groups), per_group)
        ],
        total,
        num_free_blocks,
    )
