# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3-Flash (glm5_next) kpool indexer attention backends.

Extracted from ``vllm.v1.attention.backends.mla.indexer`` so the upstream
DeepSeek indexer file keeps only generic code. Selected by the model layers'
``get_attn_backend()`` (Glm5NextIndexerCache / Glm5NextTailCache), not by a
backend registry.
"""

from typing import ClassVar

from vllm.platforms import current_platform
from vllm.v1.attention.backend import MultipleOf
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerBackend,
    DeepseekV32IndexerMetadataBuilder,
)


class Glm5NextKpoolIndexerBackend(DeepseekV32IndexerBackend):
    """GLM-5.3-Flash kpool indexer.

    The kpool indexer spec pins ``block_size = kernel_tile * index_kpool`` so
    the runtime ``block_kv = block_size // index_kpool`` lands in DeepGEMM's
    legal ``{32, 64}``. Advertising the spec's own block size (256 for
    ``index_kpool = 4``) keeps ``select_common_block_size`` from splitting the
    indexer block down to the base 64, which would make ``block_kv = 16`` and
    trip the DeepGEMM paged-MQA assert (csrc/apis/attention.hpp:262).
    """

    @staticmethod
    def get_name() -> str:
        return "GLM5NEXT_KPOOL_INDEXER"

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [1, MultipleOf(16)] if current_platform.is_rocm() else [256]


class KpoolTailMetadataBuilder(DeepseekV32IndexerMetadataBuilder):
    """Builder for the storage-only kpool tail cache.

    Produces the same DeepseekV32IndexerMetadata shape as the indexer
    builder (the kpool op asserts on it and reads only ``slot_mapping`` /
    ``num_decode_tokens``) but never schedules DeepGEMM paged-MQA work:
    the tail spec's storage block is the ratio-scaled model block size
    (e.g. 3072 tokens after page unification against the padded indexer
    page), which is not a valid DeepGEMM block_kv and trips the kernel's
    ``block_kv == 64`` assert (csrc/apis/attention.hpp:220).
    """

    schedules_deepgemm_paged_mqa: ClassVar[bool] = False

    def _prepare_decode_tensors(self, *args, block_table, **kwargs):
        # The runner sizes the tail group's block table for the full context,
        # but the tail manager allocates one manager block per request, so only
        # the first buffer-width columns are ever populated. Spec-decode's
        # variable decode lengths take a shape-checked torch copy that rejects
        # the full-width table; truncate to the populated columns.
        block_table = block_table[:, : self.expanded_block_table_buffer.shape[1]]
        return super()._prepare_decode_tensors(*args, block_table=block_table, **kwargs)


class KpoolTailBackend(DeepseekV32IndexerBackend):
    """Storage-only backend for the GLM-5.3-Flash kpool tail cache."""

    @staticmethod
    def get_name() -> str:
        return "KPOOL_TAIL"

    @staticmethod
    def get_builder_cls() -> type["KpoolTailMetadataBuilder"]:
        return KpoolTailMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        return (num_blocks, num_kv_heads, block_size, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        if include_num_layers_dimension:
            return (0, 1, 2, 3, 4)
        return (0, 1, 2, 3)

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return []

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        # The tail kernels index [num_blocks, 2, pool_size, head_dim] and
        # assert shape[2] == index_kpool (kpool_compress.py), so the kernel
        # block size must be exactly the pool size (4 for GLM-5.3-Flash).
        # MultipleOf(1) let select_common_block_size pick the full
        # ratio-scaled manager block (e.g. 3072), producing a [N, 2, 3072,
        # 128] cache view that fails the assert. A literal size forces
        # virtual block splitting (blocks_per_kv_block = 3072/4), matching
        # the tail kernels' pool-granular slot arithmetic
        # (block * kpool + pos % kpool).
        return [4]
