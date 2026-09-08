# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3-Flash (glm5_next) KV cache spec definitions.

``KpoolTailSpec`` is registered against ``KpoolTailManager`` in
``vllm.v1.core.single_type_kv_cache_manager.register_all_kvcache_specs``.
Must stay importable without GPU or model-package dependencies.
"""

from dataclasses import dataclass

from vllm.config import VllmConfig
from vllm.v1.kv_cache_interface import KVCacheSpec, SlidingWindowSpec


@dataclass(frozen=True, kw_only=True)
class KpoolTailSpec(SlidingWindowSpec):
    """One-block circular scratch cache for a kpool indexer's raw tail."""

    def max_admission_blocks_per_request(
        self, max_in_flight_tokens: int, max_model_len: int
    ) -> int:
        return 1

    def max_num_blocks_per_req(self, vllm_config: VllmConfig, max_len: int) -> int:
        return 1

    def is_uniform_with_collection(
        self, kv_cache_specs: dict[str, KVCacheSpec]
    ) -> bool:
        return all(isinstance(spec, KpoolTailSpec) for spec in kv_cache_specs.values())

    @property
    def participates_in_prefix_caching(self) -> bool:
        return False
