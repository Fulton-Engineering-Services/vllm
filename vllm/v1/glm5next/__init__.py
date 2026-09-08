# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engine-side GLM-5.3-Flash (glm5_next) integration.

Carries the fork's GLM-5.3 KV-cache logic extracted from vLLM's core engine
files (kv_cache_utils, single_type_kv_cache_manager, indexer, worker utils).
Modules here import only engine primitives — never ``vllm.models.*`` — so the
package and its tests run on CPU-only hosts.

Import direction: engine files import hooks from this package lazily
(function-local) where a cycle would otherwise form; this package imports
engine primitives at module top. ``__init__`` resolves names lazily (PEP 562)
so importing any submodule never triggers the engine-module import chain.
"""


def __getattr__(name: str):
    if name == "KpoolTailSpec":
        from vllm.v1.glm5next.kv_specs import KpoolTailSpec

        return KpoolTailSpec
    if name == "KpoolTailManager":
        from vllm.v1.glm5next.tail_manager import KpoolTailManager

        return KpoolTailManager
    if name == "try_build_kv_cache_groups":
        from vllm.v1.glm5next.kv_groups import try_build_kv_cache_groups

        return try_build_kv_cache_groups
    if name == "detect_layout":
        from vllm.v1.glm5next.tensor_layout import detect_layout

        return detect_layout
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
