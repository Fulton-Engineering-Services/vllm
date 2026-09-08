# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engine-side GLM-5.3-Flash (glm5_next) integration.

Carries the fork's GLM-5.3 KV-cache logic extracted from vLLM's core engine
files (kv_cache_utils, single_type_kv_cache_manager, indexer, worker utils).
Modules here import only engine primitives — never ``vllm.models.*`` — so the
package and its tests run on CPU-only hosts.

Import direction: engine files import hooks from this package lazily
(function-local) where a cycle would otherwise form; this package imports
engine primitives at module top. Import submodules directly
(``vllm.v1.glm5next.kv_specs``, ...) — the package init stays free of
submodule imports so importing one submodule never pulls in the others.
"""
