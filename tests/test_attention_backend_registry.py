# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionImpl,
)
from vllm.v1.attention.backends.registry import (
    AttentionBackendEnum,
    MambaAttentionBackendEnum,
    register_backend,
)


class CustomAttentionImpl(AttentionImpl):
    """Mock custom attention implementation for testing."""

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, *args, **kwargs):
        """Mock forward pass."""
        pass


class CustomAttentionBackend(AttentionBackend):
    """Mock custom attention backend for testing."""

    @staticmethod
    def get_name():
        return "CUSTOM"

    @staticmethod
    def get_impl_cls():
        return CustomAttentionImpl

    @staticmethod
    def get_builder_cls():
        """Mock builder class."""
        return None

    @staticmethod
    def get_required_kv_cache_layout():
        """Mock KV cache layout."""
        return None


class CustomMambaAttentionImpl(AttentionImpl):
    """Mock custom mamba attention implementation for testing."""

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, *args, **kwargs):
        """Mock forward pass."""
        pass


class CustomMambaAttentionBackend(AttentionBackend):
    """Mock custom mamba attention backend for testing."""

    @staticmethod
    def get_name():
        return "CUSTOM_MAMBA"

    @staticmethod
    def get_impl_cls():
        return CustomMambaAttentionImpl

    @staticmethod
    def get_builder_cls():
        """Mock builder class."""
        return None

    @staticmethod
    def get_required_kv_cache_layout():
        """Mock KV cache layout."""
        return None


def test_custom_is_not_alias_of_any_backend():
    # Get all members of AttentionBackendEnum
    all_backends = list(AttentionBackendEnum)

    # Find any aliases of CUSTOM
    aliases = []
    for backend in all_backends:
        if backend.name != "CUSTOM" and backend is AttentionBackendEnum.CUSTOM:
            aliases.append(backend.name)

    # CUSTOM should not be an alias of any other backend
    assert len(aliases) == 0, (
        f"BUG! CUSTOM is an alias of: {', '.join(aliases)}!\n"
        f"CUSTOM.value = {repr(AttentionBackendEnum.CUSTOM.value)}\n"
        f"This happens when CUSTOM has the same value as another backend.\n"
        f"When you register to CUSTOM, you're actually registering to {aliases[0]}!\n"
        f"All backend values:\n"
        + "\n".join(f"  {b.name}: {repr(b.value)}" for b in all_backends)
    )

    # Verify CUSTOM has its own unique identity
    assert AttentionBackendEnum.CUSTOM.name == "CUSTOM", (
        f"CUSTOM.name should be 'CUSTOM', but got '{AttentionBackendEnum.CUSTOM.name}'"
    )


def test_register_custom_backend_with_class_path():
    # Register with explicit class path
    register_backend(
        backend=AttentionBackendEnum.CUSTOM,
        class_path="tests.test_attention_backend_registry.CustomAttentionBackend",
        is_mamba=False,
    )

    # Check that CUSTOM backend is registered
    assert AttentionBackendEnum.CUSTOM.is_overridden(), (
        "CUSTOM should be overridden after registration"
    )

    # Get the registered class path
    class_path = AttentionBackendEnum.CUSTOM.get_path()
    assert class_path == "tests.test_attention_backend_registry.CustomAttentionBackend"

    # Get the backend class
    backend_cls = AttentionBackendEnum.CUSTOM.get_class()
    assert backend_cls.get_name() == "CUSTOM"
    assert backend_cls.get_impl_cls() == CustomAttentionImpl


def test_mamba_custom_is_not_alias_of_any_backend():
    # Get all mamba backends
    all_backends = list(MambaAttentionBackendEnum)

    # Find any aliases of CUSTOM
    aliases = []
    for backend in all_backends:
        if backend.name != "CUSTOM" and backend is MambaAttentionBackendEnum.CUSTOM:
            aliases.append(backend.name)

    # CUSTOM should not be an alias of any other backend
    assert len(aliases) == 0, (
        f"BUG! MambaAttentionBackendEnum.CUSTOM is an alias of: {', '.join(aliases)}!\n"
        f"CUSTOM.value = {repr(MambaAttentionBackendEnum.CUSTOM.value)}\n"
        f"All mamba backend values:\n"
        + "\n".join(f"  {b.name}: {repr(b.value)}" for b in all_backends)
    )


def test_register_custom_mamba_backend_with_class_path():
    # Register with explicit class path
    register_backend(
        backend=MambaAttentionBackendEnum.CUSTOM,
        class_path="tests.test_attention_backend_registry.CustomMambaAttentionBackend",
        is_mamba=True,
    )

    # Check that the backend is registered
    assert MambaAttentionBackendEnum.CUSTOM.is_overridden()

    # Get the registered class path
    class_path = MambaAttentionBackendEnum.CUSTOM.get_path()
    assert (
        class_path
        == "tests.test_attention_backend_registry.CustomMambaAttentionBackend"
    )

    # Get the backend class
    backend_cls = MambaAttentionBackendEnum.CUSTOM.get_class()
    assert backend_cls.get_name() == "CUSTOM_MAMBA"
    assert backend_cls.get_impl_cls() == CustomMambaAttentionImpl


# ---------------------------------------------------------------------------
# FlashInfer SM90 NoPE MLA feature detection tests
# ---------------------------------------------------------------------------
# has_flashinfer_sm90_nope_mla() gates whether the SM90 sparse MLA backend
# accepts fp8 KV cache. If the deployed FlashInfer build lacks the
# ckv_scale_arr parameter in BatchMLAPagedAttentionWrapper.run, SM90 rejects
# fp8 KV and SM120 (with incompatible KV layout) takes over, causing OOM
# on GLM-5.3 Flash deployments.

def test_has_flashinfer_sm90_nope_mla_returns_bool():
    """has_flashinfer_sm90_nope_mla() must return a bool (not None/exception).

    The function is @functools.cache'd, so it runs once per process. It
    inspects FlashInfer's BatchMLAPagedAttentionWrapper.run signature for
    the ckv_scale_arr keyword-only parameter.
    """
    from vllm.utils.flashinfer import has_flashinfer_sm90_nope_mla

    result = has_flashinfer_sm90_nope_mla()
    assert isinstance(result, bool), (
        f"has_flashinfer_sm90_nope_mla() must return bool, got {type(result)}"
    )


def test_flashinfer_sm90_nope_mla_enum_exists():
    """FLASHINFER_MLA_SPARSE_SM90 must be a registered backend enum."""
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    assert hasattr(AttentionBackendEnum, "FLASHINFER_MLA_SPARSE_SM90"), (
        "FLASHINFER_MLA_SPARSE_SM90 must exist in AttentionBackendEnum"
    )


def test_flashinfer_sm120_enum_exists():
    """FLASHINFER_MLA_SPARSE_SM120 must be a registered backend enum."""
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    assert hasattr(AttentionBackendEnum, "FLASHINFER_MLA_SPARSE_SM120"), (
        "FLASHINFER_MLA_SPARSE_SM120 must exist in AttentionBackendEnum"
    )


def test_sm90_and_sm120_are_distinct_enums():
    """SM90 and SM120 must be distinct enum values (not aliases).

    If they share the same value, registering one overwrites the other,
    and the priority list silently routes to the wrong backend.
    """
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    sm90 = AttentionBackendEnum.FLASHINFER_MLA_SPARSE_SM90
    sm120 = AttentionBackendEnum.FLASHINFER_MLA_SPARSE_SM120
    assert sm90 is not sm120, (
        "SM90 and SM120 must be distinct enum members, not aliases"
    )
    assert sm90.value != sm120.value, (
        f"SM90 and SM120 must have distinct values, "
        f"both are {sm90.value!r}"
    )
