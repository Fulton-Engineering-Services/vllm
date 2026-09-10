# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only tests for the singleton-TP swap util.

Covers get_or_init_singleton_tp_group / swapped_tp_group: idempotent
construction, world_size==1 (the condition parallel layers gate their
collectives on), and exact swap/restore of the active TP group.
"""

import pytest
import torch.distributed as dist

import vllm.distributed.parallel_state as ps
from vllm.distributed import (
    destroy_distributed_environment,
    destroy_model_parallel,
    get_or_init_singleton_tp_group,
    get_tensor_model_parallel_world_size,
    get_tp_group,
    init_model_parallel_group,
    init_world_group,
    swapped_tp_group,
)

pytestmark = pytest.mark.cpu_test


@pytest.fixture
def dist_env(monkeypatch):
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setenv("MASTER_PORT", "29517")
    dist.init_process_group("gloo", rank=0, world_size=1)
    ps._WORLD = init_world_group([0], 0, "gloo")
    ps._TP = init_model_parallel_group(
        [[0]], 0, "gloo", use_device_communicator=False, group_name="tp"
    )
    yield
    destroy_model_parallel()
    destroy_distributed_environment()


def test_singleton_group_shape_and_idempotent_init(dist_env):
    singleton = get_or_init_singleton_tp_group()
    assert singleton.world_size == 1
    assert singleton.rank_in_group == 0
    assert get_or_init_singleton_tp_group() is singleton
    assert get_tp_group() is not singleton


def test_swapped_tp_group_restores(dist_env):
    singleton = get_or_init_singleton_tp_group()
    real_tp = get_tp_group()
    assert get_tensor_model_parallel_world_size() == 1
    with swapped_tp_group(singleton):
        # The gating condition parallel layers read at construction time.
        assert get_tp_group() is singleton
        assert get_tensor_model_parallel_world_size() == 1
    assert get_tp_group() is real_tp


def test_swapped_tp_group_restores_on_error(dist_env):
    singleton = get_or_init_singleton_tp_group()
    real_tp = get_tp_group()
    with pytest.raises(RuntimeError, match="boom"), swapped_tp_group(singleton):
        raise RuntimeError("boom")
    assert get_tp_group() is real_tp
