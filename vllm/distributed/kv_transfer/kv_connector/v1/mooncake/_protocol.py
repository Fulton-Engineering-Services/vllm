# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Protocol data classes and utility functions for the Mooncake KV connector.

This module has zero imports from the rest of the mooncake package and
zero Worker-class dependencies — pure data containers and free functions.
"""
import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING

import msgspec
import numpy as np
import torch

from vllm import envs
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.utils import EngineId
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.logger import init_logger

if TYPE_CHECKING:
    pass

logger = init_logger(__name__)

ReqId = str
TransferId = str

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TransferRegion:
    layer_name: str
    layer_index: int
    base_addr: int
    block_len: int
    kv_block_len: int
    group_index: int = 0


class MooncakeXferResponseStatus(IntEnum):
    FINISH = 0
    CONTINUE = 1
    ERROR = 2


class MooncakeXferMetadata(
    msgspec.Struct,
    omit_defaults=True,  # type: ignore[call-arg]
):
    remote_hostname: str
    remote_port: int
    remote_tp_size: int
    remote_tp_rank: int
    req_blocks: dict[ReqId, tuple[TransferId, list[list[int]]]]
    kv_caches_base_addr: list[int]
    block_lens: list[int]
    kv_block_lens: list[int]
    registered_layer_names: list[str] = msgspec.field(default_factory=list)
    registered_layer_indices: list[int] = msgspec.field(default_factory=list)
    registered_group_indices: list[int] = msgspec.field(default_factory=list)


class MooncakeXferResponse(
    msgspec.Struct,
    omit_defaults=True,  # type: ignore[call-arg]
):
    status: MooncakeXferResponseStatus
    ok_reqs: list[ReqId] | None = None
    err_reqs: list[ReqId] | None = None
    err_msg: str | None = None


@dataclass
class PullReqMeta:
    d_req_id: ReqId
    transfer_id: TransferId
    local_block_ids: list[list[int]]
    remote_engine_id: EngineId
    remote_bootstrap_addr: str
    expire_time: float = float("inf")
    pull_tasks_count: int = 0
    staging_slots: dict[int, tuple[int, list[int]]] | None = None


@dataclass
class SendBlockMeta:
    p_req_id: ReqId
    transfer_id: TransferId
    local_block_ids: list[list[int]]
    ready: asyncio.Event
    expire_time: float = float("inf")
    need_send: int = 0
    sent: int = 0
    sending: int = 0


class MooncakeConnectorMetadata(KVConnectorMetadata):
    def __init__(self):
        super().__init__()
        self.reqs_to_recv: dict[EngineId, dict[ReqId, PullReqMeta]] = defaultdict(dict)
        self.reqs_to_send: dict[ReqId, tuple[TransferId, list[list[int]]]] = {}
        self.reqs_not_processed: set[TransferId] = set()

    def add_new_req(
        self,
        request_id: ReqId,
        local_block_ids: list[list[int]],
        kv_transfer_params: dict,
        load_remote_cache: bool = True,
    ):
        transfer_id: TransferId = kv_transfer_params["transfer_id"]
        if load_remote_cache:
            remote_engine_id: EngineId = kv_transfer_params["remote_engine_id"]
            self.reqs_to_recv[remote_engine_id][request_id] = PullReqMeta(
                d_req_id=request_id,
                local_block_ids=local_block_ids,
                remote_engine_id=remote_engine_id,
                remote_bootstrap_addr=kv_transfer_params["remote_bootstrap_addr"],
                transfer_id=transfer_id,
            )
        else:
            self.reqs_to_send[request_id] = (transfer_id, local_block_ids)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def _get_tensor_dense_flag(tensor: torch.Tensor) -> bool | None:
    is_dense = getattr(tensor, "is_non_overlapping_and_dense", None)
    if callable(is_dense):
        return bool(is_dense())
    return None


def group_concurrent_contiguous(
    src_indices: list[int], dst_indices: list[int]
) -> tuple[list[list[int]], list[list[int]]]:
    if len(src_indices) == 0:
        return [], []
    brk = np.where((np.diff(src_indices) != 1) | (np.diff(dst_indices) != 1))[0] + 1
    src_groups = np.split(src_indices, brk)
    dst_groups = np.split(dst_indices, brk)
    src_groups = [g.tolist() for g in src_groups]
    dst_groups = [g.tolist() for g in dst_groups]
    return src_groups, dst_groups


def get_mooncake_side_channel_port(vllm_config: VllmConfig) -> int:
    return (
        envs.VLLM_MOONCAKE_BOOTSTRAP_PORT
        + vllm_config.parallel_config.data_parallel_index
        * vllm_config.parallel_config.tensor_parallel_size
    )


def _async_loop(loop: asyncio.AbstractEventLoop):
    asyncio.set_event_loop(loop)
    loop.run_forever()


def should_launch_bootstrap_server(vllm_config: VllmConfig) -> bool:
    from vllm.distributed.parallel_state import (
        get_pp_group,
        get_tensor_model_parallel_rank,
    )
    assert (parallel_config := vllm_config.parallel_config)
    if get_tensor_model_parallel_rank() != 0:
        return False
    if get_pp_group().rank_in_group != 0:
        return False
    if parallel_config.local_engines_only:
        return parallel_config.data_parallel_rank_local == 0
    return parallel_config.data_parallel_index == 0


def get_mooncake_bootstrap_addr(vllm_config: VllmConfig) -> tuple[str, int]:
    assert (parallel_config := vllm_config.parallel_config)
    if parallel_config.local_engines_only:
        host = "127.0.0.1"
    elif parallel_config.nnodes_within_dp > 1:
        host = parallel_config.master_addr
    else:
        host = parallel_config.data_parallel_master_ip
    port = envs.VLLM_MOONCAKE_BOOTSTRAP_PORT
    return (host, port)