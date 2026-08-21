# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import logging
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING, Any
import httpx
import msgspec
import numpy as np
import torch
import zmq
import zmq.asyncio
from vllm import envs
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.utils import (
    EngineId,
    TransferTopology,
    get_current_attn_backends,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorStats
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_utils import (
    MooncakeBootstrapServer,
    RegisterWorkerPayload,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.stats import (
    MooncakeKVConnectorStats,
)
from vllm.distributed.parallel_state import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.forward_context import ForwardContext
from vllm.logger import init_logger
from vllm.model_executor.models.utils import extract_layer_index
from vllm.platforms import current_platform
from vllm.utils.math_utils import cdiv
from vllm.utils.network_utils import get_ip, make_zmq_path, make_zmq_socket
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID, get_kv_cache_layout
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheSpec,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    SlidingWindowSpec,
)
from vllm.v1.request import RequestStatus
from vllm.v1.worker.block_table import BlockTable
from vllm.v1.worker.utils import select_common_block_size


logger = init_logger(__name__)

from ._protocol import (
    MooncakeXferMetadata, MooncakeXferResponse,
    MooncakeXferResponseStatus, PullReqMeta, ReqId, SendBlockMeta,
    TransferId, group_concurrent_contiguous,
    get_mooncake_bootstrap_addr,
)

from ._transfer_planning import (
    _align_transfer_regions,
    _can_coalesce_block_transfers,
    _validate_asymmetric_region_lengths,
)

from .mooncake_utils import RegisterWorkerPayload

class _WorkerSendMixin:
    async def register_worker_with_bootstrap(self):
            host, port = get_mooncake_bootstrap_addr(self.vllm_config)
            url = make_zmq_path("http", host, port) + "/register"
            worker_addr = make_zmq_path("tcp", self.hostname, self.side_channel_port)
            payload = RegisterWorkerPayload(
                engine_id=self.engine_id,
                dp_rank=self.dp_rank,
                tp_rank=self.tp_rank,
                pp_rank=self.pp_rank,
                addr=worker_addr,
            )
            while True:
                try:
                    async with httpx.AsyncClient() as client:
                        response = await client.post(url, json=payload.model_dump())
                        response.raise_for_status()
                    logger.debug("Successfully registered with bootstrap server at %s", url)
                    break
                except httpx.ConnectError:
                    # Bootstrap server not ready, wait for a while and retry.
                    await asyncio.sleep(1)
                except Exception as e:
                    err_msg = (
                        e.response.text if isinstance(e, httpx.HTTPStatusError) else str(e)
                    )
                    logger.error(
                        "Error registering %s with bootstrap server: %s", payload, err_msg
                    )
                    raise e
    async def _mooncake_sender_listener(self, ready_event: threading.Event):
            """
            Background thread that listens for Mooncake requests, dispatches them
            to a thread pool, and sends acknowledgments upon completion.
            """
    
            sock = self.async_zmq_ctx.socket(zmq.ROUTER)
            self.side_channel_port = sock.bind_to_random_port(f"tcp://{self.hostname}")
            logger.debug(
                "Mooncake sender starting listening on path: tcp://%s:%d",
                self.hostname,
                self.side_channel_port,
            )
    
            await self.register_worker_with_bootstrap()
    
            # Create async worker tasks that process items from the queue
            sender_tasks = [
                asyncio.create_task(self._sender_worker(sock))
                for _ in range(self.num_sender_tasks)
            ]
    
            ready_event.set()
    
            try:
                while True:
                    identity, metadata_bytes = await sock.recv_multipart()
                    await self.sender_worker_queue.put((identity, metadata_bytes))
            except zmq.ContextTerminated:
                logger.debug("ZMQ context terminated, exiting Mooncake sender thread.")
            except Exception as e:
                logger.error("Error in Mooncake sender thread: %s. Exiting thread.", str(e))
            finally:
                # Clean up worker tasks
                for task in sender_tasks:
                    task.cancel()
                await asyncio.gather(*sender_tasks, return_exceptions=True)
                sock.close()
    async def _sender_worker(self, sock: zmq.asyncio.Socket):
            while True:
                try:
                    identity, metadata_bytes = await self.sender_worker_queue.get()
                    try:
                        metadata = self._xfer_meta_decoder.decode(metadata_bytes)
                        await self.send_kv_to_decode(identity, sock, metadata)
                    except Exception as e:
                        logger.error("Error processing Mooncake xfer request: %s", e)
                        error_response = MooncakeXferResponse(
                            status=MooncakeXferResponseStatus.ERROR, err_msg=str(e)
                        )
                        await sock.send_multipart(
                            (identity, self._encoder.encode(error_response))
                        )
                    finally:
                        self.sender_worker_queue.task_done()
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error("Error in _sender_worker: %s", e)
    async def send_kv_to_decode(
            self, identity: bytes, sock: zmq.asyncio.Socket, meta: MooncakeXferMetadata
        ):
            pending_reqs: dict[ReqId, SendBlockMeta] = {}
            remote_tp_ranks = self.transfer_topo.handshake_target_ranks(meta.remote_tp_size)
            if meta.remote_tp_rank not in remote_tp_ranks:
                # This D worker does not pair with the P worker.
                msg = (
                    "This D tp_rank "
                    f"{meta.remote_tp_rank} is not paired with P tp_rank "
                    f"{self.tp_rank}; expected one of {remote_tp_ranks}."
                )
                logger.error(msg)
                response = MooncakeXferResponse(
                    status=MooncakeXferResponseStatus.ERROR,
                    err_msg=msg,
                )
                await sock.send_multipart((identity, self._encoder.encode(response)))
                return
            local_regions = self._get_transfer_regions(
                self.kv_caches_base_addr,
                self.block_len_per_layer,
                self.kv_block_len_per_layer,
                self.registered_layer_names,
                self.registered_layer_indices,
                self.registered_group_indices,
            )
            remote_regions = self._get_transfer_regions(
                meta.kv_caches_base_addr,
                meta.block_lens,
                meta.kv_block_lens,
                meta.registered_layer_names,
                meta.registered_layer_indices,
                meta.registered_group_indices,
            )
            local_regions, remote_regions, align_err = _align_transfer_regions(
                local_regions, remote_regions
            )
            if align_err is not None:
                response = MooncakeXferResponse(
                    status=MooncakeXferResponseStatus.ERROR,
                    err_msg=align_err,
                )
                await sock.send_multipart((identity, self._encoder.encode(response)))
                return
            validation_err = _validate_asymmetric_region_lengths(
                local_regions=local_regions,
                remote_regions=remote_regions,
                local_tp_size=self.tp_size,
                remote_tp_size=meta.remote_tp_size,
                producer_cache_replicated=self._producer_cache_is_replicated(),
            )
            if validation_err is not None:
                response = MooncakeXferResponse(
                    status=MooncakeXferResponseStatus.ERROR,
                    err_msg=validation_err,
                )
                await sock.send_multipart((identity, self._encoder.encode(response)))
                return
            for d_req_id, (transfer_id, _) in meta.req_blocks.items():
                if transfer_id not in self.reqs_need_send:
                    # This req is not enqueued in P side yet, create it here.
                    self.reqs_need_send[transfer_id] = SendBlockMeta(
                        p_req_id="",
                        transfer_id=transfer_id,
                        local_block_ids=[],
                        ready=asyncio.Event(),
                    )
                send_meta = self.reqs_need_send[transfer_id]
                pending_reqs[d_req_id] = send_meta
    
            async def wait_and_ret(
                d_req_id: ReqId, send_meta: SendBlockMeta
            ) -> tuple[ReqId, SendBlockMeta]:
                await send_meta.ready.wait()
                return d_req_id, send_meta
    
            wait_tasks = [
                asyncio.create_task(wait_and_ret(d_req_id, send_meta))
                for d_req_id, send_meta in pending_reqs.items()
            ]
    
            while wait_tasks:
                done, pending = await asyncio.wait(
                    wait_tasks,
                    timeout=envs.VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT,
                    return_when=asyncio.FIRST_COMPLETED,
                )
    
                if not done:
                    # Timeout, abort all pending requests.
                    for task in wait_tasks:
                        task.cancel()
                    logger.warning(
                        "Timeout waiting for P side ready: %s", list(pending_reqs)
                    )
                    response = MooncakeXferResponse(
                        status=MooncakeXferResponseStatus.FINISH,
                        err_reqs=list(pending_reqs),
                        err_msg="Timeout waiting for P side ready.",
                    )
                    await sock.send_multipart((identity, self._encoder.encode(response)))
                    break
    
                wait_tasks = list(pending)
                response_status = (
                    MooncakeXferResponseStatus.CONTINUE
                    if wait_tasks
                    else MooncakeXferResponseStatus.FINISH
                )
                ready_reqs: list[tuple[ReqId, SendBlockMeta]] = []
                for task in done:
                    d_req_id, send_meta = task.result()
                    del pending_reqs[d_req_id]
                    # Do we still in reqs_need_send (not expired)?
                    if send_meta.transfer_id in self.reqs_need_send:
                        # Mark it sending to avoid expiration.
                        send_meta.sending += 1
                        if not send_meta.need_send:
                            self.resolve_need_send(send_meta, remote_tp_ranks)
                        ready_reqs.append((d_req_id, send_meta))
                    else:
                        # Otherwise (expired, very unlikely), just forget it.
                        logger.warning(
                            "Request %s expired before sending on P side.", d_req_id
                        )
    
                (
                    src_ptrs,
                    dst_ptrs,
                    lengths,
                    err_reqs,
                    err_msg,
                ) = await self._build_transfer_params(
                    ready_reqs,
                    meta,
                    local_regions,
                    remote_regions,
                )
                err_req_set = set(err_reqs)
                ok_ready_reqs = [
                    (d_req_id, send_meta)
                    for d_req_id, send_meta in ready_reqs
                    if d_req_id not in err_req_set
                ]
    
                if src_ptrs:
                    remote_session = f"{meta.remote_hostname}:{meta.remote_port}"
                    ret_value = await self.sender_loop.run_in_executor(
                        self._sender_executor,
                        self._send_blocks,
                        remote_session,
                        src_ptrs,
                        dst_ptrs,
                        lengths,
                    )
    
                    if ret_value != 0:
                        transfer_err_msg = f"Mooncake transfer engine returned {ret_value}"
                        err_msg = (
                            transfer_err_msg
                            if err_msg is None
                            else f"{err_msg}; {transfer_err_msg}"
                        )
                        err_reqs = list(err_reqs)
                        for d_req_id, _ in ok_ready_reqs:
                            err_reqs.append(d_req_id)
                            err_req_set.add(d_req_id)
                        ok_ready_reqs = []
    
                for d_req_id, send_meta in ready_reqs:
                    send_meta.sending -= 1
    
                    if d_req_id in err_req_set:
                        continue
    
                    send_meta.sent += 1
                    if (
                        send_meta.sent == send_meta.need_send
                        and self.reqs_need_send.pop(send_meta.transfer_id, None) is not None
                    ):
                        self.finished_sending_reqs.add(send_meta.p_req_id)
    
                response = MooncakeXferResponse(
                    status=response_status,
                    ok_reqs=[d_req_id for d_req_id, _ in ok_ready_reqs] or None,
                    err_reqs=err_reqs or None,
                    err_msg=err_msg,
                )
                await sock.send_multipart((identity, self._encoder.encode(response)))
    def resolve_need_send(
            self,
            send_meta: SendBlockMeta,
            remote_tp_ranks: list[int],
        ):
            # Prepare for heterogeneous TP (one P pairs to multiple D)
            send_meta.need_send = len(remote_tp_ranks)
            logger.debug(
                "Mooncake request %s will be served by %d consumer TP workers: TP ranks=%s",
                send_meta.transfer_id,
                send_meta.need_send,
                remote_tp_ranks,
            )
    async def _build_transfer_params(
            self,
            ready_reqs: list[tuple[ReqId, SendBlockMeta]],
            agent_meta: MooncakeXferMetadata,
            local_regions: list[TransferRegion],
            remote_regions: list[TransferRegion],
        ) -> tuple[list[int], list[int], list[int], list[ReqId], str | None]:
            src_ptrs = []
            dst_ptrs = []
            lengths = []
            err_reqs: list[ReqId] = []
            err_msg: str | None = None
            remote_session = f"{agent_meta.remote_hostname}:{agent_meta.remote_port}"
    
            for d_req_id, send_meta in ready_reqs:
                _, remote_block_ids_per_group = agent_meta.req_blocks[d_req_id]
    
                if not remote_block_ids_per_group or all(
                    len(g) == 0 for g in remote_block_ids_per_group
                ):
                    continue
    
                if len(send_meta.local_block_ids) != len(remote_block_ids_per_group):
                    logger.error(
                        "req %s: KV group count mismatch: local=%d, remote=%d",
                        d_req_id,
                        len(send_meta.local_block_ids),
                        len(remote_block_ids_per_group),
                    )
                    err_reqs.append(d_req_id)
                    if err_msg is None:
                        err_msg = "KV group count mismatch"
                    continue
    
                # Keep KV-cache group identity. Hybrid/HMA groups can carry
                # different semantics (e.g. full-attention KV pages vs GDN/Mamba
                # inner-state slots), so their block IDs must not be flattened and
                # reused for every registered region.
                local_block_ids_by_group: list[list[int]] = []
                remote_block_ids_by_group: list[list[int]] = []
                has_block_error = False
                group_specs = self.kv_cache_config.kv_cache_groups
                for group_index, (local_group, remote_group) in enumerate(
                    zip(send_meta.local_block_ids, remote_block_ids_per_group)
                ):
                    is_mamba_group = isinstance(
                        group_specs[group_index].kv_cache_spec,
                        MambaSpec,
                    )
                    if is_mamba_group:
                        # Mamba/GDN prefix caching can use null blocks only as
                        # align-mode placeholders. They do not carry transferable
                        # state, so skip them on both producer and consumer sides.
                        local_group = [
                            block_id
                            for block_id in local_group
                            if block_id != NULL_BLOCK_ID
                        ]
                        remote_group = [
                            block_id
                            for block_id in remote_group
                            if block_id != NULL_BLOCK_ID
                        ]
    
                    n_local = len(local_group)
                    n_remote = len(remote_group)
                    if n_local < n_remote:
                        logger.error(
                            "req %s: local blocks(%d) < remote blocks(%d) "
                            "in a KV cache group (is_mamba_group=%s)",
                            d_req_id,
                            n_local,
                            n_remote,
                            is_mamba_group,
                        )
                        has_block_error = True
                        break
                    elif n_local > n_remote:
                        # Partial prefix cache hit: just read uncomputed blocks.
                        local_group = local_group[-n_remote:] if n_remote > 0 else []
                    local_block_ids_by_group.append(local_group)
                    remote_block_ids_by_group.append(remote_group)
    
                if has_block_error:
                    err_reqs.append(d_req_id)
                    if err_msg is None:
                        err_msg = "P num blocks less than D"
                    continue
    
                if not any(local_block_ids_by_group):
                    continue
    
                local_block_ids_by_group = self._logical_to_kernel_block_ids(
                    local_block_ids_by_group
                )
                remote_block_ids_by_group = self._logical_to_kernel_block_ids(
                    remote_block_ids_by_group
                )
    
                for local_region, remote_region in zip(local_regions, remote_regions):
                    assert local_region.group_index == remote_region.group_index, (
                        "Aligned Mooncake transfer regions must belong to the same "
                        "KV group."
                    )
                    group_index = local_region.group_index
                    assert group_index < len(local_block_ids_by_group), (
                        "Transfer region references a missing KV group."
                    )
                    local_block_ids = local_block_ids_by_group[group_index]
                    remote_block_ids = remote_block_ids_by_group[group_index]
                    if not local_block_ids:
                        continue
    
                    # Group by indices within this region's KV-cache group only.
                    group_local_block_ids, group_remote_block_ids = (
                        group_concurrent_contiguous(local_block_ids, remote_block_ids)
                    )
                    (
                        should_transfer,
                        src_region_offset,
                        dst_region_offset,
                        transfer_len,
                    ) = self._get_sender_transfer_plan(
                        local_kv_block_len=local_region.kv_block_len,
                        remote_kv_block_len=remote_region.kv_block_len,
                        remote_tp_rank=agent_meta.remote_tp_rank,
                        remote_tp_size=agent_meta.remote_tp_size,
                    )
                    if not should_transfer:
                        # Replicated KV cache: only one producer rank in the TP group
                        # needs to send the actual bytes for this paired decoder rank.
                        # TODO: Account for replicated producer KV in
                        # get_target_remote_ranks() so we can avoid sending
                        # unnecessary ZMQ requests and remove this branch.
                        continue
    
                    assert src_region_offset + transfer_len <= local_region.kv_block_len, (
                        "Computed source transfer region exceeds local KV block size."
                    )
                    assert dst_region_offset + transfer_len <= remote_region.kv_block_len, (
                        "Destination transfer region exceeds remote KV block size."
                    )
                    # Collapse one contiguous block group into a single larger
                    # transfer descriptor when the per-block copy is identical.
                    can_coalesce = _can_coalesce_block_transfers(
                        local_region_block_len=local_region.block_len,
                        remote_region_block_len=remote_region.block_len,
                        src_region_offset=src_region_offset,
                        dst_region_offset=dst_region_offset,
                        transfer_len=transfer_len,
                    )
    
                    for group_local_block_id, group_remote_block_id in zip(
                        group_local_block_ids, group_remote_block_ids
                    ):
                        if can_coalesce:
                            src_ptrs.append(
                                local_region.base_addr
                                + group_local_block_id[0] * local_region.block_len
                                + src_region_offset
                            )
                            dst_ptrs.append(
                                remote_region.base_addr
                                + group_remote_block_id[0] * remote_region.block_len
                                + dst_region_offset
                            )
                            lengths.append(transfer_len * len(group_local_block_id))
                        else:
                            for local_block_id, remote_block_id in zip(
                                group_local_block_id, group_remote_block_id
                            ):
                                src_ptrs.append(
                                    local_region.base_addr
                                    + local_block_id * local_region.block_len
                                    + src_region_offset
                                )
                                dst_ptrs.append(
                                    remote_region.base_addr
                                    + remote_block_id * remote_region.block_len
                                    + dst_region_offset
                                )
                                lengths.append(transfer_len)
    
                logger.debug(
                    "Sending kv_caches for request %s (%d blocks) to %s",
                    d_req_id,
                    sum(len(group) for group in local_block_ids_by_group),
                    remote_session,
                )
    
            return src_ptrs, dst_ptrs, lengths, err_reqs, err_msg
    def _bind_sender_thread_device(self) -> None:
            """ThreadPoolExecutor initializer — binds each pool thread to the
            correct CUDA device.  CUDA device selection is thread-local, so
            without this, NVLink transfers fail for TP ranks > 0."""
            current_platform.set_device(self.device_id)
    def _send_blocks(
            self,
            remote_session: str,
            src_ptrs: list[int],
            dst_ptrs: list[int],
            lengths: list[int],
        ) -> int:
            if self.host_staging:
                return self._send_blocks_staged(
                    remote_session, src_ptrs, dst_ptrs, lengths
                )
            start_time = time.perf_counter()
            ret_value = self.engine.batch_transfer_sync_write(
                remote_session, src_ptrs, dst_ptrs, lengths
            )
            duration = time.perf_counter() - start_time
            if ret_value == 0:
                self.xfer_stats.record_transfer(
                    duration_s=duration,
                    total_bytes=sum(lengths),
                    num_descs=len(src_ptrs),
                )
                logger.debug("Sending to %s done, took %s", remote_session, duration)
            else:
                self.xfer_stats.record_failed_transfer()
                logger.warning(
                    "Sending to %s failed (ret=%s) after %s (%d descriptors, %d bytes)",
                    remote_session,
                    ret_value,
                    duration,
                    len(src_ptrs),
                    sum(lengths),
                )
            return ret_value
    async def record_send_reqs(self, metadata: MooncakeConnectorMetadata):
            for p_req_id, (transfer_id, block_ids) in metadata.reqs_to_send.items():
                if block_ids:
                    # Already gone through request_finished()
                    send_meta = self.reqs_need_send[transfer_id]
                    send_meta.p_req_id = p_req_id
                    send_meta.local_block_ids = block_ids
                    send_meta.expire_time = (
                        time.perf_counter() + envs.VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT
                    )
                    send_meta.ready.set()
                else:
                    # From update_state_after_alloc(),
                    # but not reach request_finished() yet
                    # This may be already created by send_kv_to_decode()
                    # when D is sending MooncakeXferMetadata.
                    if transfer_id not in self.reqs_need_send:
                        self.reqs_need_send[transfer_id] = SendBlockMeta(
                            p_req_id=p_req_id,
                            transfer_id=transfer_id,
                            local_block_ids=[],
                            ready=asyncio.Event(),
                        )
            for transfer_id in metadata.reqs_not_processed:
                send_meta = self.reqs_need_send.pop(transfer_id)
                if send_meta:
                    assert not send_meta.ready.is_set()
