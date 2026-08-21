# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Send-side protocol mixin for MooncakeConnectorWorker.

All methods here assume `self` is a MooncakeConnectorWorker (or compatible)
and access worker attributes directly.  Imported by worker.py, never
imported by _worker_receive or _worker_setup to keep the graph acyclic.
"""
import asyncio
import logging
import threading
import time
from typing import TYPE_CHECKING

import httpx
import numpy as np
import zmq
import zmq.asyncio

from vllm import envs
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.network_utils import make_zmq_path, make_zmq_socket

from ._protocol import (
    EngineId,
    MooncakeXferMetadata,
    MooncakeXferResponse,
    MooncakeXferResponseStatus,
    ReqId,
    SendBlockMeta,
    TransferId,
)
from ._transfer_planning import (
    _align_transfer_regions,
    _can_coalesce_block_transfers,
    _validate_asymmetric_region_lengths,
)
from .mooncake_utils import RegisterWorkerPayload

if TYPE_CHECKING:
    from ._protocol import MooncakeConnectorMetadata

logger = init_logger(__name__)


class _WorkerSendMixin:
    """All send-side protocol logic for MooncakeConnectorWorker."""

    # ── thread binding ─────────────────────────────────────────────────

    def _bind_sender_thread_device(self) -> None:
        current_platform.set_device(self.device_id)

    # ── resolve send targets ───────────────────────────────────────────

    def resolve_need_send(
        self,
        send_meta: SendBlockMeta,
        remote_tp_ranks: list[int],
    ):
        send_meta.need_send = len(remote_tp_ranks)
        logger.debug(
            "Mooncake request %s will be served by %d consumer TP workers: "
            "TP ranks=%s",
            send_meta.transfer_id,
            send_meta.need_send,
            remote_tp_ranks,
        )

    # ── transfer descriptor construction ───────────────────────────────

    async def _build_transfer_params(
        self,
        ready_reqs: list[tuple[ReqId, SendBlockMeta]],
        agent_meta: MooncakeXferMetadata,
        local_regions,
        remote_regions,
    ):
        src_ptrs = []
        dst_ptrs = []
        lengths = []
        err_reqs: list[ReqId] = []
        err_msg: str | None = None
        remote_session = f"{agent_meta.remote_hostname}:{agent_meta.remote_port}"
        from vllm.v1.kv_cache_interface import MambaSpec
        from vllm.v1.attention.backends.utils import NULL_BLOCK_ID
        from ._protocol import group_concurrent_contiguous

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

            local_block_ids_by_group: list[list[int]] = []
            remote_block_ids_by_group: list[list[int]] = []
            has_block_error = False
            group_specs = self.kv_cache_config.kv_cache_groups
            for group_index, (local_group, remote_group) in enumerate(
                zip(send_meta.local_block_ids, remote_block_ids_per_group)
            ):
                is_mamba_group = isinstance(
                    group_specs[group_index].kv_cache_spec, MambaSpec
                )
                if is_mamba_group:
                    local_group = [b for b in local_group if b != NULL_BLOCK_ID]
                    remote_group = [b for b in remote_group if b != NULL_BLOCK_ID]
                n_local = len(local_group)
                n_remote = len(remote_group)
                if n_local < n_remote:
                    logger.error(
                        "req %s: local blocks(%d) < remote blocks(%d) "
                        "in a KV cache group (is_mamba_group=%s)",
                        d_req_id, n_local, n_remote, is_mamba_group,
                    )
                    has_block_error = True
                    break
                elif n_local > n_remote:
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
                assert local_region.group_index == remote_region.group_index
                group_index = local_region.group_index
                assert group_index < len(local_block_ids_by_group)
                local_block_ids = local_block_ids_by_group[group_index]
                remote_block_ids = remote_block_ids_by_group[group_index]
                if not local_block_ids:
                    continue
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
                    continue
                assert src_region_offset + transfer_len <= local_region.kv_block_len
                assert dst_region_offset + transfer_len <= remote_region.kv_block_len
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

    # ── send execution ─────────────────────────────────────────────────

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
                remote_session, ret_value, duration, len(src_ptrs), sum(lengths),
            )
        return ret_value

    # ── ZMQ listener loop ──────────────────────────────────────────────

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
                logger.debug(
                    "Successfully registered with bootstrap server at %s", url
                )
                break
            except httpx.ConnectError:
                await asyncio.sleep(1)
            except Exception as e:
                err_msg = (
                    e.response.text
                    if isinstance(e, httpx.HTTPStatusError)
                    else str(e)
                )
                logger.error(
                    "Error registering %s with bootstrap server: %s", payload, err_msg
                )
                raise e

    async def _mooncake_sender_listener(self, ready_event: threading.Event):
        sock = self.async_zmq_ctx.socket(zmq.ROUTER)
        self.side_channel_port = sock.bind_to_random_port(
            f"tcp://{self.hostname}"
        )
        logger.debug(
            "Mooncake sender starting listening on path: tcp://%s:%d",
            self.hostname,
            self.side_channel_port,
        )
        await self.register_worker_with_bootstrap()
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
            logger.error(
                "Error in Mooncake sender thread: %s. Exiting thread.", str(e)
            )
        finally:
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
                    logger.error(
                        "Error processing Mooncake xfer request: %s", e
                    )
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

    # ── send_kv_to_decode (main send orchestrator) ─────────────────────

    async def send_kv_to_decode(
        self,
        identity: bytes,
        sock: zmq.asyncio.Socket,
        meta: MooncakeXferMetadata,
    ):
        pending_reqs: dict[ReqId, SendBlockMeta] = {}
        remote_tp_ranks = self.transfer_topo.handshake_target_ranks(
            meta.remote_tp_size
        )
        if meta.remote_tp_rank not in remote_tp_ranks:
            msg = (
                "This D tp_rank "
                f"{meta.remote_tp_rank} is not paired with P tp_rank "
                f"{self.tp_rank}; expected one of {remote_tp_ranks}."
            )
            logger.error(msg)
            response = MooncakeXferResponse(
                status=MooncakeXferResponseStatus.ERROR, err_msg=msg,
            )
            await sock.send_multipart((identity, self._encoder.encode(response)))
            return

        from ._protocol import TransferRegion

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
        if align_err:
            logger.error(align_err)
            response = MooncakeXferResponse(
                status=MooncakeXferResponseStatus.ERROR, err_msg=align_err,
            )
            await sock.send_multipart((identity, self._encoder.encode(response)))
            return
        err = _validate_asymmetric_region_lengths(
            local_regions,
            remote_regions,
            self.tp_size,
            meta.remote_tp_size,
            self._producer_cache_is_replicated(),
        )
        if err:
            logger.error(err)
            response = MooncakeXferResponse(
                status=MooncakeXferResponseStatus.ERROR, err_msg=err,
            )
            await sock.send_multipart((identity, self._encoder.encode(response)))
            return

        for d_req_id in meta.req_blocks:
            _, block_ids = meta.req_blocks[d_req_id]
            if d_req_id in self.reqs_need_send:
                self.resolve_need_send(
                    self.reqs_need_send[d_req_id], remote_tp_ranks
                )
                self.reqs_need_send[d_req_id].local_block_ids = block_ids
            else:
                send_meta = SendBlockMeta(
                    p_req_id=d_req_id,
                    transfer_id="",
                    local_block_ids=block_ids,
                    ready=asyncio.Event(),
                    expire_time=(
                        time.perf_counter()
                        + envs.VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT
                    ),
                )
                self.resolve_need_send(send_meta, remote_tp_ranks)
                self.reqs_need_send[d_req_id] = send_meta
            pending_reqs[d_req_id] = self.reqs_need_send[d_req_id]

        await self._wait_for_block_ids(pending_reqs)
        ready_reqs = [
            (d_req_id, pending_reqs[d_req_id])
            for d_req_id in pending_reqs
            if pending_reqs[d_req_id].ready.is_set()
        ]
        for d_req_id, _ in ready_reqs:
            pending_reqs.pop(d_req_id)

        remote_session = f"{meta.remote_hostname}:{meta.remote_port}"
        src_ptrs, dst_ptrs, lengths, err_reqs, err_msg = (
            await self._build_transfer_params(
                ready_reqs, meta, local_regions, remote_regions
            )
        )
        if not ready_reqs and not err_reqs:
            err_msg = "All P side requests have been cancelled."

        if src_ptrs:
            ret_value = await asyncio.get_running_loop().run_in_executor(
                self._sender_executor,
                self._send_blocks,
                remote_session,
                src_ptrs,
                dst_ptrs,
                lengths,
            )
            if ret_value != 0:
                if err_msg:
                    err_msg += " + "
                else:
                    err_msg = ""
                err_msg += f"Transfer failed with code {ret_value}"
                err_reqs = [d_req_id for d_req_id, _ in ready_reqs]

        ok_reqs = [d_req_id for d_req_id, _ in ready_reqs]
        for p_req_id in ok_reqs:
            self.finished_sending_reqs.add(p_req_id)

        status = (
            MooncakeXferResponseStatus.ERROR
            if (err_msg or not ok_reqs)
            else MooncakeXferResponseStatus.FINISH
        )
        response = MooncakeXferResponse(
            status=status,
            ok_reqs=ok_reqs if ok_reqs else None,
            err_reqs=err_reqs if err_reqs else None,
            err_msg=err_msg or None,
        )
        logger.debug(
            "Sending MooncakeXferResponse (ok=%s, err=%s): %s",
            len(ok_reqs),
            len(err_reqs),
            response,
        )
        await sock.send_multipart((identity, self._encoder.encode(response)))
        for d_req_id in pending_reqs:
            logger.warning(
                "Request %s block ids never received on P side.", d_req_id
            )

    async def _wait_for_block_ids(self, pending_reqs: dict[ReqId, SendBlockMeta]):
        events = {
            d_req_id: send_meta.ready
            for d_req_id, send_meta in pending_reqs.items()
        }
        coros = [asyncio.ensure_future(ev.wait()) for ev in events.values()]
        done, pending = await asyncio.wait(
            coros, timeout=envs.VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT
        )
        for coro in pending:
            coro.cancel()

    # ── record / cleanup ───────────────────────────────────────────────

    async def record_send_reqs(self, metadata: "MooncakeConnectorMetadata"):
        for p_req_id, (transfer_id, block_ids) in metadata.reqs_to_send.items():
            if block_ids:
                send_meta = self.reqs_need_send[transfer_id]
                send_meta.p_req_id = p_req_id
                send_meta.local_block_ids = block_ids
                send_meta.expire_time = (
                    time.perf_counter() + envs.VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT
                )
                send_meta.ready.set()
            else:
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