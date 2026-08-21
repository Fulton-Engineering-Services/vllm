# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure functions for Mooncake KV transfer geometry planning.

Zero worker-class dependencies — importable at module level by every other
module in the package without circularity.
"""
from collections import defaultdict

from ._protocol import TransferRegion


def _get_tp_ratio(local_tp_size: int, remote_tp_size: int) -> int:
    if local_tp_size >= remote_tp_size:
        assert local_tp_size % remote_tp_size == 0, (
            f"Local tensor parallel size {local_tp_size} is not divisible "
            f"by remote tensor parallel size {remote_tp_size}."
        )
        return local_tp_size // remote_tp_size
    assert remote_tp_size % local_tp_size == 0, (
        f"Remote tensor parallel size {remote_tp_size} is not divisible "
        f"by local tensor parallel size {local_tp_size}."
    )
    return -(remote_tp_size // local_tp_size)


def _expand_transfer_regions(
    base_addrs: list[int],
    block_lens: list[int],
    kv_block_lens: list[int],
    layer_names: list[str],
    layer_indices: list[int],
    is_kv_layout_blocks_first: bool,
    group_indices: list[int] | None = None,
    split_kv_regions: list[bool] | None = None,
) -> list[TransferRegion]:
    assert (
        len(base_addrs)
        == len(block_lens)
        == len(kv_block_lens)
        == len(layer_names)
        == len(layer_indices)
    ), (
        "Mooncake transfer regions require matching metadata lengths, got "
        f"base_addrs={len(base_addrs)}, block_lens={len(block_lens)}, "
        f"kv_block_lens={len(kv_block_lens)}, "
        f"layer_names={len(layer_names)}, "
        f"layer_indices={len(layer_indices)}."
    )
    if group_indices is None:
        group_indices = [0] * len(layer_names)
    assert len(group_indices) == len(layer_names), (
        "Mooncake transfer regions require matching group metadata lengths, "
        f"got group_indices={len(group_indices)}, layer_names={len(layer_names)}."
    )
    if split_kv_regions is None:
        split_kv_regions = [is_kv_layout_blocks_first] * len(layer_names)
    assert len(split_kv_regions) == len(layer_names), (
        "Mooncake transfer regions require matching split metadata, "
        f"got split_kv_regions={len(split_kv_regions)}, "
        f"layer_names={len(layer_names)}."
    )
    regions: list[TransferRegion] = []
    for (
        base_addr,
        block_len,
        kv_block_len,
        layer_name,
        layer_index,
        group_index,
        split_kv_region,
    ) in zip(
        base_addrs,
        block_lens,
        kv_block_lens,
        layer_names,
        layer_indices,
        group_indices,
        split_kv_regions,
    ):
        regions.append(
            TransferRegion(
                layer_name=layer_name,
                layer_index=layer_index,
                base_addr=base_addr,
                block_len=block_len,
                kv_block_len=kv_block_len,
                group_index=group_index,
            )
        )
        if split_kv_region:
            regions.append(
                TransferRegion(
                    layer_name=layer_name,
                    layer_index=layer_index,
                    base_addr=base_addr + kv_block_len,
                    block_len=block_len,
                    kv_block_len=kv_block_len,
                    group_index=group_index,
                )
            )
    return regions


def _compute_sender_transfer_plan(
    local_tp_rank: int,
    local_tp_size: int,
    remote_tp_rank: int,
    remote_tp_size: int,
    local_kv_block_len: int,
    remote_kv_block_len: int,
    producer_cache_replicated: bool,
) -> tuple[bool, int, int, int]:
    tp_ratio = _get_tp_ratio(local_tp_size, remote_tp_size)
    if tp_ratio == 1:
        return True, 0, 0, local_kv_block_len
    if tp_ratio > 0:
        if producer_cache_replicated:
            return local_tp_rank % tp_ratio == 0, 0, 0, local_kv_block_len
        return (
            True,
            0,
            (local_tp_rank % tp_ratio) * local_kv_block_len,
            local_kv_block_len,
        )
    if producer_cache_replicated:
        return True, 0, 0, local_kv_block_len
    ratio_abs = -tp_ratio
    return (
        True,
        (remote_tp_rank % ratio_abs) * remote_kv_block_len,
        0,
        remote_kv_block_len,
    )


def _can_coalesce_block_transfers(
    local_region_block_len: int,
    remote_region_block_len: int,
    src_region_offset: int,
    dst_region_offset: int,
    transfer_len: int,
) -> bool:
    return (
        src_region_offset == 0
        and dst_region_offset == 0
        and transfer_len == local_region_block_len
        and transfer_len == remote_region_block_len
    )


def _validate_asymmetric_region_lengths(
    local_regions: list[TransferRegion],
    remote_regions: list[TransferRegion],
    local_tp_size: int,
    remote_tp_size: int,
    producer_cache_replicated: bool,
) -> str | None:
    if len(local_regions) != len(remote_regions):
        return (
            "Mooncake asymmetric TP requires matching KV region counts between "
            "producer and consumer."
        )
    if producer_cache_replicated:
        return None
    tp_ratio = _get_tp_ratio(local_tp_size, remote_tp_size)
    for idx, (local_region, remote_region) in enumerate(
        zip(local_regions, remote_regions)
    ):
        if tp_ratio == 1:
            if local_region.kv_block_len != remote_region.kv_block_len:
                return (
                    "Mooncake KV region length mismatch for homogeneous TP at "
                    f"region {idx}: local={local_region.kv_block_len}, "
                    f"remote={remote_region.kv_block_len}."
                )
        elif tp_ratio > 0:
            if remote_region.kv_block_len != local_region.kv_block_len * tp_ratio:
                return (
                    "Mooncake destination KV region length does not match the "
                    "producer TP ratio at region "
                    f"{idx}: local={local_region.kv_block_len}, "
                    f"remote={remote_region.kv_block_len}, tp_ratio={tp_ratio}."
                )
        else:
            ratio_abs = -tp_ratio
            if local_region.kv_block_len != remote_region.kv_block_len * ratio_abs:
                return (
                    "Mooncake source KV region length does not match the "
                    "consumer TP ratio at region "
                    f"{idx}: local={local_region.kv_block_len}, "
                    f"remote={remote_region.kv_block_len}, tp_ratio={tp_ratio}."
                )
    return None


def _align_transfer_regions(
    local_regions: list[TransferRegion],
    remote_regions: list[TransferRegion],
) -> tuple[list[TransferRegion], list[TransferRegion], str | None]:

    def keyed_regions(
        regions: list[TransferRegion],
    ) -> list[tuple[tuple[str, int], TransferRegion]]:
        counts: dict[str, int] = defaultdict(int)
        keyed: list[tuple[tuple[str, int], TransferRegion]] = []
        for region in regions:
            occurrence = counts[region.layer_name]
            counts[region.layer_name] += 1
            keyed.append(((region.layer_name, occurrence), region))
        return keyed

    local_keyed = keyed_regions(local_regions)
    remote_keyed = keyed_regions(remote_regions)
    remote_by_key = dict(remote_keyed)
    aligned_local: list[TransferRegion] = []
    aligned_remote: list[TransferRegion] = []
    for key, local_region in local_keyed:
        remote_region = remote_by_key.get(key)
        if remote_region is None:
            return (
                [],
                [],
                (
                    "Mooncake producer registered layer has no matching "
                    f"consumer occurrence: {key[0]} occurrence {key[1]}."
                ),
            )
        if local_region.layer_index != remote_region.layer_index:
            return (
                [],
                [],
                (
                    "Mooncake registered layer index mismatch for "
                    f"{local_region.layer_name}: producer="
                    f"{local_region.layer_index}, consumer="
                    f"{remote_region.layer_index}."
                ),
            )
        if local_region.group_index != remote_region.group_index:
            return (
                [],
                [],
                (
                    "Mooncake registered group index mismatch for "
                    f"{local_region.layer_name}: producer="
                    f"{local_region.group_index}, consumer="
                    f"{remote_region.group_index}."
                ),
            )
        aligned_local.append(local_region)
        aligned_remote.append(remote_region)
    return aligned_local, aligned_remote, None