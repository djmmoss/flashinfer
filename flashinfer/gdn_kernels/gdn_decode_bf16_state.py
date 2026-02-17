"""
Copyright (c) 2025 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Gated Delta Rule Decode Kernel (BF16 Hidden State) - CuTe-DSL Implementation
============================================================================

RELOCATED: This file was previously located at flashinfer/cute_dsl/gated_delta_rule.py
           and has been moved to flashinfer/gdn_decode/gdn_decode_bf16_state.py
           to better reflect its domain-specific purpose (GDN decode with BF16 state).

High-performance CUDA kernel implementing the Gated Delta Rule linear attention
mechanism for decode-phase inference, supporting sequence lengths T=1, T=2, T=3, T=4.

Key Features:
- Unified kernel architecture: T=2/3/4 share a single compile-time specialized kernel
  using Constexpr dispatch, while T=1 uses a separate kernel with persistent K-in-registers
- L2-normalized Q/K with configurable scale
- Gated exponential decay of hidden state H via softplus
- Delta rule updates: v_delta = beta * (v - pred)
- Bank-conflict-free cross-warp reductions
- Async H memory loading with aggressive pipelining
- BF16 tensors with FP32 compute for numerical stability
- GQA (grouped-query attention) support with configurable H (query) and HV (value) heads
"""

import math
from typing import Optional

import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
import torch
from cutlass import utils
from cutlass._mlir.dialects import nvvm
from cutlass.cute.runtime import from_dlpack

from flashinfer.gdn_kernels._common import (
    H_SMEM_PADDING,
    write_h_chunk_to_smem,
    store_h_smem_to_gmem,
    load_h_chunk_async,
    normalize_and_store_qk_to_smem,
    load_v_to_smem,
    load_kq_chunk_from_smem,
    update_h_with_delta,
    compute_output,
    cross_warp_reduce_single,
    cross_warp_reduce_two,
)


# ==============================================================================
# GDN-SPECIFIC FUNCTIONS (scalar gate)
# ==============================================================================


@cute.jit
def compute_single_gate(
    alpha, beta_raw, dt_bias_val, A_log_val, softplus_beta, softplus_threshold
):
    """Compute gate values (g_exp, beta) for a single token."""
    x = alpha + dt_bias_val
    beta_x = softplus_beta * x
    softplus_x = cutlass.Float32(0.0)
    if beta_x <= softplus_threshold:
        softplus_x = (cutlass.Float32(1.0) / softplus_beta) * cute.log(
            cutlass.Float32(1.0) + cute.exp(beta_x, fastmath=True), fastmath=True
        )
    else:
        softplus_x = x
    g = -cute.exp(A_log_val, fastmath=True) * softplus_x
    g_exp = cute.exp(g, fastmath=True)
    beta = cutlass.Float32(1.0) / (
        cutlass.Float32(1.0) + cute.exp(-beta_raw, fastmath=True)
    )
    return g_exp, beta


@cute.jit
def decay_h_from_smem_and_compute_pred(
    h_sh_chunk, h_chunk, kq_chunk, g_exp, lane_idx, k_base
):
    """Load H from SMEM, apply decay, and compute pred = sum_k(h * k)."""
    pred = cutlass.Float32(0.0)
    pred2 = cutlass.Float32(0.0)

    for i in cutlass.range_constexpr(0, 32, 2):
        h_chunk[i], h_chunk[i + 1] = cute.arch.fma_packed_f32x2(
            src_a=(
                h_sh_chunk[lane_idx, k_base + i].to(cutlass.Float32),
                h_sh_chunk[lane_idx, k_base + i + 1].to(cutlass.Float32),
            ),
            src_b=(g_exp, g_exp),
            src_c=(cutlass.Float32(0.0), cutlass.Float32(0.0)),
        )

    for i in cutlass.range_constexpr(0, 32, 2):
        pred, pred2 = cute.arch.fma_packed_f32x2(
            src_a=(h_chunk[i], h_chunk[i + 1]),
            src_b=(kq_chunk[i], kq_chunk[i + 1]),
            src_c=(pred, pred2),
        )

    pred = pred + pred2
    return pred


@cute.jit
def decay_h_in_place(h_chunk, g_exp):
    """Apply decay to H in place: h = h * g_exp."""
    for i in cutlass.range_constexpr(0, 32, 2):
        h_chunk[i], h_chunk[i + 1] = cute.arch.fma_packed_f32x2(
            src_a=(h_chunk[i], h_chunk[i + 1]),
            src_b=(g_exp, g_exp),
            src_c=(cutlass.Float32(0.0), cutlass.Float32(0.0)),
        )


@cute.jit
def process_first_token(
    h_sh_chunk_curr,
    h_chunk,
    kq_chunk,
    k_sh,
    q_sh,
    v_sh,
    reduce_sh,
    o_head,
    g_exp,
    beta,
    v_offset,
    pred_slot,
    warp_idx,
    lane_idx,
    k_base,
    NUM_WARPS: cutlass.Constexpr[int],
):
    """
    Process the first token in a V-chunk (T=0).
    - Load K from SMEM
    - Decay H from SMEM and compute pred
    - Cross-warp reduce pred (uses pred_slot)
    - Update H with delta
    - Load Q and compute output
    Returns: out (partial output, not yet reduced)
    """
    # Load K for this token
    load_kq_chunk_from_smem(k_sh, kq_chunk, k_base)

    # Decay H from SMEM and compute pred = H * K
    pred = decay_h_from_smem_and_compute_pred(
        h_sh_chunk_curr, h_chunk, kq_chunk, g_exp, lane_idx, k_base
    )

    # Reduce pred across warps (slot 0 for first token)
    pred_final = cross_warp_reduce_single(
        reduce_sh, pred_slot, warp_idx, lane_idx, pred, NUM_WARPS
    )

    # Compute delta and update H
    v_delta = (v_sh[v_offset + lane_idx] - pred_final) * beta
    update_h_with_delta(h_chunk, kq_chunk, v_delta)

    # Load Q and compute output
    load_kq_chunk_from_smem(q_sh, kq_chunk, k_base)
    out = compute_output(h_chunk, kq_chunk)

    return out


@cute.jit
def process_middle_token(
    h_chunk,
    kq_chunk,
    k_sh,
    q_sh,
    v_sh,
    reduce_sh,
    o_head_prev,
    g_exp,
    beta,
    v_offset,
    out_slot_prev,
    pred_slot,
    out_prev,
    warp_idx,
    lane_idx,
    k_base,
    NUM_WARPS: cutlass.Constexpr[int],
):
    """
    Process a middle token (T=1, T=2 for T=4 kernel).
    - Decay H in place
    - Load K, compute pred
    - Joint reduction of (prev_out, this_pred)
    - Store prev output
    - Update H with delta
    - Load Q and compute output
    Returns: out (partial output, not yet reduced)
    """
    # Decay H in place
    decay_h_in_place(h_chunk, g_exp)

    # Load K and compute pred
    load_kq_chunk_from_smem(k_sh, kq_chunk, k_base)
    pred = compute_output(h_chunk, kq_chunk)

    # Joint reduction: reduce out_prev and pred together
    out_prev_final, pred_final = cross_warp_reduce_two(
        reduce_sh,
        out_slot_prev,
        pred_slot,
        warp_idx,
        lane_idx,
        out_prev,
        pred,
        NUM_WARPS,
    )

    # Store previous token's output
    if warp_idx == 0:
        o_head_prev[v_offset + lane_idx] = out_prev_final.to(cutlass.BFloat16)

    # Compute delta and update H
    v_delta = (v_sh[v_offset + lane_idx] - pred_final) * beta
    update_h_with_delta(h_chunk, kq_chunk, v_delta)

    # Load Q and compute output
    load_kq_chunk_from_smem(q_sh, kq_chunk, k_base)
    out = compute_output(h_chunk, kq_chunk)

    return out


@cute.jit
def process_last_token_and_finish(
    h_sh_chunk_curr,
    h_chunk,
    kq_chunk,
    k_sh,
    q_sh,
    v_sh,
    reduce_sh,
    o_head_prev,
    o_head_last,
    g_exp,
    beta,
    v_offset,
    out_slot_prev,
    pred_slot,
    out_slot_last,
    out_prev,
    warp_idx,
    lane_idx,
    k_base,
    NUM_WARPS: cutlass.Constexpr[int],
):
    """
    Process the last token and finalize the V-chunk.
    - Decay H in place
    - Load K, compute pred
    - Joint reduction of (prev_out, this_pred)
    - Store prev output
    - Update H with delta
    - Compute last output and reduce
    - Write H back to SMEM
    - Store last output
    """
    # Decay H in place
    decay_h_in_place(h_chunk, g_exp)

    # Load K and compute pred
    load_kq_chunk_from_smem(k_sh, kq_chunk, k_base)
    pred = compute_output(h_chunk, kq_chunk)

    # Joint reduction: reduce out_prev and pred together
    out_prev_final, pred_final = cross_warp_reduce_two(
        reduce_sh,
        out_slot_prev,
        pred_slot,
        warp_idx,
        lane_idx,
        out_prev,
        pred,
        NUM_WARPS,
    )

    # Store previous token's output
    if warp_idx == 0:
        o_head_prev[v_offset + lane_idx] = out_prev_final.to(cutlass.BFloat16)

    # Compute delta and update H
    v_delta = (v_sh[v_offset + lane_idx] - pred_final) * beta
    update_h_with_delta(h_chunk, kq_chunk, v_delta)

    # Compute last output
    load_kq_chunk_from_smem(q_sh, kq_chunk, k_base)
    out_last = compute_output(h_chunk, kq_chunk)

    # Final reduction and store
    out_last_final = cross_warp_reduce_single(
        reduce_sh, out_slot_last, warp_idx, lane_idx, out_last, NUM_WARPS
    )
    write_h_chunk_to_smem(h_chunk, h_sh_chunk_curr, lane_idx, k_base)
    if warp_idx == 0:
        o_head_last[v_offset + lane_idx] = out_last_final.to(cutlass.BFloat16)


# ==============================================================================
# UNIFIED V-CHUNK PROCESSING FOR SEQLEN=2/3/4
# ==============================================================================


@cute.jit
def process_vchunk_unified_234(
    h_sh_chunk_curr,
    h_sh_chunk_prev,
    h_out,
    h_chunk,
    kq_chunk,
    k_sh0,
    k_sh1,
    k_sh2,
    k_sh3,
    q_sh0,
    q_sh1,
    q_sh2,
    q_sh3,
    v_sh0,
    v_sh1,
    v_sh2,
    v_sh3,
    reduce_sh,
    o_head0,
    o_head1,
    o_head2,
    o_head3,
    g_exp0,
    g_exp1,
    g_exp2,
    g_exp3,
    beta0,
    beta1,
    beta2,
    beta3,
    v_offset,
    prev_v_offset,
    store_prev,
    tidx,
    warp_idx,
    lane_idx,
    k_base,
    NUM_TOKENS: cutlass.Constexpr[int],
    HEAD_DIM: cutlass.Constexpr[int],
):
    """
    Unified V-chunk processing for 2, 3, or 4 tokens using Constexpr parameter.

    This function handles V-chunk processing for all multi-token cases (T=2, T=3, T=4)
    using compile-time specialization via NUM_TOKENS.

    Pattern:
    - Token 0: First token (always)
    - Tokens 1 to NUM_TOKENS-2: Middle tokens (compile-time unrolled)
    - Token NUM_TOKENS-1: Last token (always)
    """
    # Store previous H chunk if needed
    if store_prev:
        store_h_smem_to_gmem(h_sh_chunk_prev, h_out, tidx, prev_v_offset, HEAD_DIM)

    # Token 0: First token processing (always executed)
    out0 = process_first_token(
        h_sh_chunk_curr,
        h_chunk,
        kq_chunk,
        k_sh0,
        q_sh0,
        v_sh0,
        reduce_sh,
        o_head0,
        g_exp0,
        beta0,
        v_offset,
        0,  # pred_slot=0
        warp_idx,
        lane_idx,
        k_base,
        HEAD_DIM // 32,  # NUM_WARPS
    )

    # Compile-time dispatch based on NUM_TOKENS
    if NUM_TOKENS == 2:
        # For T=2: Token 1 is the last token
        process_last_token_and_finish(
            h_sh_chunk_curr,
            h_chunk,
            kq_chunk,
            k_sh1,
            q_sh1,
            v_sh1,
            reduce_sh,
            o_head0,
            o_head1,
            g_exp1,
            beta1,
            v_offset,
            1,
            2,
            3,  # out_slot_prev=1, pred_slot=2, out_slot_last=3
            out0,
            warp_idx,
            lane_idx,
            k_base,
            HEAD_DIM // 32,  # NUM_WARPS
        )
    elif NUM_TOKENS == 3:
        # For T=3: Token 1 is middle, Token 2 is last
        out1 = process_middle_token(
            h_chunk,
            kq_chunk,
            k_sh1,
            q_sh1,
            v_sh1,
            reduce_sh,
            o_head0,
            g_exp1,
            beta1,
            v_offset,
            1,
            2,  # out_slot_prev=1, pred_slot=2
            out0,
            warp_idx,
            lane_idx,
            k_base,
            HEAD_DIM // 32,  # NUM_WARPS
        )
        process_last_token_and_finish(
            h_sh_chunk_curr,
            h_chunk,
            kq_chunk,
            k_sh2,
            q_sh2,
            v_sh2,
            reduce_sh,
            o_head1,
            o_head2,
            g_exp2,
            beta2,
            v_offset,
            3,
            4,
            5,  # out_slot_prev=3, pred_slot=4, out_slot_last=5
            out1,
            warp_idx,
            lane_idx,
            k_base,
            HEAD_DIM // 32,  # NUM_WARPS
        )
    else:
        # For T=4: Tokens 1,2 are middle, Token 3 is last
        out1 = process_middle_token(
            h_chunk,
            kq_chunk,
            k_sh1,
            q_sh1,
            v_sh1,
            reduce_sh,
            o_head0,
            g_exp1,
            beta1,
            v_offset,
            1,
            2,  # out_slot_prev=1, pred_slot=2
            out0,
            warp_idx,
            lane_idx,
            k_base,
            HEAD_DIM // 32,  # NUM_WARPS
        )
        out2 = process_middle_token(
            h_chunk,
            kq_chunk,
            k_sh2,
            q_sh2,
            v_sh2,
            reduce_sh,
            o_head1,
            g_exp2,
            beta2,
            v_offset,
            3,
            4,  # out_slot_prev=3, pred_slot=4
            out1,
            warp_idx,
            lane_idx,
            k_base,
            HEAD_DIM // 32,  # NUM_WARPS
        )
        # Last token for NUM_TOKENS=4: Token 3
        process_last_token_and_finish(
            h_sh_chunk_curr,
            h_chunk,
            kq_chunk,
            k_sh3,
            q_sh3,
            v_sh3,
            reduce_sh,
            o_head2,
            o_head3,
            g_exp3,
            beta3,
            v_offset,
            5,
            6,
            7,  # out_slot_prev=5, pred_slot=6, out_slot_last=7
            out2,
            warp_idx,
            lane_idx,
            k_base,
            HEAD_DIM // 32,  # NUM_WARPS
        )


# ==============================================================================
# SEQLEN=1 KERNEL (Persistent K Optimization)
# ==============================================================================


@cute.kernel
def gated_delta_rule_decode_kernel_seqlen1(
    gQ: cute.Tensor,
    gK: cute.Tensor,
    gV: cute.Tensor,
    ga: cute.Tensor,
    gb: cute.Tensor,
    gA_log: cute.Tensor,
    gdt_bias: cute.Tensor,
    gH: cute.Tensor,
    gO: cute.Tensor,
    scale: cutlass.Float32,
    softplus_beta: cutlass.Float32,
    softplus_threshold: cutlass.Float32,
    eps: cutlass.Float32,
    HEAD_DIM: cutlass.Constexpr[int],
):
    """
    Seqlen=1 kernel with persistent K optimization.
    HEAD_DIM: 64 or 128 (compile-time). Derives NUM_WARPS = HEAD_DIM // 32,
    NUM_V_CHUNKS = HEAD_DIM // 32, BLOCK_SIZE = HEAD_DIM.
    """
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    HV = cutlass.Int32(gV.shape[2])
    H = cutlass.Int32(gQ.shape[2])

    batch_idx = bidx // HV
    value_head_idx = bidx % HV
    query_head_idx = value_head_idx // (HV // H)

    smem = utils.SmemAllocator()

    # Compute gates using shared helper
    alpha = ga[(batch_idx, 0, value_head_idx)].to(cutlass.Float32)
    beta_raw = gb[(batch_idx, 0, value_head_idx)].to(cutlass.Float32)
    A_log_val = gA_log[value_head_idx]
    dt_bias_val = gdt_bias[value_head_idx]
    g_exp, beta = compute_single_gate(
        alpha, beta_raw, dt_bias_val, A_log_val, softplus_beta, softplus_threshold
    )

    # Allocate SMEM — always 4 H chunk buffers for simplicity
    h_sh_chunk0 = smem.allocate_tensor(
        cutlass.BFloat16,
        cute.make_layout((32, HEAD_DIM), stride=(HEAD_DIM + H_SMEM_PADDING, 1)),
        byte_alignment=128,
    )
    h_sh_chunk1 = smem.allocate_tensor(
        cutlass.BFloat16,
        cute.make_layout((32, HEAD_DIM), stride=(HEAD_DIM + H_SMEM_PADDING, 1)),
        byte_alignment=128,
    )
    h_sh_chunk2 = smem.allocate_tensor(
        cutlass.BFloat16,
        cute.make_layout((32, HEAD_DIM), stride=(HEAD_DIM + H_SMEM_PADDING, 1)),
        byte_alignment=128,
    )
    h_sh_chunk3 = smem.allocate_tensor(
        cutlass.BFloat16,
        cute.make_layout((32, HEAD_DIM), stride=(HEAD_DIM + H_SMEM_PADDING, 1)),
        byte_alignment=128,
    )

    q_sh = smem.allocate_tensor(cutlass.Float32, HEAD_DIM)
    k_sh = smem.allocate_tensor(cutlass.Float32, HEAD_DIM)

    pred_sh = smem.allocate_tensor(
        cutlass.Float32, cute.make_layout((32, HEAD_DIM // 32), stride=(1, 32))
    )
    out_sh = smem.allocate_tensor(
        cutlass.Float32, cute.make_layout((32, HEAD_DIM // 32), stride=(1, 32))
    )

    h_global = gH[(batch_idx, value_head_idx, None, None)]

    # Launch first 2 async loads
    load_h_chunk_async(h_sh_chunk0, h_global, tidx, 0, HEAD_DIM)
    nvvm.cp_async_commit_group()
    load_h_chunk_async(h_sh_chunk1, h_global, tidx, 32, HEAD_DIM)
    nvvm.cp_async_commit_group()

    # L2 normalization
    q_head = gQ[(batch_idx, 0, query_head_idx, None)]
    k_head = gK[(batch_idx, 0, query_head_idx, None)]

    warp_idx = tidx // 32
    lane_idx = tidx % 32

    # Use shared helper for Q/K normalization (only warp 0 does the work)
    if warp_idx == 0:
        normalize_and_store_qk_to_smem(
            q_head, k_head, q_sh, k_sh, lane_idx, scale, eps, HEAD_DIM
        )

    cute.arch.sync_threads()

    # Load V
    v_head = gV[(batch_idx, 0, value_head_idx, None)]
    v_sh = smem.allocate_tensor(cutlass.Float32, HEAD_DIM)
    v_sh[tidx] = v_head[tidx].to(cutlass.Float32)

    # Registers: h_chunk + k_chunk (persistent) + qk_temp (reused for Q)
    h_chunk = cute.make_rmem_tensor((32,), cutlass.Float32)
    k_chunk = cute.make_rmem_tensor((32,), cutlass.Float32)  # PERSISTENT K!
    qk_temp = cute.make_rmem_tensor((32,), cutlass.Float32)

    k_base = warp_idx * 32

    # Load K ONCE - keep for entire kernel
    for i in cutlass.range_constexpr(32):
        k_chunk[i] = k_sh[k_base + i]

    h_out = gH[(batch_idx, value_head_idx, None, None)]
    o_head = gO[(batch_idx, 0, value_head_idx, None)]

    # ========================================================================
    # CHUNK 0
    # ========================================================================
    nvvm.cp_async_wait_group(1)
    cute.arch.sync_threads()

    pred = cutlass.Float32(0.0)
    pred2 = cutlass.Float32(0.0)
    for i in cutlass.range_constexpr(0, 32, 2):
        h_chunk[i], h_chunk[i + 1] = cute.arch.fma_packed_f32x2(
            src_a=(
                h_sh_chunk0[lane_idx, k_base + i].to(cutlass.Float32),
                h_sh_chunk0[lane_idx, k_base + i + 1].to(cutlass.Float32),
            ),
            src_b=(g_exp, g_exp),
            src_c=(cutlass.Float32(0.0), cutlass.Float32(0.0)),
        )
    for i in cutlass.range_constexpr(0, 32, 2):
        pred, pred2 = cute.arch.fma_packed_f32x2(
            src_a=(h_chunk[i], h_chunk[i + 1]),
            src_b=(k_chunk[i], k_chunk[i + 1]),
            src_c=(pred, pred2),
        )
    pred = pred + pred2

    pred_sh[lane_idx, warp_idx] = pred
    cute.arch.sync_threads()
    pred_final = cutlass.Float32(0.0)
    if HEAD_DIM == 64:
        pred_final = pred_sh[lane_idx, 0] + pred_sh[lane_idx, 1]
    elif HEAD_DIM == 128:
        pred_final = (
            pred_sh[lane_idx, 0]
            + pred_sh[lane_idx, 1]
            + pred_sh[lane_idx, 2]
            + pred_sh[lane_idx, 3]
        )

    v_val = (v_sh[lane_idx] - pred_final) * beta

    for i in cutlass.range_constexpr(0, 32, 2):
        h_chunk[i], h_chunk[i + 1] = cute.arch.fma_packed_f32x2(
            src_a=(k_chunk[i], k_chunk[i + 1]),
            src_b=(v_val, v_val),
            src_c=(h_chunk[i], h_chunk[i + 1]),
        )

    # Load Q for output computation
    for i in cutlass.range_constexpr(32):
        qk_temp[i] = q_sh[k_base + i]

    out = cutlass.Float32(0.0)
    out2 = cutlass.Float32(0.0)
    for i in cutlass.range_constexpr(0, 32, 2):
        out, out2 = cute.arch.fma_packed_f32x2(
            src_a=(h_chunk[i], h_chunk[i + 1]),
            src_b=(qk_temp[i], qk_temp[i + 1]),
            src_c=(out, out2),
        )
    out = out + out2

    out_sh[lane_idx, warp_idx] = out
    cute.arch.sync_threads()
    out_final = cutlass.Float32(0.0)
    if HEAD_DIM == 64:
        out_final = out_sh[lane_idx, 0] + out_sh[lane_idx, 1]
    elif HEAD_DIM == 128:
        out_final = (
            out_sh[lane_idx, 0]
            + out_sh[lane_idx, 1]
            + out_sh[lane_idx, 2]
            + out_sh[lane_idx, 3]
        )

    write_h_chunk_to_smem(h_chunk, h_sh_chunk0, lane_idx, k_base)
    if warp_idx == 0:
        o_head[lane_idx] = out_final.to(cutlass.BFloat16)

    # ========================================================================
    # CHUNK 1
    # ========================================================================
    nvvm.cp_async_wait_group(0)
    cute.arch.sync_threads()

    if HEAD_DIM == 128:
        load_h_chunk_async(h_sh_chunk2, h_global, tidx, 64, HEAD_DIM)
        nvvm.cp_async_commit_group()
        load_h_chunk_async(h_sh_chunk3, h_global, tidx, 96, HEAD_DIM)
        nvvm.cp_async_commit_group()

    store_h_smem_to_gmem(h_sh_chunk0, h_out, tidx, 0, HEAD_DIM)

    pred = cutlass.Float32(0.0)
    pred2 = cutlass.Float32(0.0)
    for i in cutlass.range_constexpr(0, 32, 2):
        h_chunk[i], h_chunk[i + 1] = cute.arch.fma_packed_f32x2(
            src_a=(
                h_sh_chunk1[lane_idx, k_base + i].to(cutlass.Float32),
                h_sh_chunk1[lane_idx, k_base + i + 1].to(cutlass.Float32),
            ),
            src_b=(g_exp, g_exp),
            src_c=(cutlass.Float32(0.0), cutlass.Float32(0.0)),
        )
    for i in cutlass.range_constexpr(0, 32, 2):
        pred, pred2 = cute.arch.fma_packed_f32x2(
            src_a=(h_chunk[i], h_chunk[i + 1]),
            src_b=(k_chunk[i], k_chunk[i + 1]),
            src_c=(pred, pred2),
        )
    pred = pred + pred2

    pred_sh[lane_idx, warp_idx] = pred
    cute.arch.sync_threads()
    pred_final = cutlass.Float32(0.0)
    if HEAD_DIM == 64:
        pred_final = pred_sh[lane_idx, 0] + pred_sh[lane_idx, 1]
    elif HEAD_DIM == 128:
        pred_final = (
            pred_sh[lane_idx, 0]
            + pred_sh[lane_idx, 1]
            + pred_sh[lane_idx, 2]
            + pred_sh[lane_idx, 3]
        )

    v_val = (v_sh[32 + lane_idx] - pred_final) * beta

    for i in cutlass.range_constexpr(0, 32, 2):
        h_chunk[i], h_chunk[i + 1] = cute.arch.fma_packed_f32x2(
            src_a=(k_chunk[i], k_chunk[i + 1]),
            src_b=(v_val, v_val),
            src_c=(h_chunk[i], h_chunk[i + 1]),
        )

    for i in cutlass.range_constexpr(32):
        qk_temp[i] = q_sh[k_base + i]

    out = cutlass.Float32(0.0)
    out2 = cutlass.Float32(0.0)
    for i in cutlass.range_constexpr(0, 32, 2):
        out, out2 = cute.arch.fma_packed_f32x2(
            src_a=(h_chunk[i], h_chunk[i + 1]),
            src_b=(qk_temp[i], qk_temp[i + 1]),
            src_c=(out, out2),
        )
    out = out + out2

    out_sh[lane_idx, warp_idx] = out
    cute.arch.sync_threads()
    out_final = cutlass.Float32(0.0)
    if HEAD_DIM == 64:
        out_final = out_sh[lane_idx, 0] + out_sh[lane_idx, 1]
    elif HEAD_DIM == 128:
        out_final = (
            out_sh[lane_idx, 0]
            + out_sh[lane_idx, 1]
            + out_sh[lane_idx, 2]
            + out_sh[lane_idx, 3]
        )

    write_h_chunk_to_smem(h_chunk, h_sh_chunk1, lane_idx, k_base)
    if warp_idx == 0:
        o_head[32 + lane_idx] = out_final.to(cutlass.BFloat16)

    # For HEAD_DIM=64: done after 2 chunks. Store chunk1 H and return.
    if HEAD_DIM == 64:
        cute.arch.sync_threads()
        store_h_smem_to_gmem(h_sh_chunk1, h_out, tidx, 32, HEAD_DIM)

    # ========================================================================
    # CHUNK 2 (HEAD_DIM=128 only)
    # ========================================================================
    if HEAD_DIM == 128:
        nvvm.cp_async_wait_group(1)
        cute.arch.sync_threads()

        store_h_smem_to_gmem(h_sh_chunk1, h_out, tidx, 32, HEAD_DIM)

        pred = cutlass.Float32(0.0)
        pred2 = cutlass.Float32(0.0)
        for i in cutlass.range_constexpr(0, 32, 2):
            h_chunk[i], h_chunk[i + 1] = cute.arch.fma_packed_f32x2(
                src_a=(
                    h_sh_chunk2[lane_idx, k_base + i].to(cutlass.Float32),
                    h_sh_chunk2[lane_idx, k_base + i + 1].to(cutlass.Float32),
                ),
                src_b=(g_exp, g_exp),
                src_c=(cutlass.Float32(0.0), cutlass.Float32(0.0)),
            )
        for i in cutlass.range_constexpr(0, 32, 2):
            pred, pred2 = cute.arch.fma_packed_f32x2(
                src_a=(h_chunk[i], h_chunk[i + 1]),
                src_b=(k_chunk[i], k_chunk[i + 1]),
                src_c=(pred, pred2),
            )
        pred = pred + pred2

        pred_sh[lane_idx, warp_idx] = pred
        cute.arch.sync_threads()
        pred_final = (
            pred_sh[lane_idx, 0]
            + pred_sh[lane_idx, 1]
            + pred_sh[lane_idx, 2]
            + pred_sh[lane_idx, 3]
        )

        v_val = (v_sh[64 + lane_idx] - pred_final) * beta

        for i in cutlass.range_constexpr(0, 32, 2):
            h_chunk[i], h_chunk[i + 1] = cute.arch.fma_packed_f32x2(
                src_a=(k_chunk[i], k_chunk[i + 1]),
                src_b=(v_val, v_val),
                src_c=(h_chunk[i], h_chunk[i + 1]),
            )

        for i in cutlass.range_constexpr(32):
            qk_temp[i] = q_sh[k_base + i]

        out = cutlass.Float32(0.0)
        out2 = cutlass.Float32(0.0)
        for i in cutlass.range_constexpr(0, 32, 2):
            out, out2 = cute.arch.fma_packed_f32x2(
                src_a=(h_chunk[i], h_chunk[i + 1]),
                src_b=(qk_temp[i], qk_temp[i + 1]),
                src_c=(out, out2),
            )
        out = out + out2

        out_sh[lane_idx, warp_idx] = out
        cute.arch.sync_threads()
        out_final = (
            out_sh[lane_idx, 0]
            + out_sh[lane_idx, 1]
            + out_sh[lane_idx, 2]
            + out_sh[lane_idx, 3]
        )

        write_h_chunk_to_smem(h_chunk, h_sh_chunk2, lane_idx, k_base)
        if warp_idx == 0:
            o_head[64 + lane_idx] = out_final.to(cutlass.BFloat16)

        # ====================================================================
        # CHUNK 3 (HEAD_DIM=128 only)
        # ====================================================================
        nvvm.cp_async_wait_group(0)
        cute.arch.sync_threads()

        store_h_smem_to_gmem(h_sh_chunk2, h_out, tidx, 64, HEAD_DIM)

        pred = cutlass.Float32(0.0)
        pred2 = cutlass.Float32(0.0)
        for i in cutlass.range_constexpr(0, 32, 2):
            h_chunk[i], h_chunk[i + 1] = cute.arch.fma_packed_f32x2(
                src_a=(
                    h_sh_chunk3[lane_idx, k_base + i].to(cutlass.Float32),
                    h_sh_chunk3[lane_idx, k_base + i + 1].to(cutlass.Float32),
                ),
                src_b=(g_exp, g_exp),
                src_c=(cutlass.Float32(0.0), cutlass.Float32(0.0)),
            )
        for i in cutlass.range_constexpr(0, 32, 2):
            pred, pred2 = cute.arch.fma_packed_f32x2(
                src_a=(h_chunk[i], h_chunk[i + 1]),
                src_b=(k_chunk[i], k_chunk[i + 1]),
                src_c=(pred, pred2),
            )
        pred = pred + pred2

        pred_sh[lane_idx, warp_idx] = pred
        cute.arch.sync_threads()
        pred_final = (
            pred_sh[lane_idx, 0]
            + pred_sh[lane_idx, 1]
            + pred_sh[lane_idx, 2]
            + pred_sh[lane_idx, 3]
        )

        v_val = (v_sh[96 + lane_idx] - pred_final) * beta

        for i in cutlass.range_constexpr(0, 32, 2):
            h_chunk[i], h_chunk[i + 1] = cute.arch.fma_packed_f32x2(
                src_a=(k_chunk[i], k_chunk[i + 1]),
                src_b=(v_val, v_val),
                src_c=(h_chunk[i], h_chunk[i + 1]),
            )

        for i in cutlass.range_constexpr(32):
            qk_temp[i] = q_sh[k_base + i]

        out = cutlass.Float32(0.0)
        out2 = cutlass.Float32(0.0)
        for i in cutlass.range_constexpr(0, 32, 2):
            out, out2 = cute.arch.fma_packed_f32x2(
                src_a=(h_chunk[i], h_chunk[i + 1]),
                src_b=(qk_temp[i], qk_temp[i + 1]),
                src_c=(out, out2),
            )
        out = out + out2

        out_sh[lane_idx, warp_idx] = out
        cute.arch.sync_threads()
        out_final = (
            out_sh[lane_idx, 0]
            + out_sh[lane_idx, 1]
            + out_sh[lane_idx, 2]
            + out_sh[lane_idx, 3]
        )

        write_h_chunk_to_smem(h_chunk, h_sh_chunk3, lane_idx, k_base)
        if warp_idx == 0:
            o_head[96 + lane_idx] = out_final.to(cutlass.BFloat16)

        cute.arch.sync_threads()
        store_h_smem_to_gmem(h_sh_chunk3, h_out, tidx, 96, HEAD_DIM)


# ==============================================================================
# UNIFIED SEQLEN=2/3/4 MAIN KERNEL
# ==============================================================================


@cute.kernel
def gated_delta_rule_decode_kernel_seqlen234_unified(
    gQ: cute.Tensor,  # [B, T=2/3/4, H, K]
    gK: cute.Tensor,  # [B, T=2/3/4, H, K]
    gV: cute.Tensor,  # [B, T=2/3/4, HV, V]
    ga: cute.Tensor,  # [B, T=2/3/4, HV]
    gb: cute.Tensor,  # [B, T=2/3/4, HV]
    gA_log: cute.Tensor,  # [HV]
    gdt_bias: cute.Tensor,  # [HV]
    gH: cute.Tensor,  # [B, HV, V, K] - K-fast layout
    gO: cute.Tensor,  # [B, T=2/3/4, HV, V]
    scale: cutlass.Float32,
    softplus_beta: cutlass.Float32,
    softplus_threshold: cutlass.Float32,
    eps: cutlass.Float32,
    NUM_TOKENS: cutlass.Constexpr[int],  # 2, 3, or 4
    HEAD_DIM: cutlass.Constexpr[int],  # 64 or 128
):
    """
    Unified kernel for Seqlen=2, Seqlen=3 and Seqlen=4 with compile-time specialization.
    HEAD_DIM: 64 or 128 (compile-time). Derives NUM_WARPS = HEAD_DIM // 32,
    NUM_V_CHUNKS = HEAD_DIM // 32.
    """
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    HV = cutlass.Int32(gV.shape[2])
    H = cutlass.Int32(gQ.shape[2])

    batch_idx = bidx // HV
    value_head_idx = bidx % HV
    query_head_idx = value_head_idx // (HV // H)

    warp_idx = tidx // 32
    lane_idx = tidx % 32
    k_base = warp_idx * 32

    smem = utils.SmemAllocator()

    # SMEM Allocation - H chunks (always 4 for simplicity)
    h_sh_chunk0 = smem.allocate_tensor(
        cutlass.BFloat16,
        cute.make_layout((32, HEAD_DIM), stride=(HEAD_DIM + H_SMEM_PADDING, 1)),
        byte_alignment=128,
    )
    h_sh_chunk1 = smem.allocate_tensor(
        cutlass.BFloat16,
        cute.make_layout((32, HEAD_DIM), stride=(HEAD_DIM + H_SMEM_PADDING, 1)),
        byte_alignment=128,
    )
    h_sh_chunk2 = smem.allocate_tensor(
        cutlass.BFloat16,
        cute.make_layout((32, HEAD_DIM), stride=(HEAD_DIM + H_SMEM_PADDING, 1)),
        byte_alignment=128,
    )
    h_sh_chunk3 = smem.allocate_tensor(
        cutlass.BFloat16,
        cute.make_layout((32, HEAD_DIM), stride=(HEAD_DIM + H_SMEM_PADDING, 1)),
        byte_alignment=128,
    )

    # Q/K buffers for tokens 0 and 1 (always needed for T>=2)
    q_sh0 = smem.allocate_tensor(cutlass.Float32, HEAD_DIM)
    k_sh0 = smem.allocate_tensor(cutlass.Float32, HEAD_DIM)
    q_sh1 = smem.allocate_tensor(cutlass.Float32, HEAD_DIM)
    k_sh1 = smem.allocate_tensor(cutlass.Float32, HEAD_DIM)

    # Q/K buffers for token 2 (only for NUM_TOKENS >= 3)
    q_sh2 = smem.allocate_tensor(cutlass.Float32, HEAD_DIM)
    k_sh2 = smem.allocate_tensor(cutlass.Float32, HEAD_DIM)

    # Q/K buffers for token 3 (only for NUM_TOKENS=4)
    q_sh3 = smem.allocate_tensor(cutlass.Float32, HEAD_DIM)
    k_sh3 = smem.allocate_tensor(cutlass.Float32, HEAD_DIM)

    # V buffers
    v_sh0 = smem.allocate_tensor(cutlass.Float32, HEAD_DIM)
    v_sh1 = smem.allocate_tensor(cutlass.Float32, HEAD_DIM)
    v_sh2 = smem.allocate_tensor(cutlass.Float32, HEAD_DIM)
    v_sh3 = smem.allocate_tensor(cutlass.Float32, HEAD_DIM)

    # Bank-conflict-free reduce_sh: [slot, lane_idx, warp_idx]
    # NUM_WARPS = HEAD_DIM // 32
    reduce_sh = smem.allocate_tensor(
        cutlass.Float32,
        cute.make_layout(
            (8, 32, HEAD_DIM // 32), stride=(32 * (HEAD_DIM // 32), HEAD_DIM // 32, 1)
        ),
    )

    # Register allocation
    h_chunk = cute.make_rmem_tensor((32,), cutlass.Float32)
    kq_chunk = cute.make_rmem_tensor((32,), cutlass.Float32)

    # Gate computation - always compute gates 0, 1 (for T>=2)
    A_log_val = gA_log[value_head_idx]
    dt_bias_val = gdt_bias[value_head_idx]

    alpha0 = ga[(batch_idx, 0, value_head_idx)].to(cutlass.Float32)
    beta_raw0 = gb[(batch_idx, 0, value_head_idx)].to(cutlass.Float32)
    g_exp0, beta0 = compute_single_gate(
        alpha0, beta_raw0, dt_bias_val, A_log_val, softplus_beta, softplus_threshold
    )

    alpha1 = ga[(batch_idx, 1, value_head_idx)].to(cutlass.Float32)
    beta_raw1 = gb[(batch_idx, 1, value_head_idx)].to(cutlass.Float32)
    g_exp1, beta1 = compute_single_gate(
        alpha1, beta_raw1, dt_bias_val, A_log_val, softplus_beta, softplus_threshold
    )

    # Gate 2 - only for NUM_TOKENS >= 3
    g_exp2 = cutlass.Float32(0.0)
    beta2 = cutlass.Float32(0.0)
    if NUM_TOKENS >= 3:
        alpha2 = ga[(batch_idx, 2, value_head_idx)].to(cutlass.Float32)
        beta_raw2 = gb[(batch_idx, 2, value_head_idx)].to(cutlass.Float32)
        g_exp2, beta2 = compute_single_gate(
            alpha2, beta_raw2, dt_bias_val, A_log_val, softplus_beta, softplus_threshold
        )

    # Gate 3 - only for NUM_TOKENS = 4
    g_exp3 = cutlass.Float32(0.0)
    beta3 = cutlass.Float32(0.0)
    if NUM_TOKENS == 4:
        alpha3 = ga[(batch_idx, 3, value_head_idx)].to(cutlass.Float32)
        beta_raw3 = gb[(batch_idx, 3, value_head_idx)].to(cutlass.Float32)
        g_exp3, beta3 = compute_single_gate(
            alpha3, beta_raw3, dt_bias_val, A_log_val, softplus_beta, softplus_threshold
        )

    # Upfront H loading — load all NUM_V_CHUNKS
    h_global = gH[(batch_idx, value_head_idx, None, None)]
    load_h_chunk_async(h_sh_chunk0, h_global, tidx, 0, HEAD_DIM)
    nvvm.cp_async_commit_group()
    load_h_chunk_async(h_sh_chunk1, h_global, tidx, 32, HEAD_DIM)
    nvvm.cp_async_commit_group()
    if HEAD_DIM == 128:
        load_h_chunk_async(h_sh_chunk2, h_global, tidx, 64, HEAD_DIM)
        nvvm.cp_async_commit_group()
        load_h_chunk_async(h_sh_chunk3, h_global, tidx, 96, HEAD_DIM)
        nvvm.cp_async_commit_group()

    # Q/K normalization
    q_head0 = gQ[(batch_idx, 0, query_head_idx, None)]
    k_head0 = gK[(batch_idx, 0, query_head_idx, None)]
    q_head1 = gQ[(batch_idx, 1, query_head_idx, None)]
    k_head1 = gK[(batch_idx, 1, query_head_idx, None)]

    # Round 1: warp 0 -> token 0, warp 1 -> token 1
    if warp_idx == 0:
        normalize_and_store_qk_to_smem(
            q_head0, k_head0, q_sh0, k_sh0, lane_idx, scale, eps, HEAD_DIM
        )
    if warp_idx == 1:
        normalize_and_store_qk_to_smem(
            q_head1, k_head1, q_sh1, k_sh1, lane_idx, scale, eps, HEAD_DIM
        )

    if HEAD_DIM == 128:
        # 4 warps: warp 2 -> token 2, warp 3 -> token 3 (if needed)
        if NUM_TOKENS >= 3:
            q_head2 = gQ[(batch_idx, 2, query_head_idx, None)]
            k_head2 = gK[(batch_idx, 2, query_head_idx, None)]
            if warp_idx == 2:
                normalize_and_store_qk_to_smem(
                    q_head2, k_head2, q_sh2, k_sh2, lane_idx, scale, eps, HEAD_DIM
                )

        if NUM_TOKENS == 4:
            q_head3 = gQ[(batch_idx, 3, query_head_idx, None)]
            k_head3 = gK[(batch_idx, 3, query_head_idx, None)]
            if warp_idx == 3:
                normalize_and_store_qk_to_smem(
                    q_head3, k_head3, q_sh3, k_sh3, lane_idx, scale, eps, HEAD_DIM
                )

    if HEAD_DIM == 64:
        # 2 warps: need extra round(s) for tokens 2 and 3
        cute.arch.sync_threads()
        if NUM_TOKENS >= 3:
            q_head2 = gQ[(batch_idx, 2, query_head_idx, None)]
            k_head2 = gK[(batch_idx, 2, query_head_idx, None)]
            if warp_idx == 0:
                normalize_and_store_qk_to_smem(
                    q_head2, k_head2, q_sh2, k_sh2, lane_idx, scale, eps, HEAD_DIM
                )

        if NUM_TOKENS == 4:
            q_head3 = gQ[(batch_idx, 3, query_head_idx, None)]
            k_head3 = gK[(batch_idx, 3, query_head_idx, None)]
            if warp_idx == 1:
                normalize_and_store_qk_to_smem(
                    q_head3, k_head3, q_sh3, k_sh3, lane_idx, scale, eps, HEAD_DIM
                )

    cute.arch.sync_threads()

    # V loading - tokens 0, 1 always
    v_head0 = gV[(batch_idx, 0, value_head_idx, None)]
    v_head1 = gV[(batch_idx, 1, value_head_idx, None)]
    load_v_to_smem(v_head0, v_sh0, tidx)
    load_v_to_smem(v_head1, v_sh1, tidx)

    # Token 2 V loading - only for NUM_TOKENS >= 3
    if NUM_TOKENS >= 3:
        v_head2 = gV[(batch_idx, 2, value_head_idx, None)]
        load_v_to_smem(v_head2, v_sh2, tidx)

    # Token 3 V loading - only for NUM_TOKENS = 4
    if NUM_TOKENS == 4:
        v_head3 = gV[(batch_idx, 3, value_head_idx, None)]
        load_v_to_smem(v_head3, v_sh3, tidx)

    # Output pointers - tokens 0, 1 always
    h_out = gH[(batch_idx, value_head_idx, None, None)]
    o_head0 = gO[(batch_idx, 0, value_head_idx, None)]
    o_head1 = gO[(batch_idx, 1, value_head_idx, None)]

    # Token 2 output pointer
    o_head2 = o_head1  # Default for T=2
    if NUM_TOKENS >= 3:
        o_head2 = gO[(batch_idx, 2, value_head_idx, None)]

    # Token 3 output pointer
    o_head3 = o_head2  # Default for T=2,3
    if NUM_TOKENS == 4:
        o_head3 = gO[(batch_idx, 3, value_head_idx, None)]

    # Process V-CHUNK 0
    if HEAD_DIM == 64:
        nvvm.cp_async_wait_group(1)
    elif HEAD_DIM == 128:
        nvvm.cp_async_wait_group(3)
    cute.arch.sync_threads()
    process_vchunk_unified_234(
        h_sh_chunk0,
        h_sh_chunk0,
        h_out,
        h_chunk,
        kq_chunk,
        k_sh0,
        k_sh1,
        k_sh2,
        k_sh3,
        q_sh0,
        q_sh1,
        q_sh2,
        q_sh3,
        v_sh0,
        v_sh1,
        v_sh2,
        v_sh3,
        reduce_sh,
        o_head0,
        o_head1,
        o_head2,
        o_head3,
        g_exp0,
        g_exp1,
        g_exp2,
        g_exp3,
        beta0,
        beta1,
        beta2,
        beta3,
        0,
        0,
        cutlass.Int32(0),
        tidx,
        warp_idx,
        lane_idx,
        k_base,
        NUM_TOKENS,
        HEAD_DIM,
    )

    # Process V-CHUNK 1
    if HEAD_DIM == 64:
        nvvm.cp_async_wait_group(0)
    elif HEAD_DIM == 128:
        nvvm.cp_async_wait_group(2)
    cute.arch.sync_threads()
    process_vchunk_unified_234(
        h_sh_chunk1,
        h_sh_chunk0,
        h_out,
        h_chunk,
        kq_chunk,
        k_sh0,
        k_sh1,
        k_sh2,
        k_sh3,
        q_sh0,
        q_sh1,
        q_sh2,
        q_sh3,
        v_sh0,
        v_sh1,
        v_sh2,
        v_sh3,
        reduce_sh,
        o_head0,
        o_head1,
        o_head2,
        o_head3,
        g_exp0,
        g_exp1,
        g_exp2,
        g_exp3,
        beta0,
        beta1,
        beta2,
        beta3,
        32,
        0,
        cutlass.Int32(1),
        tidx,
        warp_idx,
        lane_idx,
        k_base,
        NUM_TOKENS,
        HEAD_DIM,
    )

    if HEAD_DIM == 64:
        # For HEAD_DIM=64: done after 2 chunks. Store chunk1 H and return.
        cute.arch.sync_threads()
        store_h_smem_to_gmem(h_sh_chunk1, h_out, tidx, 32, HEAD_DIM)

    if HEAD_DIM == 128:
        # Process V-CHUNK 2
        nvvm.cp_async_wait_group(1)
        cute.arch.sync_threads()
        process_vchunk_unified_234(
            h_sh_chunk2,
            h_sh_chunk1,
            h_out,
            h_chunk,
            kq_chunk,
            k_sh0,
            k_sh1,
            k_sh2,
            k_sh3,
            q_sh0,
            q_sh1,
            q_sh2,
            q_sh3,
            v_sh0,
            v_sh1,
            v_sh2,
            v_sh3,
            reduce_sh,
            o_head0,
            o_head1,
            o_head2,
            o_head3,
            g_exp0,
            g_exp1,
            g_exp2,
            g_exp3,
            beta0,
            beta1,
            beta2,
            beta3,
            64,
            32,
            cutlass.Int32(1),
            tidx,
            warp_idx,
            lane_idx,
            k_base,
            NUM_TOKENS,
            HEAD_DIM,
        )

        # Process V-CHUNK 3
        nvvm.cp_async_wait_group(0)
        cute.arch.sync_threads()
        process_vchunk_unified_234(
            h_sh_chunk3,
            h_sh_chunk2,
            h_out,
            h_chunk,
            kq_chunk,
            k_sh0,
            k_sh1,
            k_sh2,
            k_sh3,
            q_sh0,
            q_sh1,
            q_sh2,
            q_sh3,
            v_sh0,
            v_sh1,
            v_sh2,
            v_sh3,
            reduce_sh,
            o_head0,
            o_head1,
            o_head2,
            o_head3,
            g_exp0,
            g_exp1,
            g_exp2,
            g_exp3,
            beta0,
            beta1,
            beta2,
            beta3,
            96,
            64,
            cutlass.Int32(1),
            tidx,
            warp_idx,
            lane_idx,
            k_base,
            NUM_TOKENS,
            HEAD_DIM,
        )

        # Final H store
        cute.arch.sync_threads()
        store_h_smem_to_gmem(h_sh_chunk3, h_out, tidx, 96, HEAD_DIM)


# ==============================================================================
# LAUNCH WRAPPERS
# ==============================================================================


@cute.jit
def gated_delta_rule_launch_seqlen1(
    mQ: cute.Tensor,
    mK: cute.Tensor,
    mV: cute.Tensor,
    ma: cute.Tensor,
    mb: cute.Tensor,
    mA_log: cute.Tensor,
    mdt_bias: cute.Tensor,
    mH: cute.Tensor,
    mO: cute.Tensor,
    scale: cutlass.Float32,
    softplus_beta: cutlass.Float32,
    softplus_threshold: cutlass.Float32,
    eps: cutlass.Float32,
    stream: cuda.CUstream,
    HEAD_DIM: cutlass.Constexpr[int],
):
    batch_size = mQ.shape[0]
    HV = mV.shape[2]

    gated_delta_rule_decode_kernel_seqlen1(
        mQ,
        mK,
        mV,
        ma,
        mb,
        mA_log,
        mdt_bias,
        mH,
        mO,
        scale,
        softplus_beta,
        softplus_threshold,
        eps,
        HEAD_DIM,
    ).launch(
        grid=[batch_size * HV, 1, 1],
        block=[HEAD_DIM, 1, 1],
        stream=stream,
    )


# ==============================================================================
# LOW-BS SEQLEN=1 KERNEL - 1 V-CHUNK PER CTA (T=1, BS<=4)
# ==============================================================================


@cute.kernel
def gated_delta_rule_decode_kernel_seqlen1_lowBS_1chunk(
    gQ: cute.Tensor,
    gK: cute.Tensor,
    gV: cute.Tensor,
    ga: cute.Tensor,
    gb: cute.Tensor,
    gA_log: cute.Tensor,
    gdt_bias: cute.Tensor,
    gH: cute.Tensor,
    gO: cute.Tensor,
    scale: cutlass.Float32,
    softplus_beta: cutlass.Float32,
    softplus_threshold: cutlass.Float32,
    eps: cutlass.Float32,
    HEAD_DIM: cutlass.Constexpr[int],
):
    """
    Seqlen=1 kernel with 1 V-chunk (32 V rows) per CTA.
    For T=1, batch_size <= 4: more CTAs per batch*head for better SM utilization.
    Grid: batch_idx * HV * NUM_V_CHUNKS + value_head_idx * NUM_V_CHUNKS + v_chunk_idx.
    HEAD_DIM: 64 or 128 (compile-time). NUM_V_CHUNKS = HEAD_DIM // 32.
    """
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    HV = cutlass.Int32(gV.shape[2])
    H = cutlass.Int32(gQ.shape[2])

    batch_idx = bidx // (HV * (HEAD_DIM // 32))
    remainder = bidx % (HV * (HEAD_DIM // 32))
    value_head_idx = remainder // (HEAD_DIM // 32)
    v_chunk_idx = remainder % (HEAD_DIM // 32)

    query_head_idx = value_head_idx // (HV // H)
    v_row_base = v_chunk_idx * 32

    smem = utils.SmemAllocator()

    alpha = ga[(batch_idx, 0, value_head_idx)].to(cutlass.Float32)
    beta_raw = gb[(batch_idx, 0, value_head_idx)].to(cutlass.Float32)
    A_log_val = gA_log[value_head_idx]
    dt_bias_val = gdt_bias[value_head_idx]
    g_exp, beta = compute_single_gate(
        alpha, beta_raw, dt_bias_val, A_log_val, softplus_beta, softplus_threshold
    )

    h_sh_chunk = smem.allocate_tensor(
        cutlass.BFloat16,
        cute.make_layout((32, HEAD_DIM), stride=(HEAD_DIM + H_SMEM_PADDING, 1)),
        byte_alignment=128,
    )

    q_sh = smem.allocate_tensor(cutlass.Float32, HEAD_DIM)
    k_sh = smem.allocate_tensor(cutlass.Float32, HEAD_DIM)

    pred_sh = smem.allocate_tensor(
        cutlass.Float32, cute.make_layout((32, HEAD_DIM // 32), stride=(1, 32))
    )
    out_sh = smem.allocate_tensor(
        cutlass.Float32, cute.make_layout((32, HEAD_DIM // 32), stride=(1, 32))
    )

    h_global = gH[(batch_idx, value_head_idx, None, None)]

    load_h_chunk_async(h_sh_chunk, h_global, tidx, v_row_base, HEAD_DIM)
    nvvm.cp_async_commit_group()

    q_head = gQ[(batch_idx, 0, query_head_idx, None)]
    k_head = gK[(batch_idx, 0, query_head_idx, None)]

    warp_idx = tidx // 32
    lane_idx = tidx % 32

    if warp_idx == 0:
        normalize_and_store_qk_to_smem(
            q_head, k_head, q_sh, k_sh, lane_idx, scale, eps, HEAD_DIM
        )

    cute.arch.sync_threads()

    v_head = gV[(batch_idx, 0, value_head_idx, None)]
    v_sh = smem.allocate_tensor(cutlass.Float32, 32)
    if tidx < 32:
        v_sh[tidx] = v_head[v_row_base + tidx].to(cutlass.Float32)

    h_chunk = cute.make_rmem_tensor((32,), cutlass.Float32)
    k_chunk = cute.make_rmem_tensor((32,), cutlass.Float32)  # PERSISTENT K!
    qk_temp = cute.make_rmem_tensor((32,), cutlass.Float32)

    k_base = warp_idx * 32

    for i in cutlass.range_constexpr(32):
        k_chunk[i] = k_sh[k_base + i]

    h_out = gH[(batch_idx, value_head_idx, None, None)]
    o_head = gO[(batch_idx, 0, value_head_idx, None)]

    nvvm.cp_async_wait_group(0)
    cute.arch.sync_threads()

    pred = cutlass.Float32(0.0)
    pred2 = cutlass.Float32(0.0)
    for i in cutlass.range_constexpr(0, 32, 2):
        h_chunk[i], h_chunk[i + 1] = cute.arch.fma_packed_f32x2(
            src_a=(
                h_sh_chunk[lane_idx, k_base + i].to(cutlass.Float32),
                h_sh_chunk[lane_idx, k_base + i + 1].to(cutlass.Float32),
            ),
            src_b=(g_exp, g_exp),
            src_c=(cutlass.Float32(0.0), cutlass.Float32(0.0)),
        )
    for i in cutlass.range_constexpr(0, 32, 2):
        pred, pred2 = cute.arch.fma_packed_f32x2(
            src_a=(h_chunk[i], h_chunk[i + 1]),
            src_b=(k_chunk[i], k_chunk[i + 1]),
            src_c=(pred, pred2),
        )
    pred = pred + pred2

    pred_sh[lane_idx, warp_idx] = pred
    cute.arch.sync_threads()
    pred_final = cutlass.Float32(0.0)
    if HEAD_DIM == 64:
        pred_final = pred_sh[lane_idx, 0] + pred_sh[lane_idx, 1]
    elif HEAD_DIM == 128:
        pred_final = (
            pred_sh[lane_idx, 0]
            + pred_sh[lane_idx, 1]
            + pred_sh[lane_idx, 2]
            + pred_sh[lane_idx, 3]
        )

    v_val = (v_sh[lane_idx] - pred_final) * beta

    for i in cutlass.range_constexpr(0, 32, 2):
        h_chunk[i], h_chunk[i + 1] = cute.arch.fma_packed_f32x2(
            src_a=(k_chunk[i], k_chunk[i + 1]),
            src_b=(v_val, v_val),
            src_c=(h_chunk[i], h_chunk[i + 1]),
        )

    for i in cutlass.range_constexpr(32):
        qk_temp[i] = q_sh[k_base + i]

    out = cutlass.Float32(0.0)
    out2 = cutlass.Float32(0.0)
    for i in cutlass.range_constexpr(0, 32, 2):
        out, out2 = cute.arch.fma_packed_f32x2(
            src_a=(h_chunk[i], h_chunk[i + 1]),
            src_b=(qk_temp[i], qk_temp[i + 1]),
            src_c=(out, out2),
        )
    out = out + out2

    out_sh[lane_idx, warp_idx] = out
    cute.arch.sync_threads()
    out_final = cutlass.Float32(0.0)
    if HEAD_DIM == 64:
        out_final = out_sh[lane_idx, 0] + out_sh[lane_idx, 1]
    elif HEAD_DIM == 128:
        out_final = (
            out_sh[lane_idx, 0]
            + out_sh[lane_idx, 1]
            + out_sh[lane_idx, 2]
            + out_sh[lane_idx, 3]
        )

    write_h_chunk_to_smem(h_chunk, h_sh_chunk, lane_idx, k_base)
    if warp_idx == 0:
        o_head[v_row_base + lane_idx] = out_final.to(cutlass.BFloat16)

    cute.arch.sync_threads()
    store_h_smem_to_gmem(h_sh_chunk, h_out, tidx, v_row_base, HEAD_DIM)


@cute.jit
def gated_delta_rule_launch_seqlen1_lowBS_1chunk(
    mQ: cute.Tensor,
    mK: cute.Tensor,
    mV: cute.Tensor,
    ma: cute.Tensor,
    mb: cute.Tensor,
    mA_log: cute.Tensor,
    mdt_bias: cute.Tensor,
    mH: cute.Tensor,
    mO: cute.Tensor,
    scale: cutlass.Float32,
    softplus_beta: cutlass.Float32,
    softplus_threshold: cutlass.Float32,
    eps: cutlass.Float32,
    stream: cuda.CUstream,
    HEAD_DIM: cutlass.Constexpr[int],
):
    """Launch LowBS-1 kernel: NUM_V_CHUNKS CTAs per (batch, value_head)."""
    batch_size = mQ.shape[0]
    HV = mV.shape[2]

    gated_delta_rule_decode_kernel_seqlen1_lowBS_1chunk(
        mQ,
        mK,
        mV,
        ma,
        mb,
        mA_log,
        mdt_bias,
        mH,
        mO,
        scale,
        softplus_beta,
        softplus_threshold,
        eps,
        HEAD_DIM,
    ).launch(
        grid=[batch_size * HV * (HEAD_DIM // 32), 1, 1],
        block=[HEAD_DIM, 1, 1],
        stream=stream,
    )


@cute.jit
def gated_delta_rule_launch_seqlen2(
    mQ: cute.Tensor,
    mK: cute.Tensor,
    mV: cute.Tensor,
    ma: cute.Tensor,
    mb: cute.Tensor,
    mA_log: cute.Tensor,
    mdt_bias: cute.Tensor,
    mH: cute.Tensor,
    mO: cute.Tensor,
    scale: cutlass.Float32,
    softplus_beta: cutlass.Float32,
    softplus_threshold: cutlass.Float32,
    eps: cutlass.Float32,
    stream: cuda.CUstream,
    HEAD_DIM: cutlass.Constexpr[int],
):
    batch_size = mQ.shape[0]
    HV = mV.shape[2]

    gated_delta_rule_decode_kernel_seqlen234_unified(
        mQ,
        mK,
        mV,
        ma,
        mb,
        mA_log,
        mdt_bias,
        mH,
        mO,
        scale,
        softplus_beta,
        softplus_threshold,
        eps,
        2,  # NUM_TOKENS=2
        HEAD_DIM,
    ).launch(
        grid=[batch_size * HV, 1, 1],
        block=[HEAD_DIM, 1, 1],
        stream=stream,
    )


@cute.jit
def gated_delta_rule_launch_seqlen3(
    mQ: cute.Tensor,
    mK: cute.Tensor,
    mV: cute.Tensor,
    ma: cute.Tensor,
    mb: cute.Tensor,
    mA_log: cute.Tensor,
    mdt_bias: cute.Tensor,
    mH: cute.Tensor,
    mO: cute.Tensor,
    scale: cutlass.Float32,
    softplus_beta: cutlass.Float32,
    softplus_threshold: cutlass.Float32,
    eps: cutlass.Float32,
    stream: cuda.CUstream,
    HEAD_DIM: cutlass.Constexpr[int],
):
    batch_size = mQ.shape[0]
    HV = mV.shape[2]

    gated_delta_rule_decode_kernel_seqlen234_unified(
        mQ,
        mK,
        mV,
        ma,
        mb,
        mA_log,
        mdt_bias,
        mH,
        mO,
        scale,
        softplus_beta,
        softplus_threshold,
        eps,
        3,  # NUM_TOKENS=3
        HEAD_DIM,
    ).launch(
        grid=[batch_size * HV, 1, 1],
        block=[HEAD_DIM, 1, 1],
        stream=stream,
    )


@cute.jit
def gated_delta_rule_launch_seqlen4(
    mQ: cute.Tensor,
    mK: cute.Tensor,
    mV: cute.Tensor,
    ma: cute.Tensor,
    mb: cute.Tensor,
    mA_log: cute.Tensor,
    mdt_bias: cute.Tensor,
    mH: cute.Tensor,
    mO: cute.Tensor,
    scale: cutlass.Float32,
    softplus_beta: cutlass.Float32,
    softplus_threshold: cutlass.Float32,
    eps: cutlass.Float32,
    stream: cuda.CUstream,
    HEAD_DIM: cutlass.Constexpr[int],
):
    batch_size = mQ.shape[0]
    HV = mV.shape[2]

    gated_delta_rule_decode_kernel_seqlen234_unified(
        mQ,
        mK,
        mV,
        ma,
        mb,
        mA_log,
        mdt_bias,
        mH,
        mO,
        scale,
        softplus_beta,
        softplus_threshold,
        eps,
        4,  # NUM_TOKENS=4
        HEAD_DIM,
    ).launch(
        grid=[batch_size * HV, 1, 1],
        block=[HEAD_DIM, 1, 1],
        stream=stream,
    )


# ==============================================================================
# KERNEL CLASS
# ==============================================================================


class GatedDeltaRuleKernel:
    """
    Gated Delta Rule Kernel for linear attention decode.

    Args:
        seq_len: Sequence length (1, 2, 3, or 4)
        head_dim: Head dimension (64 or 128)
    """

    def __init__(self, seq_len: int, head_dim: int = 128):
        assert seq_len in [1, 2, 3, 4], f"Supported seq_len: 1,2,3,4, got {seq_len}"
        assert head_dim in [64, 128], f"Supported head_dim: 64,128, got {head_dim}"
        self.seq_len = seq_len
        self.head_dim = head_dim
        self._compiled_kernel = None

    def _get_launch_fn(self):
        if self.seq_len == 1:
            return gated_delta_rule_launch_seqlen1
        elif self.seq_len == 2:
            return gated_delta_rule_launch_seqlen2
        elif self.seq_len == 3:
            return gated_delta_rule_launch_seqlen3
        else:
            return gated_delta_rule_launch_seqlen4


# ==============================================================================
# PUBLIC API
# ==============================================================================

_compiled_kernels = {}  # Cache: (T, B, H, HV, K, V) -> compiled kernel


def gated_delta_rule(
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    softplus_beta: float = 1.0,
    softplus_threshold: float = 20.0,
    q: Optional[torch.Tensor] = None,
    k: Optional[torch.Tensor] = None,
    v: Optional[torch.Tensor] = None,
    b: Optional[torch.Tensor] = None,
    initial_state_source: Optional[torch.Tensor] = None,
    initial_state_indices: Optional[torch.Tensor] = None,
    use_qk_l2norm_in_kernel: bool = True,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Gated Delta Rule linear attention kernel.

    Implements the Gated Delta Rule mechanism for decode-phase inference,
    supporting sequence lengths T=1, T=2, T=3, T=4 and HEAD_DIM 64 or 128.

    Args:
        A_log: Log decay parameter [HV]
        a: Alpha gate input [B, T, HV]
        dt_bias: Delta-t bias [HV]
        softplus_beta: Softplus beta parameter (default: 1.0)
        softplus_threshold: Softplus threshold (default: 20.0)
        q: Query tensor [B, T, H, K] where K=HEAD_DIM (64 or 128)
        k: Key tensor [B, T, H, K]
        v: Value tensor [B, T, HV, V] where V=HEAD_DIM
        b: Beta gate input [B, T, HV]
        initial_state_source: H state [B, HV, V, K] (K-fast layout), modified in-place
        initial_state_indices: Not used (for compatibility)
        use_qk_l2norm_in_kernel: Whether to L2-normalize Q/K in kernel (default: True)
        scale: Optional attention scale (default: 1/sqrt(K))

    Returns:
        output: [B, T, HV, V]
    """
    global _compiled_kernels

    # Validate required Optional parameters
    if q is None:
        raise ValueError("q (query tensor) is required")
    if k is None:
        raise ValueError("k (key tensor) is required")
    if v is None:
        raise ValueError("v (value tensor) is required")
    if b is None:
        raise ValueError("b (beta gate tensor) is required")
    if initial_state_source is None:
        raise ValueError("initial_state_source (H state tensor) is required")

    B, T, H, K = q.shape
    assert T in [1, 2, 3, 4], f"Supported T=1,2,3,4, got T={T}"
    assert K in [64, 128], f"Supported K=64,128, got K={K}"
    HV = v.shape[2]
    V = v.shape[3]
    HEAD_DIM = K

    if scale is None:
        scale = 1.0 / math.sqrt(K)

    output = torch.empty(B, T, HV, V, device=q.device, dtype=q.dtype)

    q_ = from_dlpack(q, assumed_align=32, enable_tvm_ffi=True)
    k_ = from_dlpack(k, assumed_align=32, enable_tvm_ffi=True)
    v_ = from_dlpack(v, assumed_align=32, enable_tvm_ffi=True)
    a_ = from_dlpack(a, assumed_align=32, enable_tvm_ffi=True)
    b_ = from_dlpack(b, assumed_align=32, enable_tvm_ffi=True)
    A_log_ = from_dlpack(A_log, assumed_align=32, enable_tvm_ffi=True)
    dt_bias_ = from_dlpack(dt_bias, assumed_align=32, enable_tvm_ffi=True)
    h_ = from_dlpack(initial_state_source, assumed_align=32, enable_tvm_ffi=True)
    o_ = from_dlpack(output, assumed_align=32, enable_tvm_ffi=True)

    scale_f32 = cutlass.Float32(scale)
    softplus_beta_f32 = cutlass.Float32(softplus_beta)
    softplus_threshold_f32 = cutlass.Float32(softplus_threshold)
    eps_f32 = cutlass.Float32(1e-6)

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    # Check cache — key includes HEAD_DIM
    cache_key = (T, B, H, HV, K, V)
    if cache_key not in _compiled_kernels:
        # Select and compile the appropriate kernel
        if T == 1 and B <= 4:
            launch_fn = gated_delta_rule_launch_seqlen1_lowBS_1chunk
        elif T == 1:
            launch_fn = gated_delta_rule_launch_seqlen1
        elif T == 2:
            launch_fn = gated_delta_rule_launch_seqlen2
        elif T == 3:
            launch_fn = gated_delta_rule_launch_seqlen3
        else:  # T == 4
            launch_fn = gated_delta_rule_launch_seqlen4

        _compiled_kernels[cache_key] = cute.compile(
            launch_fn,
            q_,
            k_,
            v_,
            a_,
            b_,
            A_log_,
            dt_bias_,
            h_,
            o_,
            scale_f32,
            softplus_beta_f32,
            softplus_threshold_f32,
            eps_f32,
            stream,
            HEAD_DIM,
            options="--enable-tvm-ffi --generate-line-info",
        )

    # Execute
    _compiled_kernels[cache_key](
        q_,
        k_,
        v_,
        a_,
        b_,
        A_log_,
        dt_bias_,
        h_,
        o_,
        scale_f32,
        softplus_beta_f32,
        softplus_threshold_f32,
        eps_f32,
        stream,
    )

    return output
