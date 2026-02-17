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
"""

"""
Shared CuTe DSL Helpers for GDN and KDA Decode Kernels
======================================================

Gate-independent helper functions used by both GDN (scalar gate) and KDA
(per-K gate) decode kernels. All functions parameterized by HEAD_DIM
where needed.

Functions:
- write_h_chunk_to_smem: F32 register H chunk → BF16 SMEM
- store_h_smem_to_gmem: SMEM → GMEM (128-bit stores)
- load_h_chunk_async: GMEM → SMEM async copy
- normalize_and_store_qk_to_smem: L2-normalize Q/K, store to SMEM
- load_v_to_smem: Load V to SMEM
- load_kq_chunk_from_smem: Load K/Q chunk from SMEM to registers
- update_h_with_delta: Delta rule state update
- compute_output: Output from state
- cross_warp_reduce_single / cross_warp_reduce_two: Bank-conflict-free warp reductions

Constant:
- H_SMEM_PADDING = 8
"""

import cutlass
import cutlass.cute as cute

# ==============================================================================
# CONSTANTS
# ==============================================================================
H_SMEM_PADDING = 8


# ==============================================================================
# SHARED HELPER FUNCTIONS
# ==============================================================================


@cute.jit
def write_h_chunk_to_smem(h_chunk_f32, h_sh_chunk, lane_idx, k_base):
    """Write F32 register H chunk to BF16 SMEM."""
    for i in cutlass.range_constexpr(32):
        h_sh_chunk[lane_idx, k_base + i] = h_chunk_f32[i].to(cutlass.BFloat16)


@cute.jit
def store_h_smem_to_gmem(
    h_sh_chunk, h_out, tidx, v_row_offset, HEAD_DIM: cutlass.Constexpr[int]
):
    """Store H from SMEM to GMEM using 128-bit stores."""
    copy_bits = 128
    copy_elems = copy_bits // cutlass.BFloat16.width

    from cutlass.cute.nvgpu import CopyUniversalOp

    if HEAD_DIM == 64:
        # 64 threads: use (8, 8) thread layout, (8, 64) tiles, 4 row iterations
        thr_layout = cute.make_layout((8, 8), stride=(8, 1))
        val_layout = cute.make_layout((1, copy_elems))
        atom_store = cute.make_copy_atom(
            CopyUniversalOp(), cutlass.BFloat16, num_bits_per_copy=copy_bits
        )
        tiled_copy = cute.make_tiled_copy_tv(atom_store, thr_layout, val_layout)
        thr_copy = tiled_copy.get_slice(tidx)
        for row_iter in cutlass.range_constexpr(4):
            s_tile = cute.local_tile(h_sh_chunk, (8, 64), (row_iter, 0))
            g_tile = cute.local_tile(
                h_out, (8, 64), (row_iter + (v_row_offset // 8), 0)
            )
            tS = thr_copy.partition_S(s_tile)
            tD = thr_copy.partition_D(g_tile)
            cute.copy(atom_store, tS, tD)
    elif HEAD_DIM == 128:
        # 128 threads: use (16, 8) thread layout, (16, 64) tiles, 2x2 iterations
        thr_layout = cute.make_layout((16, 8), stride=(8, 1))
        val_layout = cute.make_layout((1, copy_elems))
        atom_store = cute.make_copy_atom(
            CopyUniversalOp(), cutlass.BFloat16, num_bits_per_copy=copy_bits
        )
        tiled_copy = cute.make_tiled_copy_tv(atom_store, thr_layout, val_layout)
        thr_copy = tiled_copy.get_slice(tidx)
        for row_iter in cutlass.range_constexpr(2):
            for col_iter in cutlass.range_constexpr(2):
                s_tile = cute.local_tile(h_sh_chunk, (16, 64), (row_iter, col_iter))
                g_tile = cute.local_tile(
                    h_out, (16, 64), (row_iter + (v_row_offset // 16), col_iter)
                )
                tS = thr_copy.partition_S(s_tile)
                tD = thr_copy.partition_D(g_tile)
                cute.copy(atom_store, tS, tD)


@cute.jit
def load_h_chunk_async(
    h_sh_chunk, h_global, tidx, row_offset, HEAD_DIM: cutlass.Constexpr[int]
):
    """Load H chunk from GMEM to SMEM using async copy."""
    copy_bits = 128
    copy_elems = copy_bits // cutlass.BFloat16.width

    if HEAD_DIM == 64:
        # 64 threads: use (8, 8) thread layout, (8, 64) tiles, 4 row iterations
        thr_layout = cute.make_layout((8, 8), stride=(8, 1))
        val_layout = cute.make_layout((1, copy_elems))
        atom_async_copy = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(
                cache_mode=cute.nvgpu.cpasync.LoadCacheMode.GLOBAL
            ),
            cutlass.BFloat16,
            num_bits_per_copy=copy_bits,
        )
        tiled_copy = cute.make_tiled_copy_tv(atom_async_copy, thr_layout, val_layout)
        thr_copy = tiled_copy.get_slice(tidx)
        for row_iter in cutlass.range_constexpr(4):
            g_tile = cute.local_tile(
                h_global, (8, 64), (row_iter + (row_offset // 8), 0)
            )
            s_tile = cute.local_tile(h_sh_chunk, (8, 64), (row_iter, 0))
            tS = thr_copy.partition_S(g_tile)
            tD = thr_copy.partition_D(s_tile)
            cute.copy(atom_async_copy, tS, tD)
    elif HEAD_DIM == 128:
        # 128 threads: use (16, 8) thread layout, (16, 64) tiles, 2x2 iterations
        thr_layout = cute.make_layout((16, 8), stride=(8, 1))
        val_layout = cute.make_layout((1, copy_elems))
        atom_async_copy = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(
                cache_mode=cute.nvgpu.cpasync.LoadCacheMode.GLOBAL
            ),
            cutlass.BFloat16,
            num_bits_per_copy=copy_bits,
        )
        tiled_copy = cute.make_tiled_copy_tv(atom_async_copy, thr_layout, val_layout)
        thr_copy = tiled_copy.get_slice(tidx)
        for row_iter in cutlass.range_constexpr(2):
            for col_iter in cutlass.range_constexpr(2):
                g_tile = cute.local_tile(
                    h_global, (16, 64), (row_iter + (row_offset // 16), col_iter)
                )
                s_tile = cute.local_tile(h_sh_chunk, (16, 64), (row_iter, col_iter))
                tS = thr_copy.partition_S(g_tile)
                tD = thr_copy.partition_D(s_tile)
                cute.copy(atom_async_copy, tS, tD)


@cute.jit
def normalize_and_store_qk_to_smem(
    q_head, k_head, q_sh, k_sh, lane_idx, scale, eps, HEAD_DIM: cutlass.Constexpr[int]
):
    """L2-normalize Q and K vectors, then store to shared memory."""
    # ELEMS_PER_LANE = HEAD_DIM // 32 (2 for HD=64, 4 for HD=128)
    q_reg = cute.make_rmem_tensor((HEAD_DIM // 32,), cutlass.Float32)
    k_reg = cute.make_rmem_tensor((HEAD_DIM // 32,), cutlass.Float32)

    for i in cutlass.range_constexpr(HEAD_DIM // 32):
        q_reg[i] = q_head[lane_idx + i * 32].to(cutlass.Float32)
        k_reg[i] = k_head[lane_idx + i * 32].to(cutlass.Float32)

    q_sum_sq = cutlass.Float32(0.0)
    k_sum_sq = cutlass.Float32(0.0)
    q_sum_sq2 = cutlass.Float32(0.0)
    k_sum_sq2 = cutlass.Float32(0.0)

    for i in cutlass.range_constexpr(0, HEAD_DIM // 32, 2):
        q_sum_sq, q_sum_sq2 = cute.arch.fma_packed_f32x2(
            src_a=(q_reg[i], q_reg[i + 1]),
            src_b=(q_reg[i], q_reg[i + 1]),
            src_c=(q_sum_sq, q_sum_sq2),
        )
        k_sum_sq, k_sum_sq2 = cute.arch.fma_packed_f32x2(
            src_a=(k_reg[i], k_reg[i + 1]),
            src_b=(k_reg[i], k_reg[i + 1]),
            src_c=(k_sum_sq, k_sum_sq2),
        )

    q_sum_sq = q_sum_sq + q_sum_sq2
    k_sum_sq = k_sum_sq + k_sum_sq2

    for i in cutlass.range_constexpr(5):
        q_sum_sq = q_sum_sq + cute.arch.shuffle_sync_bfly(
            q_sum_sq, offset=1 << i, mask=0xFFFFFFFF
        )
        k_sum_sq = k_sum_sq + cute.arch.shuffle_sync_bfly(
            k_sum_sq, offset=1 << i, mask=0xFFFFFFFF
        )

    q_norm = cute.rsqrt(q_sum_sq + eps, fastmath=True)
    k_norm = cute.rsqrt(k_sum_sq + eps, fastmath=True)
    q_scale_factor = q_norm * scale

    for i in cutlass.range_constexpr(HEAD_DIM // 32):
        q_sh[lane_idx + i * 32] = q_reg[i] * q_scale_factor
        k_sh[lane_idx + i * 32] = k_reg[i] * k_norm


@cute.jit
def load_v_to_smem(v_head, v_sh, tidx):
    """Load V values from GMEM to SMEM."""
    v_sh[tidx] = v_head[tidx].to(cutlass.Float32)


@cute.jit
def load_kq_chunk_from_smem(kq_sh, kq_chunk, k_base):
    """Load K or Q chunk from SMEM to registers."""
    for i in cutlass.range_constexpr(32):
        kq_chunk[i] = kq_sh[k_base + i]


@cute.jit
def update_h_with_delta(h_chunk, kq_chunk, v_delta):
    """Update H with delta: h = h + k * v_delta."""
    for i in cutlass.range_constexpr(0, 32, 2):
        h_chunk[i], h_chunk[i + 1] = cute.arch.fma_packed_f32x2(
            src_a=(kq_chunk[i], kq_chunk[i + 1]),
            src_b=(v_delta, v_delta),
            src_c=(h_chunk[i], h_chunk[i + 1]),
        )


@cute.jit
def compute_output(h_chunk, kq_chunk):
    """Compute output = sum_k(h * q)."""
    out = cutlass.Float32(0.0)
    out2 = cutlass.Float32(0.0)
    for i in cutlass.range_constexpr(0, 32, 2):
        out, out2 = cute.arch.fma_packed_f32x2(
            src_a=(h_chunk[i], h_chunk[i + 1]),
            src_b=(kq_chunk[i], kq_chunk[i + 1]),
            src_c=(out, out2),
        )
    out = out + out2
    return out


@cute.jit
def cross_warp_reduce_single(
    reduce_sh, slot, warp_idx, lane_idx, value, NUM_WARPS: cutlass.Constexpr[int]
):
    """
    Cross-warp reduction for a single value using bank-conflict-free layout.
    Layout: [slot, lane_idx, warp_idx]
    """
    reduce_sh[slot, lane_idx, warp_idx] = value
    cute.arch.sync_threads()
    reduced_value = cutlass.Float32(0.0)
    if NUM_WARPS == 2:
        reduced_value = reduce_sh[slot, lane_idx, 0] + reduce_sh[slot, lane_idx, 1]
    elif NUM_WARPS == 4:
        reduced_value = (
            reduce_sh[slot, lane_idx, 0]
            + reduce_sh[slot, lane_idx, 1]
            + reduce_sh[slot, lane_idx, 2]
            + reduce_sh[slot, lane_idx, 3]
        )
    return reduced_value


@cute.jit
def cross_warp_reduce_two(
    reduce_sh,
    slot1,
    slot2,
    warp_idx,
    lane_idx,
    value1,
    value2,
    NUM_WARPS: cutlass.Constexpr[int],
):
    """
    Cross-warp reduction for two values simultaneously using bank-conflict-free layout.
    Layout: [slot, lane_idx, warp_idx]
    """
    reduce_sh[slot1, lane_idx, warp_idx] = value1
    reduce_sh[slot2, lane_idx, warp_idx] = value2
    cute.arch.sync_threads()
    reduced1 = cutlass.Float32(0.0)
    reduced2 = cutlass.Float32(0.0)
    if NUM_WARPS == 2:
        reduced1 = reduce_sh[slot1, lane_idx, 0] + reduce_sh[slot1, lane_idx, 1]
        reduced2 = reduce_sh[slot2, lane_idx, 0] + reduce_sh[slot2, lane_idx, 1]
    elif NUM_WARPS == 4:
        reduced1 = (
            reduce_sh[slot1, lane_idx, 0]
            + reduce_sh[slot1, lane_idx, 1]
            + reduce_sh[slot1, lane_idx, 2]
            + reduce_sh[slot1, lane_idx, 3]
        )
        reduced2 = (
            reduce_sh[slot2, lane_idx, 0]
            + reduce_sh[slot2, lane_idx, 1]
            + reduce_sh[slot2, lane_idx, 2]
            + reduce_sh[slot2, lane_idx, 3]
        )
    return reduced1, reduced2
