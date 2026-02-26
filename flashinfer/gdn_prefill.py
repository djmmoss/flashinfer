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

import functools
from types import SimpleNamespace
from typing import Optional, Union, Tuple
import torch

from .api_logging import flashinfer_api
from .jit.gdn import gen_gdn_prefill_sm90_module
from .utils import (
    register_custom_op,
    register_fake_op,
    get_compute_capability,
    get_device_sm_count,
    _get_cache_buf,
)

try:
    from .gdn_kernels import chunk_gated_delta_rule_sm100, _has_cute_dsl_prefill
except ImportError:
    _has_cute_dsl_prefill = False
    chunk_gated_delta_rule_sm100 = None


def _can_use_sm100_prefill(device, num_q_heads, num_k_heads, num_v_heads):
    """Check if SM100 CuTe DSL kernel can handle this config."""
    if not _has_cute_dsl_prefill:
        return False
    cc = get_compute_capability(device)
    if cc[0] < 10:
        return False
    # SM100 kernel requires num_q_heads == num_k_heads (no separate GQA dispatch)
    # and h_v divisible by h_q
    if num_q_heads != num_k_heads:
        return False
    if num_v_heads % num_q_heads != 0:
        return False
    return True


@functools.cache
def get_gdn_prefill_module():
    module = gen_gdn_prefill_sm90_module().build_and_load()

    @register_custom_op(
        "flashinfer::gdn_prefill", mutates_args=("output", "output_state")
    )
    def gdn_prefill(
        output: torch.Tensor,
        output_state: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: torch.Tensor,
        initial_state: Optional[torch.Tensor],
        g: Optional[torch.Tensor],
        beta: Optional[torch.Tensor],
        scale: float,
        workspace_buffer: torch.Tensor,
    ) -> None:
        module.gdn_prefill(
            output,
            output_state,
            q,
            k,
            v,
            cu_seqlens,
            initial_state,
            g,
            beta,
            scale,
            workspace_buffer,
        )

    @register_fake_op("flashinfer::gdn_prefill")
    def _fake_gdn_prefill(
        output: torch.Tensor,
        output_state: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: torch.Tensor,
        initial_state: Optional[torch.Tensor],
        g: Optional[torch.Tensor],
        beta: Optional[torch.Tensor],
        scale: float,
        workspace_buffer: torch.Tensor,
    ) -> None:
        pass

    return SimpleNamespace(gdn_prefill=gdn_prefill)


@flashinfer_api
def chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    beta: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
    use_qk_l2norm_in_kernel: bool = False,
    output: Optional[torch.Tensor] = None,
    output_state: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    r"""Chunked Gated Delta Rule (GDN) attention for prefill.

    This implements the gated delta rule linear attention mechanism for efficient
    training and inference. Supports both GQA (grouped query attention) and GVA
    (grouped value attention) configurations.

    Args:
        q (torch.Tensor):
            Queries of shape ``[total_seq_len, num_q_heads, head_size]``.
            Must be contiguous and on CUDA.
        k (torch.Tensor):
            Keys of shape ``[total_seq_len, num_k_heads, head_size]``.
            Must be contiguous and on CUDA.
        v (torch.Tensor):
            Values of shape ``[total_seq_len, num_v_heads, head_size]``.
            Must be contiguous and on CUDA.
        g (Optional[torch.Tensor]):
            Forget gate (alpha) of shape ``[total_seq_len, num_sab_heads]`` where
            ``num_sab_heads = max(num_q_heads, num_v_heads)``. Must be float32.
            If None, defaults to all ones. Default: ``None``.
        beta (Optional[torch.Tensor]):
            Update gate (beta) of shape ``[total_seq_len, num_sab_heads]``.
            Must be float32. If None, defaults to all ones. Default: ``None``.
        scale (Optional[float]):
            Scale factor for the attention scores.
            If not provided, defaults to ``1 / sqrt(head_size)``. Default: ``None``.
        initial_state (Optional[torch.Tensor]):
            Initial KV state of shape ``[num_seqs, num_sab_heads, head_size, head_size]``.
            Must be float32. If None, starts from zero state. Default: ``None``.
        output_final_state (bool):
            Whether to output the final state. Default: ``False``.
        cu_seqlens (torch.Tensor):
            Cumulative sequence lengths of shape ``[num_seqs + 1]``, int64.
            Required for variable-length sequences (varlen mode).
        use_qk_l2norm_in_kernel (bool):
            Whether to use QK L2 normalization in kernel. Default: ``False``.
        output (Optional[torch.Tensor]):
            Pre-allocated output tensor of shape ``[total_seq_len, num_o_heads, head_size]``
            where ``num_o_heads = max(num_q_heads, num_v_heads)``.
            If None, will be allocated automatically. Default: ``None``.
        output_state (Optional[torch.Tensor]):
            Pre-allocated output state tensor of shape
            ``[num_seqs, num_sab_heads, head_size, head_size]``, float32.
            Required if ``output_final_state=True``. Default: ``None``.

    Returns:
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
            - If ``output_final_state=False``: Returns output tensor of shape
              ``[total_seq_len, num_o_heads, head_size]``.
            - If ``output_final_state=True``: Returns tuple of (output, final_state) where
              final_state has shape ``[num_seqs, num_sab_heads, head_size, head_size]``.

    Note:
        - Supports GQA: ``num_q_heads > num_k_heads = num_v_heads``
        - Supports GVA: ``num_v_heads > num_q_heads = num_k_heads``
        - The final state is in k-last layout ``[N, H, V, K]``.
        - Requires SM90 (Hopper) or SM100 (Blackwell) architecture.
        - On SM100+ with CuTe DSL installed, uses an optimized SM100 kernel.
    """
    assert cu_seqlens is not None, "cu_seqlens is required for varlen mode"

    num_seqs = cu_seqlens.size(0) - 1
    total_seq_len = q.size(0)
    num_q_heads = q.size(1)
    num_v_heads = v.size(1)
    head_size = q.size(2)
    num_o_heads = max(num_q_heads, num_v_heads)
    num_sab_heads = num_o_heads

    # Allocate output if not provided
    if output is None:
        output = torch.empty(
            (total_seq_len, num_o_heads, head_size),
            dtype=q.dtype,
            device=q.device,
        )

    # Allocate output_state if needed
    if output_final_state and output_state is None:
        output_state = torch.empty(
            (num_seqs, num_sab_heads, head_size, head_size),
            dtype=torch.float32,
            device=q.device,
        )
    elif not output_final_state and output_state is None:
        # Still need to allocate since kernel always writes state
        output_state = torch.empty(
            (num_seqs, num_sab_heads, head_size, head_size),
            dtype=torch.float32,
            device=q.device,
        )

    num_k_heads = k.size(1)

    # SM100 CuTe DSL dispatch path
    if _can_use_sm100_prefill(q.device, num_q_heads, num_k_heads, num_v_heads):
        # Convert gate format: FlashInfer alpha (linear) → SM100 g (log-space)
        if g is not None:
            g_sm100 = torch.log(g.float() + 1e-10).to(q.dtype)
        else:
            g_sm100 = torch.zeros(
                total_seq_len, num_sab_heads, dtype=q.dtype, device=q.device
            )

        if beta is not None:
            beta_sm100 = beta.to(q.dtype)
        else:
            beta_sm100 = torch.ones(
                total_seq_len, num_sab_heads, dtype=q.dtype, device=q.device
            )

        # Reshape 3D → 4D: [total_seq_len, H, D] → [1, total_seq_len, H, D]
        q_4d = q.unsqueeze(0)
        k_4d = k.unsqueeze(0)
        v_4d = v.unsqueeze(0)
        g_4d = g_sm100.unsqueeze(0)
        beta_4d = beta_sm100.unsqueeze(0)
        o_4d = output.unsqueeze(0)

        # SM100 kernel uses same state layout as FlashInfer (k-last [N,H,V,K]):
        # state_layout stride=(head_dim, 1) maps dim-2=V, dim-3=K, matching contiguous [N,H,V,K].
        chunk_gated_delta_rule_sm100(
            q_4d,
            k_4d,
            v_4d,
            g_4d,
            beta_4d,
            o_4d,
            scale,
            initial_state,
            cu_seqlens.to(torch.int32),
            output_state,
        )

        if output_final_state:
            return output, output_state
        else:
            return output

    # SM90 CUTLASS kernel path
    # Prepare workspace buffer for TMA Store in kernel
    # 128B tensormap for each SM on Hopper architecture
    workspace_size = get_device_sm_count(q.device) * 128
    workspace_buffer = _get_cache_buf("gdn_prefill_workspace", workspace_size, q.device)

    get_gdn_prefill_module().gdn_prefill(
        output,
        output_state,
        q,
        k,
        v,
        cu_seqlens.to(torch.int64),  # C++ kernel expects int64
        initial_state,
        g,
        beta,
        scale if scale is not None else 0.0,
        workspace_buffer,
    )

    if output_final_state:
        return output, output_state
    else:
        return output
