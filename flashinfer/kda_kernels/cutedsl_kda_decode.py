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
chunk_kda-compatible wrapper for the KDA CuTe DSL decode kernel.

Bridges the interface gap between fla's chunk_kda and the CuTe DSL kernel:
- State layout: chunk_kda [B, H, K, V] <-> CuTe DSL [B, H, V, K]
- All other tensors (q, k, v, g, beta) pass through directly
"""

import torch
from .kda_decode_bf16_state import kda_gated_delta_rule


def cutedsl_kda_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """CuTe DSL decode kernel with chunk_kda-compatible API.

    Args:
        q: [B, T, H, K] queries
        k: [B, T, H, K] keys
        v: [B, T, H, V] values (H=HV when no GQA)
        g: [B, T, H, K] log-space per-K gate
        beta: [B, T, H] pre-sigmoided delta-rule learning rate
        scale: attention scale (default: 1/sqrt(K))
        initial_state: [B, H, K, V] state (chunk_kda convention, K-first)
        output_final_state: whether to return the final state
        use_qk_l2norm_in_kernel: whether to L2-normalize Q/K in kernel

    Returns:
        o: [B, T, H, V] output
        final_state: [B, H, K, V] if output_final_state else None
    """
    if initial_state is None:
        raise ValueError("initial_state is required for KDA decode (cannot be None)")

    # Transpose state: chunk_kda [B,H,K,V] -> CuTe DSL [B,H,V,K]
    # Cast to bf16 since the CuTe DSL kernel expects bf16 storage
    state_vk = initial_state.to(torch.bfloat16).transpose(-1, -2).contiguous()

    o = kda_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state_source=state_vk,
        scale=scale,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )

    # Transpose state back: CuTe DSL [B,H,V,K] -> chunk_kda [B,H,K,V]
    # Return as float32 for compatibility with chunk_kda conventions
    final_state = (
        state_vk.float().transpose(-1, -2).contiguous() if output_final_state else None
    )
    return o, final_state
