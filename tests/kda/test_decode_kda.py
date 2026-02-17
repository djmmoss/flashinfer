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

KDA (Key-Driven Attention) decode kernel tests.

Tests the CuTe DSL KDA decode kernel against a naive reference implementation.
KDA differs from GDN by having per-K-dimension gates instead of scalar gates.
"""

from __future__ import annotations

import math

import torch
import pytest

from flashinfer.utils import get_compute_capability
from flashinfer.kda_kernels.kda_decode_bf16_state import (
    kda_gated_delta_rule as kda_kernel,
)


# BF16 tolerances
ATOL = 1e-1
RTOL = 5e-2

# Fixed head configuration
NUM_Q_HEADS = 16
NUM_K_HEADS = 16
NUM_V_HEADS = 32


def _skip_if_not_sm90_or_later():
    """Skip test if not Hopper (SM90+) or Blackwell (SM100+) architecture."""
    cc = get_compute_capability(torch.device("cuda"))
    if cc[0] < 9:
        pytest.skip(f"KDA decode requires SM90+, but got SM{cc[0]}{cc[1]}")


# ============================================================================
# Naive KDA decode reference (self-contained, no external deps beyond torch)
# ============================================================================


def kda_decode_reference(q, k, v, g, beta, state, scale=None, use_qk_l2norm=True):
    """
    Naive KDA decode reference with per-K gating.

    Args:
        q: [B, T, H, K] query
        k: [B, T, H, K] key
        v: [B, T, HV, V] value
        g: [B, T, HV, K] per-K log-space gate
        beta: [B, T, HV] pre-sigmoided learning rate
        state: [B, HV, V, K] (V-major, K-last) — cloned internally
        scale: attention scale (default: 1/sqrt(K))
        use_qk_l2norm: whether to L2-normalize Q and K

    Returns:
        output: [B, T, HV, V] bfloat16
        state: [B, HV, V, K] float32 (updated)
    """
    B, T, H, K = q.shape
    HV = v.shape[2]
    if scale is None:
        scale = 1.0 / math.sqrt(K)

    state = state.float().clone()  # [B, HV, V, K]
    outputs = []

    for t in range(T):
        qt = q[:, t].float()  # [B, H, K]
        kt = k[:, t].float()  # [B, H, K]
        vt = v[:, t].float()  # [B, HV, V]
        gt = g[:, t].float()  # [B, HV, K]
        bt = beta[:, t].float()  # [B, HV]

        # L2 normalize Q, K if requested
        if use_qk_l2norm:
            qt = torch.nn.functional.normalize(qt, dim=-1)
            kt = torch.nn.functional.normalize(kt, dim=-1)

        # Per-K decay: state[b,h,v,k] *= exp(g[b,h,k])
        g_exp = torch.exp(gt)  # [B, HV, K]
        state = state * g_exp.unsqueeze(2)  # [B, HV, V, K] * [B, HV, 1, K]

        # Prediction: k @ state^T -> [B, HV, V]
        # GQA: HV >= H, each Q/K head shared across (HV // H) V-heads
        v_per_q = HV // H  # e.g. 32 // 16 = 2
        # Expand Q/K heads to match V-heads: repeat each Q/K head v_per_q times
        kt_v = kt.repeat_interleave(v_per_q, dim=1)  # [B, HV, K]
        pred = torch.einsum("bhk,bhvk->bhv", kt_v, state)  # [B, HV, V]

        # Delta rule update
        v_err = vt - pred  # [B, HV, V]
        state = state + bt.unsqueeze(-1).unsqueeze(-1) * torch.einsum(
            "bhk,bhv->bhvk", kt_v, v_err
        )

        # Output: q @ state^T -> [B, HV, V]
        qt_v = qt.repeat_interleave(v_per_q, dim=1)  # [B, HV, K]
        out_t = torch.einsum("bhk,bhvk->bhv", qt_v, state) * scale
        outputs.append(out_t)

    output = torch.stack(outputs, dim=1)  # [B, T, HV, V]
    return output.to(torch.bfloat16), state


# ============================================================================
# Data generation helpers
# ============================================================================


def _generate_kda_inputs(batch_size, seq_len, head_dim, device="cuda", seed=42):
    """Generate random KDA inputs with the correct shapes and dtypes."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    B, T, K, V = batch_size, seq_len, head_dim, head_dim
    H, HV = NUM_Q_HEADS, NUM_V_HEADS

    q = torch.randn(B, T, H, K, device=device, dtype=torch.bfloat16) * 0.1
    k = torch.randn(B, T, H, K, device=device, dtype=torch.bfloat16) * 0.1
    v = torch.randn(B, T, HV, V, device=device, dtype=torch.bfloat16) * 0.1
    # Log-space gate: small negative values so exp(g) is a decay < 1
    g = -torch.rand(B, T, HV, K, device=device, dtype=torch.bfloat16) * 0.5
    beta = torch.rand(B, T, HV, device=device, dtype=torch.bfloat16) * 0.5
    state = torch.randn(B, HV, V, K, device=device, dtype=torch.bfloat16) * 0.01

    return q, k, v, g, beta, state


# ============================================================================
# Tests
# ============================================================================


@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("seq_len", [1, 2, 3, 4])
@pytest.mark.parametrize("batch_size", [1, 2, 4, 8, 16, 32, 64, 128])
def test_kda_decode_correctness(batch_size, seq_len, head_dim):
    """KDA kernel output matches naive reference for all T and HEAD_DIM combos."""
    _skip_if_not_sm90_or_later()

    q, k, v, g, beta, state = _generate_kda_inputs(
        batch_size, seq_len, head_dim, seed=42
    )

    # Reference
    ref_output, _ref_state = kda_decode_reference(
        q, k, v, g, beta, state, use_qk_l2norm=True
    )

    # Kernel (state modified in-place, so clone)
    kernel_state = state.clone()
    kernel_output = kda_kernel(
        q, k, v, g, beta, kernel_state, scale=None, use_qk_l2norm_in_kernel=True
    )

    torch.testing.assert_close(
        kernel_output,
        ref_output,
        atol=ATOL,
        rtol=RTOL,
        msg=f"Output mismatch: B={batch_size}, T={seq_len}, HD={head_dim}",
    )


@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
def test_kda_state_update(batch_size, head_dim):
    """Verify that the kernel modifies state in-place."""
    _skip_if_not_sm90_or_later()

    q, k, v, g, beta, state = _generate_kda_inputs(
        batch_size, seq_len=1, head_dim=head_dim, seed=123
    )

    state_before = state.clone()
    _ = kda_kernel(q, k, v, g, beta, state, scale=None, use_qk_l2norm_in_kernel=True)

    # State should have been modified in-place
    assert not torch.equal(state, state_before), (
        f"State was not modified in-place: B={batch_size}, HD={head_dim}"
    )


@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("seq_len", [1, 2, 3, 4])
def test_kda_uniform_gate_determinism(seq_len, head_dim):
    """Uniform gate across K dims should produce deterministic results.

    When all K dimensions of the gate have the same value (uniform gate),
    we verify the kernel is deterministic (two runs match exactly) and
    that outputs match the naive reference implementation.
    """
    _skip_if_not_sm90_or_later()

    B = 4
    T, K, V = seq_len, head_dim, head_dim
    H, HV = NUM_Q_HEADS, NUM_V_HEADS
    device = "cuda"

    torch.manual_seed(99)
    torch.cuda.manual_seed(99)

    q = torch.randn(B, T, H, K, device=device, dtype=torch.bfloat16) * 0.1
    k = torch.randn(B, T, H, K, device=device, dtype=torch.bfloat16) * 0.1
    v = torch.randn(B, T, HV, V, device=device, dtype=torch.bfloat16) * 0.1
    beta = torch.rand(B, T, HV, device=device, dtype=torch.bfloat16) * 0.5
    state = torch.randn(B, HV, V, K, device=device, dtype=torch.bfloat16) * 0.01

    # Uniform gate: same scalar value broadcast to all K dims
    g_scalar = -torch.rand(B, T, HV, 1, device=device, dtype=torch.bfloat16) * 0.5
    g_uniform = g_scalar.expand(B, T, HV, K).contiguous()

    # Run kernel with uniform per-K gate
    state_a = state.clone()
    output_a = kda_kernel(
        q, k, v, g_uniform, beta, state_a, scale=None, use_qk_l2norm_in_kernel=True
    )

    # Run kernel again with the same uniform gate (fresh state)
    state_b = state.clone()
    output_b = kda_kernel(
        q, k, v, g_uniform, beta, state_b, scale=None, use_qk_l2norm_in_kernel=True
    )

    # Outputs should be bitwise identical (same inputs, deterministic kernel)
    torch.testing.assert_close(
        output_a,
        output_b,
        atol=0.0,
        rtol=0.0,
        msg=f"Uniform gate not deterministic: T={seq_len}, HD={head_dim}",
    )

    # Also verify against the naive reference with uniform gate
    ref_output, _ = kda_decode_reference(
        q, k, v, g_uniform, beta, state, use_qk_l2norm=True
    )

    torch.testing.assert_close(
        output_a,
        ref_output,
        atol=ATOL,
        rtol=RTOL,
        msg=f"Uniform gate vs reference mismatch: T={seq_len}, HD={head_dim}",
    )
