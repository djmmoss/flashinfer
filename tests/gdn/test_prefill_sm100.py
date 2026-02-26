"""SM100 CuTe DSL GDN prefill kernel tests.

Direct kernel tests run in subprocesses to work around CuTe DSL JIT caching bug
(CUDA_ERROR_MISALIGNED_ADDRESS with different seq lengths in same process).

API integration tests call chunk_gated_delta_rule() and verify SM100 dispatch.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys

import torch
import pytest

from flashinfer.utils import is_sm100a_supported


def _skip_if_not_sm100():
    if not is_sm100a_supported(torch.device("cuda")):
        pytest.skip("Requires SM100a (Blackwell)")


def _run_test_subprocess(test_args: dict):
    """Run a single test case in a subprocess."""
    args_json = json.dumps(test_args)
    # The subprocess script imports the kernel and reference, runs the test
    script = f'''
import json, torch, torch.nn.functional as F
args = json.loads("""{args_json}""")

from flashinfer.gdn_kernels.gdn_prefill_sm100 import chunk_gated_delta_rule_sm100
from tests.gdn.test_gdn_prefill_sm100_ref import recurrent_ref

H = args["H"]
D = args["D"]
mask_p = args["mask_p"]
cu_seqlens = args["cu_seqlens"]
dtype = getattr(torch, args["dtype"])
use_initial_state = args.get("use_initial_state", True)

torch.manual_seed(42)
device = "cuda"
T = cu_seqlens[-1]
N = len(cu_seqlens) - 1

q = torch.randn((1, T, H, D), dtype=dtype, device=device)
k = F.normalize(torch.randn(1, T, H, D, dtype=torch.float32, device=device), p=2, dim=-1).to(dtype)
v = torch.randn((1, T, H, D), dtype=dtype, device=device)
g = F.logsigmoid(torch.rand(1, T, H, dtype=dtype, device=device))
if mask_p > 0:
    g = g * (torch.rand_like(g) > mask_p)
beta = torch.rand(1, T, H, dtype=dtype, device=device).sigmoid()

h0 = torch.randn((N, H, D, D), dtype=torch.float32, device=device) if use_initial_state else None
cu_seqlens_t = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)

o = torch.zeros_like(v)
state_out = torch.zeros(N, H, D, D, dtype=torch.float32, device=device) if use_initial_state else None

chunk_gated_delta_rule_sm100(
    q, k, v, g, beta,
    o, None, h0, cu_seqlens_t, state_out,
)

ref_o, ref_state = recurrent_ref(q, k, v, g, beta, h0, cu_seqlens)

torch.testing.assert_close(ref_o, o.to(torch.float), atol=5e-3, rtol=1e-3)
if state_out is not None and ref_state is not None:
    torch.testing.assert_close(ref_state, state_out.to(torch.float), atol=5e-3, rtol=1e-3)
print("PASS")
'''
    # Run from repo root so imports work
    repo_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=repo_root,
    )
    assert result.returncode == 0, (
        f"Test failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "PASS" in result.stdout


TEST_CASES = [
    {
        "H": 4,
        "D": 128,
        "mask_p": 0,
        "cu_seqlens": [0, 8192],
        "dtype": "float16",
    },
    {
        "H": 4,
        "D": 128,
        "mask_p": 0,
        "cu_seqlens": [0, 15],
        "dtype": "float16",
    },
    {
        "H": 4,
        "D": 128,
        "mask_p": 0,
        "cu_seqlens": [0, 256, 500, 1000],
        "dtype": "float16",
    },
    {
        "H": 4,
        "D": 128,
        "mask_p": 0.5,
        "cu_seqlens": [0, 256, 500, 1000],
        "dtype": "float16",
    },
    {
        "H": 4,
        "D": 60,
        "mask_p": 0,
        "cu_seqlens": [0, 15, 100, 300, 1200, 2000],
        "dtype": "float16",
    },
    {
        "H": 4,
        "D": 64,
        "mask_p": 0,
        "cu_seqlens": [0, 256, 500, 1000],
        "dtype": "float16",
    },
    {
        "H": 4,
        "D": 100,
        "mask_p": 0,
        "cu_seqlens": [0, 15, 100, 300, 1200, 2000],
        "dtype": "float16",
    },
]


@pytest.mark.parametrize(
    "test_args",
    TEST_CASES,
    ids=[
        f"H{c['H']}_D{c['D']}_mp{c['mask_p']}_N{len(c['cu_seqlens']) - 1}"
        for c in TEST_CASES
    ],
)
def test_prefill_sm100(test_args):
    _skip_if_not_sm100()
    _run_test_subprocess(test_args)


@pytest.mark.parametrize(
    "test_args",
    [{**c, "use_initial_state": False} for c in TEST_CASES[:3]],
    ids=[
        f"no_state_H{c['H']}_D{c['D']}_N{len(c['cu_seqlens']) - 1}"
        for c in TEST_CASES[:3]
    ],
)
def test_prefill_sm100_no_initial_state(test_args):
    _skip_if_not_sm100()
    _run_test_subprocess(test_args)


# === API Integration Tests ===
# These test the SM100 kernel through the FlashInfer public API
# (chunk_gated_delta_rule), verifying dispatch + gate conversion + layout transpose.


@pytest.mark.parametrize("beta", [True])
@pytest.mark.parametrize("alpha", [True])
@pytest.mark.parametrize("head_size", [128])
@pytest.mark.parametrize(
    "num_q_heads, num_k_heads, num_v_heads",
    [
        (4, 4, 4),  # Equal heads
        (4, 4, 8),  # GVA: v > q=k (supported by SM100)
    ],
)
@pytest.mark.parametrize("seq_lens", [[128], [256, 256], [128, 256, 512]])
@pytest.mark.parametrize("dtype", ["float16"])
def test_prefill_sm100_api(
    qkv_factory,
    dtype,
    num_q_heads,
    num_k_heads,
    num_v_heads,
    head_size,
    seq_lens,
    alpha,
    beta,
):
    """Test SM100 kernel through the FlashInfer public API."""
    _skip_if_not_sm100()
    from .test_prefill_delta_rule import _test_prefill_kernel

    _test_prefill_kernel(
        qkv_factory,
        dtype,
        num_q_heads,
        num_k_heads,
        num_v_heads,
        head_size,
        128,
        seq_lens,
        1.0 / math.sqrt(head_size),
        alpha,
        beta,
        seed=0,
    )
