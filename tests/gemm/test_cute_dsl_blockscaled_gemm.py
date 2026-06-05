"""
This is the test file for MaskedBatchedMatmulCuteDSL kernel.
`test_blockscaled_gemm_python_interface` is the python interface test. For pytorch DLFW, refer to this.
"""

import math
from typing import Tuple

import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
import pytest
import torch
from cutlass.cute.runtime import from_dlpack

from flashinfer.gemm import (
    Sm100BlockScaledPersistentDenseGemmKernel,  # not used in python interface
    grouped_gemm_nt_masked,  # deepgemm-like python interface for DLFW integration
    create_scale_factor_tensor,
)
from flashinfer.cute_dsl.utils import (
    get_cutlass_dtype,
    get_num_sm,
    is_cute_dsl_available,
)


def _requires_sm100_or_sm103(device: torch.device) -> None:
    device_ver = torch.cuda.get_device_capability(device)
    if device_ver not in [(10, 0), (10, 3)]:
        pytest.skip(
            "CuTeDSL masked grouped GEMM is only supported on SM100/SM103, "
            f"got {device_ver}."
        )


def _masked_rows_are_finite(out: torch.Tensor, masked_m: torch.Tensor) -> bool:
    for batch_idx in range(masked_m.numel()):
        valid_m = int(masked_m[batch_idx].item())
        if valid_m < out.shape[0]:
            if not torch.isfinite(out[valid_m:, :, batch_idx].float()).all():
                return False
    return True


def _dirty_mxfp8_padded_rows(
    a_torch: torch.Tensor,
    sfa_torch: torch.Tensor,
    masked_m: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    dirty_a = a_torch.clone()
    dirty_sfa = sfa_torch.clone()
    l = masked_m.numel()
    m = dirty_a.shape[0]

    for batch_idx in range(l):
        valid_m = int(masked_m[batch_idx].item())
        if valid_m < m:
            dirty_a[valid_m:, :, batch_idx] = torch.randn_like(
                dirty_a[valid_m:, :, batch_idx].float()
            ).to(dirty_a.dtype)

    # SFA physical layout is (M32, M4, rm, K4, rk, L).
    sf_k = a_torch.shape[1] // 32
    assert dirty_sfa.shape == (
        32,
        4,
        math.ceil(m / 128),
        4,
        math.ceil(sf_k / 4),
        l,
    )
    for batch_idx in range(l):
        valid_m = int(masked_m[batch_idx].item())
        for row in range(valid_m, m):
            mt = row // 128
            rem = row % 128
            m4 = rem // 32
            m32 = rem % 32
            dirty_sfa[m32, m4, mt, :, :, batch_idx] = -1

    return dirty_a, dirty_sfa


def _copy_cute_output_to_logical_rows(
    c_tensor: cute.Tensor,
    c_ref: torch.Tensor,
) -> torch.Tensor:
    cute.testing.convert(
        c_tensor,
        from_dlpack(c_ref, assumed_align=16).mark_layout_dynamic(leading_dim=1),
    )
    return c_ref


@pytest.mark.skipif(
    not is_cute_dsl_available(), reason="Please `pip install nvidia-cutlass-dsl`"
)
def test_mxfp8_masked_gemm_padded_rows_do_not_change_valid_rows():
    torch.manual_seed(1234)
    device = torch.device("cuda:0")
    _requires_sm100_or_sm103(device)

    l, m, k, n = 2, 256, 128, 128
    ab_dtype = "float8_e4m3fn"
    sf_dtype = "float8_e8m0fnu"
    c_dtype = "bfloat16"
    sf_vec_size = 32
    masked_m = torch.tensor([96, 33], dtype=torch.int32, device=device)

    a_ref = cutlass_torch.matrix(l, m, k, False, cutlass.Float32, device=device)
    b_ref = cutlass_torch.matrix(l, n, k, False, cutlass.Float32, device=device)
    _, a_torch = cutlass_torch.cute_tensor_like(
        a_ref,
        get_cutlass_dtype(ab_dtype),
        is_dynamic_layout=True,
        assumed_align=16,
    )
    _, b_torch = cutlass_torch.cute_tensor_like(
        b_ref,
        get_cutlass_dtype(ab_dtype),
        is_dynamic_layout=True,
        assumed_align=16,
    )
    _, _, clean_sfa = create_scale_factor_tensor(
        l, m, k, sf_vec_size, get_cutlass_dtype(sf_dtype), device
    )
    _, _, sfb_torch = create_scale_factor_tensor(
        l, n, k, sf_vec_size, get_cutlass_dtype(sf_dtype), device
    )

    dirty_a, dirty_sfa = _dirty_mxfp8_padded_rows(a_torch, clean_sfa, masked_m)
    clean_ref = cutlass_torch.matrix(l, m, n, False, cutlass.Float32, device=device)
    dirty_ref = cutlass_torch.matrix(l, m, n, False, cutlass.Float32, device=device)
    clean_c_tensor, clean_out = cutlass_torch.cute_tensor_like(
        clean_ref,
        get_cutlass_dtype(c_dtype),
        is_dynamic_layout=True,
        assumed_align=16,
    )
    dirty_c_tensor, dirty_out = cutlass_torch.cute_tensor_like(
        dirty_ref,
        get_cutlass_dtype(c_dtype),
        is_dynamic_layout=True,
        assumed_align=16,
    )
    clean_out.fill_(float("nan"))
    dirty_out.fill_(float("nan"))

    grouped_gemm_nt_masked(
        (a_torch, clean_sfa),
        (b_torch, sfb_torch),
        clean_out,
        masked_m,
        ab_dtype=ab_dtype,
        sf_dtype=sf_dtype,
        c_dtype=c_dtype,
        sf_vec_size=sf_vec_size,
    )
    grouped_gemm_nt_masked(
        (dirty_a, dirty_sfa),
        (b_torch, sfb_torch),
        dirty_out,
        masked_m,
        ab_dtype=ab_dtype,
        sf_dtype=sf_dtype,
        c_dtype=c_dtype,
        sf_vec_size=sf_vec_size,
    )

    clean_logical = _copy_cute_output_to_logical_rows(clean_c_tensor, clean_ref)
    dirty_logical = _copy_cute_output_to_logical_rows(dirty_c_tensor, dirty_ref)

    valid_row_failures = []
    for batch_idx in range(l):
        valid_m = int(masked_m[batch_idx].item())
        try:
            torch.testing.assert_close(
                dirty_logical[:valid_m, :, batch_idx],
                clean_logical[:valid_m, :, batch_idx],
                atol=0,
                rtol=0,
            )
        except AssertionError as err:
            valid_row_failures.append(f"batch {batch_idx}: {err}")

    assert not valid_row_failures, (
        "padded MXFP8 rows changed valid output rows\n"
        + "\n".join(valid_row_failures)
    )
    print(
        "masked output rows finite: "
        f"{_masked_rows_are_finite(dirty_logical, masked_m)}"
    )


@pytest.mark.skipif(
    not is_cute_dsl_available(), reason="Please `pip install nvidia-cutlass-dsl`"
)
@pytest.mark.parametrize("lm", [(1, 1024), (2, 512), (4, 256)])
@pytest.mark.parametrize("kn", [(7168, 4096), (2048, 7168)])
@pytest.mark.parametrize(
    "ab_dtype,sf_dtype,c_dtype,sf_vec_size",
    [
        ("float4_e2m1fn", "float8_e8m0fnu", "float16", 16),
        ("float4_e2m1fn", "float8_e8m0fnu", "bfloat16", 16),
        ("float4_e2m1fn", "float8_e8m0fnu", "float32", 16),
        ("float4_e2m1fn", "float8_e4m3fn", "float16", 16),
        ("float4_e2m1fn", "float8_e4m3fn", "bfloat16", 16),
        ("float4_e2m1fn", "float8_e4m3fn", "float32", 16),
        ("float8_e4m3fn", "float8_e8m0fnu", "bfloat16", 32),
        ("float8_e4m3fn", "float8_e8m0fnu", "float16", 32),
        ("float8_e4m3fn", "float8_e8m0fnu", "float32", 32),
        ("float8_e4m3fn", "float8_e8m0fnu", "float8_e4m3fn", 32),
        ("float8_e4m3fn", "float8_e8m0fnu", "float8_e5m2", 32),
        ("float8_e5m2", "float8_e8m0fnu", "bfloat16", 32),
        ("float8_e5m2", "float8_e8m0fnu", "float16", 32),
        ("float8_e5m2", "float8_e8m0fnu", "float32", 32),
        ("float8_e5m2", "float8_e8m0fnu", "float8_e4m3fn", 32),
        ("float8_e5m2", "float8_e8m0fnu", "float8_e5m2", 32),
    ],
)
@pytest.mark.parametrize("a_major", ["k"])
@pytest.mark.parametrize("b_major", ["k"])
@pytest.mark.parametrize("c_major", ["n"])
@pytest.mark.parametrize("fuse_alpha", [False, True])
@pytest.mark.parametrize("alpha_dtype", ["float32"])
@pytest.mark.parametrize("mma_tiler_mn", [(128, 128)])
@pytest.mark.parametrize("cluster_shape_mn", [(1, 1)])
@pytest.mark.parametrize("tolerance", [1e-01])
@pytest.mark.parametrize("iterations", [3])
@pytest.mark.parametrize("enable_dst_signals", [False, True])
def test_blockscaled_gemm_python_interface(
    lm: Tuple[int, int],
    kn: Tuple[int, int],
    ab_dtype: cutlass.dtype,
    sf_dtype: cutlass.dtype,
    sf_vec_size: int,
    c_dtype: cutlass.dtype,
    a_major: str,
    b_major: str,
    c_major: str,
    fuse_alpha: bool,
    alpha_dtype: cutlass.dtype,
    mma_tiler_mn: Tuple[int, int],
    cluster_shape_mn: Tuple[int, int],
    tolerance: float,
    iterations: int,
    enable_dst_signals: int,
):
    torch.manual_seed(42)
    device = torch.device("cuda:0")
    device_ver = torch.cuda.get_device_capability(device)
    supported_device_vers = [(10, 0), (10, 3)]
    if device_ver not in supported_device_vers:
        pytest.skip(
            f"Cute-dsl backend is only supported on {supported_device_vers}, skipping {device_ver}."
        )

    l, m = lm
    k, n = kn
    if l == 1:
        pytest.xfail("nvidia-cutlass-dsl has issue when l=1")

    sm_count = get_num_sm(device) if enable_dst_signals else None

    print(f"device: {device}")

    if not Sm100BlockScaledPersistentDenseGemmKernel.can_implement(
        get_cutlass_dtype(ab_dtype),
        get_cutlass_dtype(sf_dtype),
        sf_vec_size,
        get_cutlass_dtype(c_dtype),
        mma_tiler_mn,
        cluster_shape_mn,
        m,
        n,
        k,
        l,
        a_major,
        b_major,
        c_major,
    ):
        pytest.skip(
            f"Unsupported testcase {ab_dtype}, {sf_dtype}, {sf_vec_size}, {c_dtype},  {mma_tiler_mn}, {cluster_shape_mn}, {m}, {n}, {k}, {l}, {a_major}, {b_major}, {c_major}"
        )

    if not (a_major == "k" and b_major == "k" and c_major == "n"):
        # not supported since we try to align deepgemm for now
        pytest.skip(
            f"Skip non deepgemm-like cases {a_major}, {b_major}, {c_major}. Might be added later"
        )

    a_ref = cutlass_torch.matrix(
        l, m, k, a_major == "m", cutlass.Float32, device=device
    )
    b_ref = cutlass_torch.matrix(
        l, n, k, b_major == "n", cutlass.Float32, device=device
    )
    c_ref = cutlass_torch.matrix(
        l, m, n, c_major == "m", cutlass.Float32, device=device
    )

    a_tensor, a_torch = cutlass_torch.cute_tensor_like(
        a_ref,
        get_cutlass_dtype(ab_dtype),
        is_dynamic_layout=True,
        assumed_align=16,
    )
    b_tensor, b_torch = cutlass_torch.cute_tensor_like(
        b_ref,
        get_cutlass_dtype(ab_dtype),
        is_dynamic_layout=True,
        assumed_align=16,
    )
    c_tensor, c_torch = cutlass_torch.cute_tensor_like(
        c_ref,
        get_cutlass_dtype(c_dtype),
        is_dynamic_layout=True,
        assumed_align=16,
    )
    alpha_tensor = (
        torch.randn(l, dtype=torch.float32, device=device) if fuse_alpha else None
    )

    # for deepgemm-like python interface
    if ab_dtype == "float4_e2m1fn":
        m, k, l = a_torch.shape
        n, k, l = b_torch.shape
        # slice into half after flatten
        half_len_a = a_torch.numel() // 2
        half_len_b = b_torch.numel() // 2
        a_torch = (
            a_torch.permute(2, 0, 1)
            .flatten()[:half_len_a]
            .reshape(l, m, k // 2)
            .permute(1, 2, 0)
        )
        b_torch = (
            b_torch.permute(2, 0, 1)
            .flatten()[:half_len_b]
            .reshape(l, n, k // 2)
            .permute(1, 2, 0)
        )

    sfa_ref, sfa_tensor, sfa_torch = create_scale_factor_tensor(
        l, m, k, sf_vec_size, get_cutlass_dtype(sf_dtype), device
    )
    sfb_ref, sfb_tensor, sfb_torch = create_scale_factor_tensor(
        l, n, k, sf_vec_size, get_cutlass_dtype(sf_dtype), device
    )
    masked_m_tensor = torch.randint(0, m, (l,), dtype=torch.int32, device=device)

    for _ in range(iterations):
        dst_signals = (
            torch.zeros((l,), dtype=torch.uint32, device="cuda")
            if enable_dst_signals
            else None
        )

        # deepgemm-like python interface: fp4 packed, for DLFW integration
        grouped_gemm_nt_masked(
            (a_torch, sfa_torch),
            (b_torch, sfb_torch),
            c_torch,
            masked_m_tensor,
            ab_dtype=ab_dtype,
            sf_dtype=sf_dtype,
            c_dtype=c_dtype,
            sf_vec_size=sf_vec_size,
            mma_tiler_mn=mma_tiler_mn,
            cluster_shape_mn=cluster_shape_mn,
            alpha=alpha_tensor,
            alpha_dtype=alpha_dtype,
            sm_count=sm_count,
            dst_signals=dst_signals,
        )

        if enable_dst_signals:
            assert torch.all(dst_signals == sm_count), f"{dst_signals}"

    # compute ref output
    if not fuse_alpha:
        alpha_tensor = torch.ones(l, dtype=torch.float32, device=device)
    res_a = torch.einsum("mkl,mkl->mkl", a_ref, sfa_ref)
    res_b = torch.einsum("nkl,nkl->nkl", b_ref, sfb_ref)
    ref = torch.einsum("mkl,nkl->mnl", res_a, res_b)
    ref = torch.einsum("mnl,l->mnl", ref, alpha_tensor)

    # Convert c back to f32 for comparison.
    cute.testing.convert(
        c_tensor,
        from_dlpack(c_ref, assumed_align=16).mark_layout_dynamic(
            leading_dim=(1 if c_major == "n" else 0)
        ),
    )

    if c_dtype in ("float32", "float16", "bfloat16"):
        for i in range(l):
            # skip testing c_ref & ref
            torch.testing.assert_close(
                c_ref[: masked_m_tensor[i].item(), :, i],
                ref[: masked_m_tensor[i].item(), :, i],
                atol=tolerance,
                rtol=1e-02,
            )
    elif c_dtype in ("float8_e5m2", "float8_e4m3fn"):
        # Convert ref : f32 -> f8 -> f32
        ref_f8_ = torch.empty(*(l, m, n), dtype=torch.uint8, device=device).permute(
            1, 2, 0
        )
        ref_f8 = from_dlpack(ref_f8_, assumed_align=16).mark_layout_dynamic(
            leading_dim=1
        )
        ref_f8.element_type = get_cutlass_dtype(c_dtype)
        ref = ref.permute(2, 0, 1).contiguous().permute(1, 2, 0)
        ref_tensor = from_dlpack(ref, assumed_align=16).mark_layout_dynamic(
            leading_dim=1
        )
        cute.testing.convert(ref_tensor, ref_f8)
        cute.testing.convert(ref_f8, ref_tensor)
        for i in range(l):
            # skip testing c_ref & ref
            torch.testing.assert_close(
                c_ref[: masked_m_tensor[i].item(), :, i],
                ref[: masked_m_tensor[i].item(), :, i],
                atol=tolerance,
                rtol=1e-02,
            )


if __name__ == "__main__":
    test_blockscaled_gemm_python_interface(
        lm=(1, 1024),
        kn=(7168, 4096),
        ab_dtype="float4_e2m1fn",
        sf_dtype="float8_e8m0fnu",
        sf_vec_size=16,
        c_dtype="float16",
        a_major="k",
        b_major="k",
        c_major="n",
        fuse_alpha=False,
        alpha_dtype="float32",
        mma_tiler_mn=(128, 128),
        cluster_shape_mn=(2, 1),
        tolerance=1e-01,
        iterations=3,
        sm_count=132,
        enable_dst_signals=True,
    )
