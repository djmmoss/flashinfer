# CuTeDSL MXFP8 Strict Mask Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine whether padded MXFP8 fixed-slot rows corrupt valid output rows in FlashInfer's CuTeDSL masked grouped GEMM, then implement the smallest FlashInfer fix: strict input masking if valid rows are corrupt, or output sanitation if only masked rows are unsafe for downstream consumers.

**Architecture:** Start with a root-cause regression test against `grouped_gemm_nt_masked`. Keep changes inside the existing CuTeDSL grouped GEMM machinery and Python wrapper unless the test proves the issue is only downstream output hygiene. Preserve default non-strict behavior and make any new behavior opt-in.

**Tech Stack:** Python, PyTorch, pytest, FlashInfer JIT, CuTe DSL, CUTLASS DSL, Blackwell SM100/SM103 GPUs.

---

## File Structure

- Modify `tests/gemm/test_cute_dsl_blockscaled_gemm.py`
  - Add a focused MXFP8 fixed-slot root-cause test.
  - Reuse existing `create_scale_factor_tensor`, `grouped_gemm_nt_masked`, and `is_cute_dsl_available` patterns.
- Modify `flashinfer/gemm/kernels/grouped_gemm_masked_blackwell.py`
  - Path A: add `strict_masked_m` keyword plumbing, validation, compile-cache keying, and kernel-mode threading.
  - Path B: add an opt-in output sanitation mode if root cause is masked-output contamination only.

---

### Task 1: Add Root-Cause Test

**Files:**
- Modify: `tests/gemm/test_cute_dsl_blockscaled_gemm.py`

- [ ] **Step 1: Add helpers and the root-cause test**

Add these imports near the top of `tests/gemm/test_cute_dsl_blockscaled_gemm.py`:

```python
import math
```

Add these helper functions after the imports:

```python
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

    # SFA physical layout is (M32, M4, rm, K4, rk, L) flattened as
    # (l, rm, rk, m32, m4, k4) by create_scale_factor_tensor.
    sfa_view = dirty_sfa.view(l, math.ceil(m / 128), -1, 32, 4, 4)
    for batch_idx in range(l):
        valid_m = int(masked_m[batch_idx].item())
        for row in range(valid_m, m):
            mt = row // 128
            rem = row % 128
            m4 = rem // 32
            m32 = rem % 32
            sfa_view[batch_idx, mt, :, m32, m4, :] = 0xFF

    return dirty_a, dirty_sfa
```

Add this test near `test_blockscaled_gemm_python_interface`:

```python
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
    _, _, clean_sfa = create_scale_factor_tensor(
        l, m, k, sf_vec_size, get_cutlass_dtype(sf_dtype), device
    )
    _, _, sfb_torch = create_scale_factor_tensor(
        l, n, k, sf_vec_size, get_cutlass_dtype(sf_dtype), device
    )

    dirty_a, dirty_sfa = _dirty_mxfp8_padded_rows(a_torch, clean_sfa, masked_m)
    clean_out = torch.empty((m, n, l), dtype=torch.bfloat16, device=device)
    dirty_out = torch.empty_like(clean_out)
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

    valid_rows_match = True
    for batch_idx in range(l):
        valid_m = int(masked_m[batch_idx].item())
        try:
            torch.testing.assert_close(
                dirty_out[:valid_m, :, batch_idx],
                clean_out[:valid_m, :, batch_idx],
                atol=1e-1,
                rtol=1e-2,
            )
        except AssertionError:
            valid_rows_match = False

    assert valid_rows_match, "padded MXFP8 rows changed valid output rows"
    assert _masked_rows_are_finite(dirty_out, masked_m), (
        "valid rows are stable, but masked rows are not finite; implement "
        "output sanitation instead of strict input masking"
    )
```

- [ ] **Step 2: Run the root-cause test**

Run:

```bash
PATH=/home/dmoss/scratch/repos/kernels/flashinfer/.venv/bin:$PATH \
FLASHINFER_WORKSPACE_BASE=$PWD \
python -m pytest -q tests/gemm/test_cute_dsl_blockscaled_gemm.py::test_mxfp8_masked_gemm_padded_rows_do_not_change_valid_rows -s
```

Expected outcomes:

- If the test fails with `padded MXFP8 rows changed valid output rows`, execute Path A.
- If the test fails with `masked rows are not finite`, execute Path B.
- If the test passes, commit only the regression test and stop; the current kernel already satisfies both valid-row and output-finite requirements for this case.

- [ ] **Step 3: Commit the root-cause test**

Run:

```bash
git add tests/gemm/test_cute_dsl_blockscaled_gemm.py
git commit -m "test: classify MXFP8 masked grouped GEMM padding behavior"
```

Expected: one commit containing only the root-cause test.

---

### Task 2A: Implement Strict Input Masking If Valid Rows Are Corrupt

Execute this task only if Task 1 fails because valid rows change.

**Files:**
- Modify: `flashinfer/gemm/kernels/grouped_gemm_masked_blackwell.py`
- Modify: `tests/gemm/test_cute_dsl_blockscaled_gemm.py`

- [ ] **Step 1: Add strict flag plumbing and validation**

In `grouped_gemm_nt_masked`, pop `strict_masked_m` before the unsupported-kwargs assertion:

```python
    strict_masked_m = kwargs.pop("strict_masked_m", False)
```

After `alpha_dtype = kwargs.pop("alpha_dtype", None)`, add:

```python
    if strict_masked_m:
        if not (
            ab_dtype == "float8_e4m3fn"
            and sf_dtype == "float8_e8m0fnu"
            and sf_vec_size == 32
        ):
            raise ValueError(
                "strict_masked_m=True is currently supported only for "
                'ab_dtype="float8_e4m3fn", sf_dtype="float8_e8m0fnu", '
                "sf_vec_size=32"
            )
        if is_swap_ab:
            raise ValueError(
                "strict_masked_m=True with is_swap_ab=True is not enabled until "
                "the masked physical operand is tested"
            )
```

Thread `strict_masked_m=strict_masked_m` into the `get_cute_dsl_compiled_masked_gemm_kernel(...)` call.

- [ ] **Step 2: Add strict flag to the compile key and kernel constructor**

Add `strict_masked_m: bool = False` to `get_cute_dsl_compiled_masked_gemm_kernel(...)`.

Add `strict_masked_m: bool = False` to `MaskedBatchedMatmulCuteDSL.__init__`, store it as `self._strict_masked_m`, and pass it into `Sm100BlockScaledPersistentDenseGemmKernel(...)`.

Add `strict_masked_m: bool = False` to `Sm100BlockScaledPersistentDenseGemmKernel.__init__`, store it as `self.strict_masked_m`, and keep the default false.

- [ ] **Step 3: Make strict mode zero/predicate masked A fragments before MMA**

In `Sm100BlockScaledPersistentDenseGemmKernel.mainloop`, after `tCrA = tiled_mma.make_fragment_A(sA)` and before `cute.gemm(...)`, derive each fragment row's logical M coordinate for the current tile. For strict mode, zero `tCrA` elements whose logical row is `>= masked_m[current_l]`.

Use this shape of implementation:

```python
if cutlass.const_expr(self.strict_masked_m):
    valid_m = tile_sched_params.masked_m[mma_tile_coord_mnl[2]]
    tile_m_base = mma_tile_coord_mnl[0] * self.mma_tiler[0]
    for i in cutlass.range(cute.size(tCrA), unroll_full=True):
        coord = tCrA.layout.get_hier_coord(i)
        row_in_tile = coord[1]
        logical_m = tile_m_base + row_in_tile
        if logical_m >= valid_m:
            tCrA[i] = self.a_dtype.zero
```

If CuTe DSL rejects direct assignment or the coordinate mode is wrong, stop after capturing the compiler error and inspect `tCrA.layout` and the nearest existing fragment-copy point. The next edit must preserve this invariant: masked A payload entering MMA is zero. Do not add a separate cleanup kernel.

- [ ] **Step 4: Keep SFA finite for masked fragments**

If the Task 1 reproducer still fails because `0 * NaN` from `0xFF` SFA poisons the operation, apply the same strict predicate to SFA at the shared-to-tensor-memory copy point:

```python
if cutlass.const_expr(self.strict_masked_m):
    valid_m = tile_sched_params.masked_m[mma_tile_coord_mnl[2]]
    tile_m_base = mma_tile_coord_mnl[0] * self.mma_tiler[0]
    for i in cutlass.range(cute.size(tCtSFA_compact_s2t), unroll_full=True):
        coord = tCtSFA_compact_s2t.layout.get_hier_coord(i)
        row_in_tile = coord[1]
        logical_m = tile_m_base + row_in_tile
        if logical_m >= valid_m:
            tCtSFA_compact_s2t[i] = cutlass.Float8E8M0FNU(0)
```

If `cutlass.Float8E8M0FNU(0)` does not compile, stop after capturing the compiler error and replace it with the numeric construction pattern already accepted for `self.sf_dtype` in this file. The semantic requirement is fixed: masked SFA must be finite, while zero payload provides the neutral contribution.

- [ ] **Step 5: Enable strict mode in the root-cause test**

Update the dirty invocation in the Task 1 test to pass:

```python
        strict_masked_m=True,
```

Add a validation test:

```python
def test_mxfp8_strict_mask_rejects_unsupported_tuple():
    torch.manual_seed(123)
    device = torch.device("cuda:0")
    _requires_sm100_or_sm103(device)
    out = torch.empty((128, 128, 2), dtype=torch.bfloat16, device=device)
    masked_m = torch.tensor([64, 64], dtype=torch.int32, device=device)
    a = torch.empty((128, 128, 2), dtype=torch.float8_e4m3fn, device=device)
    b = torch.empty((128, 128, 2), dtype=torch.float8_e4m3fn, device=device)
    sf = torch.empty((2, 1, 1, 32, 4, 4), dtype=torch.uint8, device=device).view(
        torch.float8_e8m0fnu
    )
    with pytest.raises(ValueError, match="strict_masked_m=True"):
        grouped_gemm_nt_masked(
            (a, sf),
            (b, sf),
            out,
            masked_m,
            ab_dtype="float8_e4m3fn",
            sf_dtype="float8_e8m0fnu",
            c_dtype="bfloat16",
            sf_vec_size=16,
            strict_masked_m=True,
        )
```

- [ ] **Step 6: Run strict-mode tests**

Run:

```bash
rm -rf ~/.cache/flashinfer
PATH=/home/dmoss/scratch/repos/kernels/flashinfer/.venv/bin:$PATH \
FLASHINFER_WORKSPACE_BASE=$PWD \
python -m pytest -q \
  tests/gemm/test_cute_dsl_blockscaled_gemm.py::test_mxfp8_masked_gemm_padded_rows_do_not_change_valid_rows \
  tests/gemm/test_cute_dsl_blockscaled_gemm.py::test_mxfp8_strict_mask_rejects_unsupported_tuple -s
```

Expected: both tests pass on SM100/SM103.

- [ ] **Step 7: Commit strict input masking**

Run:

```bash
git add flashinfer/gemm/kernels/grouped_gemm_masked_blackwell.py tests/gemm/test_cute_dsl_blockscaled_gemm.py
git commit -m "fix: add strict MXFP8 masking to CuTeDSL grouped GEMM"
```

Expected: one implementation commit.

---

### Task 2B: Implement Output Sanitation If Only Masked Rows Are Unsafe

Execute this task only if Task 1 shows valid rows are stable but masked output rows are non-finite or stale.

**Files:**
- Modify: `flashinfer/gemm/kernels/grouped_gemm_masked_blackwell.py`
- Modify: `tests/gemm/test_cute_dsl_blockscaled_gemm.py`

- [ ] **Step 1: Add an opt-in output sanitation keyword**

In `grouped_gemm_nt_masked`, pop this kwarg before the unsupported-kwargs assertion:

```python
    zero_masked_output = kwargs.pop("zero_masked_output", False)
```

Validate:

```python
    if zero_masked_output and is_combine_fusion:
        raise ValueError("zero_masked_output=True is not supported with combine fusion")
```

Thread `zero_masked_output=zero_masked_output` into `get_cute_dsl_compiled_masked_gemm_kernel(...)`.

- [ ] **Step 2: Include sanitation in the compile key and kernel constructor**

Add `zero_masked_output: bool = False` to `get_cute_dsl_compiled_masked_gemm_kernel(...)`.

Add `zero_masked_output: bool = False` to `MaskedBatchedMatmulCuteDSL.__init__`, store it as `self._zero_masked_output`, and pass it into `Sm100BlockScaledPersistentDenseGemmKernel(...)`.

Add `zero_masked_output: bool = False` to `Sm100BlockScaledPersistentDenseGemmKernel.__init__`, store it as `self.zero_masked_output`, and keep the default false.

- [ ] **Step 3: Zero padded output rows inside the epilogue path**

Find the existing TMA store or combine-fusion store path for `c_tensor` in `Sm100BlockScaledPersistentDenseGemmKernel.mainloop`. Before storing a C element, compute:

```python
valid_m = tile_sched_params.masked_m[mma_tile_coord_mnl[2]]
tile_m_base = mma_tile_coord_mnl[0] * self.mma_tiler[0]
```

For `zero_masked_output=True`, store zero for rows where `tile_m_base + row_in_tile >= valid_m`.

Use the same row-coordinate convention as the existing epilogue fragment layout. If the epilogue stores only scheduled tiles and never reaches padded rows, add a tiny post-GEMM in-kernel loop over the current tile's padded rows before tile advance. Do not launch a separate Triton/CUDA cleanup kernel.

- [ ] **Step 4: Update the root-cause test to request output sanitation**

In the dirty invocation from Task 1, add:

```python
        zero_masked_output=True,
```

Keep both assertions:

```python
    assert valid_rows_match, "padded MXFP8 rows changed valid output rows"
    assert _masked_rows_are_finite(dirty_out, masked_m)
```

- [ ] **Step 5: Add validation test for combine fusion rejection**

Add:

```python
def test_zero_masked_output_rejects_combine_fusion():
    torch.manual_seed(123)
    device = torch.device("cuda:0")
    _requires_sm100_or_sm103(device)
    out = torch.empty((128, 128, 2), dtype=torch.bfloat16, device=device)
    masked_m = torch.tensor([64, 64], dtype=torch.int32, device=device)
    a = torch.empty((128, 128, 2), dtype=torch.float8_e4m3fn, device=device)
    b = torch.empty((128, 128, 2), dtype=torch.float8_e4m3fn, device=device)
    sf = torch.empty((2, 1, 1, 32, 4, 4), dtype=torch.uint8, device=device).view(
        torch.float8_e8m0fnu
    )
    with pytest.raises(ValueError, match="zero_masked_output=True"):
        grouped_gemm_nt_masked(
            (a, sf),
            (b, sf),
            out,
            masked_m,
            ab_dtype="float8_e4m3fn",
            sf_dtype="float8_e8m0fnu",
            c_dtype="bfloat16",
            sf_vec_size=32,
            is_combine_fusion=True,
            zero_masked_output=True,
        )
```

- [ ] **Step 6: Run output sanitation tests**

Run:

```bash
rm -rf ~/.cache/flashinfer
PATH=/home/dmoss/scratch/repos/kernels/flashinfer/.venv/bin:$PATH \
FLASHINFER_WORKSPACE_BASE=$PWD \
python -m pytest -q \
  tests/gemm/test_cute_dsl_blockscaled_gemm.py::test_mxfp8_masked_gemm_padded_rows_do_not_change_valid_rows \
  tests/gemm/test_cute_dsl_blockscaled_gemm.py::test_zero_masked_output_rejects_combine_fusion -s
```

Expected: both tests pass on SM100/SM103.

- [ ] **Step 7: Commit output sanitation**

Run:

```bash
git add flashinfer/gemm/kernels/grouped_gemm_masked_blackwell.py tests/gemm/test_cute_dsl_blockscaled_gemm.py
git commit -m "fix: add masked output sanitation to CuTeDSL grouped GEMM"
```

Expected: one implementation commit.

---

### Task 3: Validation And Benchmark Evidence

**Files:**
- No source changes expected unless tests reveal a bug.

- [ ] **Step 1: Run focused regression suite**

Run:

```bash
rm -rf ~/.cache/flashinfer
PATH=/home/dmoss/scratch/repos/kernels/flashinfer/.venv/bin:$PATH \
FLASHINFER_WORKSPACE_BASE=$PWD \
python -m pytest -q tests/gemm/test_cute_dsl_blockscaled_gemm.py -s
```

Expected: tests pass or unsupported-device tests skip cleanly.

- [ ] **Step 2: Run representative existing MXFP8 grouped tests**

Run:

```bash
PATH=/home/dmoss/scratch/repos/kernels/flashinfer/.venv/bin:$PATH \
FLASHINFER_WORKSPACE_BASE=$PWD \
python -m pytest -q tests/grouped_mm/test_grouped_mm_mxfp8.py -s
```

Expected: tests pass or skip based on cuDNN/MOE availability.

- [ ] **Step 3: Capture performance comparison**

If Task 2A was implemented, benchmark:

```bash
PATH=/home/dmoss/scratch/repos/kernels/flashinfer/.venv/bin:$PATH \
FLASHINFER_WORKSPACE_BASE=$PWD \
python -m pytest -q tests/gemm/test_cute_dsl_blockscaled_gemm.py::test_mxfp8_masked_gemm_padded_rows_do_not_change_valid_rows -s
```

Record whether strict input masking removes the vLLM cleanup need and whether total elapsed time is neutral or improved for the test shape. If Task 2B was implemented, record the same for `zero_masked_output=True`.

- [ ] **Step 4: Commit validation notes if docs changed**

If you add a short validation note to the design or plan, run:

```bash
git add docs/superpowers/specs/2026-06-04-cutedsl-mxfp8-strict-mask-design.md docs/superpowers/plans/2026-06-04-cutedsl-mxfp8-strict-mask.md
git commit -m "docs: record MXFP8 mask validation"
```

Expected: commit only if documentation changed.

---

## Self-Review Checklist

- The plan implements the spec's root-cause gate before any kernel change.
- Path A implements strict input masking only after a valid-row corruption reproducer.
- Path B implements output sanitation when valid rows are stable but padded rows are unsafe.
- The `strict_masked_m` and `zero_masked_output` flags are opt-in and included in compile-cache keys.
- Unsupported swapped-A/B or combine-fusion combinations fail explicitly until tested.
- Tests compare valid rows separately from masked rows.
- No vLLM-specific cleanup kernels are moved into FlashInfer as the main solution.
