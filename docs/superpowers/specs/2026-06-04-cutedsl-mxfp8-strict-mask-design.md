# CuTeDSL MXFP8 Strict Mask Design

## Objective

Extend FlashInfer's existing CuTeDSL masked grouped GEMM so MXFP8 fixed-slot callers can rely on `masked_m` semantics without pre-zeroing padded activation or scale rows.

The immediate consumer is vLLM's DeepEP low-latency native MXFP8 path, where dispatch receives per-expert fixed slots shaped like `(E, M_max, K)` plus `expert_num_tokens`. Rows past `expert_num_tokens[e]` may contain stale data. FlashInfer should either prove and fix any valid-output corruption caused by those rows, or expose a clearer downstream-safe output hygiene contract if stale data only affects masked output rows.

## Non-Goals

- Do not move vLLM's Triton cleanup kernels into FlashInfer as the primary solution.
- Do not create a separate GEMM implementation stack unless the existing CuTeDSL masked grouped GEMM cannot support strict MXFP8 masking without unacceptable regressions.
- Do not change the existing default behavior for non-MXFP8 callers or non-strict masked GEMM callers.
- Do not require vLLM to encode FlashInfer's internal scale swizzle layout.

## Root-Cause Gate

Before implementing strict input masking, add a FlashInfer regression test that fails on the current kernel and demonstrates valid-output corruption from padded MXFP8 activation or scale rows.

The test must distinguish two cases:

- Valid-row corruption: rows `m < masked_m[e]` differ when padded rows contain stale FP8 payloads, random scale bytes, or `0xFF` scale bytes. This justifies strict input/load masking inside the CuTeDSL GEMM.
- Masked-row contamination only: rows `m >= masked_m[e]` contain NaNs or stale values, but valid rows match the reference. In this case, the primary fix is output-row hygiene for downstream consumers, not input neutralization.

If no valid-row corruption reproducer exists, do not implement strict input masking as the main change. Revise the implementation plan to add a FlashInfer-owned fixed-slot wrapper or optional output sanitation mode that makes padded output rows safe for consumers such as vLLM's activation kernel.

## Kernel Contract

If the root-cause gate proves valid-row corruption, add a strict MXFP8 masked mode to the existing CuTeDSL masked grouped GEMM machinery. In this mode:

- For expert `e`, rows `m >= masked_m[e]` must not contribute to valid output rows.
- MXFP8 activation payload loads for masked rows must be predicated or neutralized inside the kernel.
- MXFP8 scale loads whose physical tile covers masked rows must be predicated or neutralized inside the kernel.
- Stale or sentinel scale bytes in masked regions, including `0xFF`, must not affect valid output rows.
- Output rows past `masked_m[e]` are unspecified unless a future API explicitly requests full-output sanitation.

This contract is stronger than simply masking stores. It covers all data movement and scale interpretation that can affect valid rows through tile-level MXFP8 processing.

For `is_swap_ab=False`, `masked_m` applies to the logical M dimension of A/C. For `is_swap_ab=True`, including combine-fusion mode, `masked_m` applies to the scheduler's masked dimension exactly as the existing `MaskedScheduler` interprets it. The implementation must document and test which physical operand and output dimension that corresponds to before enabling strict MXFP8 mode with swapped A/B.

## API Shape

Keep the low-level entrypoint close to the existing API and add an opt-in strict mask flag:

```python
flashinfer.gemm.grouped_gemm_nt_masked(
    lhs,
    rhs,
    out,
    masked_m,
    ab_dtype="float8_e4m3fn",
    sf_dtype="float8_e8m0fnu",
    c_dtype="bfloat16",
    sf_vec_size=32,
    strict_masked_m=True,
    ...
)
```

`strict_masked_m=True` is valid initially only for the MXFP8 tuple:

- `ab_dtype == "float8_e4m3fn"`
- `sf_dtype == "float8_e8m0fnu"`
- `c_dtype` supported by the existing CuTeDSL MXFP8 path
- `sf_vec_size == 32`

Unsupported combinations should fail explicitly rather than silently falling back to weak masking.

The Python wrapper must pop `strict_masked_m` before the unsupported-kwargs assertion, validate it against the MXFP8 tuple, and thread it into `get_cute_dsl_compiled_masked_gemm_kernel`. The strict flag must participate in the compiled-kernel cache key so strict and non-strict variants cannot collide.

Add a higher-level fixed-slot helper only after the kernel contract is in place:

```python
flashinfer.gemm.grouped_gemm_nt_masked_mxfp8_fixed_slots(
    lhs_values,          # (E, M_max, K)
    lhs_scales,          # row-major or cute-swizzled scales
    rhs_values,          # (E, N, K)
    rhs_scales,          # existing FlashInfer/CuTe layout
    out,                 # (E, M_max, N)
    expert_num_tokens,   # (E,) int32
    *,
    lhs_scale_layout="row_major",
    c_dtype="bfloat16",
    ...
)
```

This wrapper should own layout validation and scale conversion for integrations such as vLLM, but it should not depend on separate cleanup kernels for correctness.

## Implementation Direction

If the failing regression demonstrates valid-row corruption, extend `flashinfer/gemm/kernels/grouped_gemm_masked_blackwell.py` and its compile/cache key to include a strict MXFP8 masked mode. The mode should be keyed into `get_cute_dsl_compiled_masked_gemm_kernel` so default users keep the current generated kernel.

The CuTeDSL load path should derive the logical row covered by each activation and scale element. When that row is outside `masked_m[e]`, it should feed neutral values to the MMA path:

- activation payload: zero
- MXFP8 scale: finite scale handling or predication that prevents `0 * NaN` from stale `0xFF` scale bytes

`float8_e8m0fnu` has no true zero scale encoding; neutrality must come from zeroing or predicating the payload contribution. The scale handling exists to keep masked fragments finite and non-poisoning, not to encode zero through the scale byte.

The exact placement should be as close to the global-to-shared or shared-to-fragment load as the current CuTeDSL structure allows. The implementation should avoid a separate pre-pass kernel.

If the generic masked grouped GEMM becomes too awkward or measurably regresses non-strict users, split only the generated strict MXFP8 specialization while keeping it under the same FlashInfer CuTeDSL grouped GEMM machinery.

## vLLM Integration Target

After FlashInfer exposes strict masking, vLLM's `FlashInferCuteDSLBatchedExpertsMxfp8` path should stop carrying FlashInfer-specific cleanup and layout code:

- remove `_zero_invalid_scale_rows`
- remove `_zero_invalid_swizzled_scale_rows`
- avoid pre-zeroing dispatch activation rows for GEMM correctness
- replace vLLM-owned `swizzle_mxfp8_scales_batched_for_cute` with a FlashInfer-owned fixed-slot wrapper or layout utility

vLLM may still need to handle downstream consumers that read full padded workspaces, such as activation functions over `(E * M_max, N)`. That is a separate workspace hygiene issue, not an input correctness requirement for the GEMM.

If the root-cause gate shows only masked-output contamination, then vLLM integration should target a FlashInfer-owned fixed-slot wrapper or output sanitation option instead. That mode should guarantee padded output rows are finite and harmless for downstream activation without claiming valid-row input corruption.

## Tests

Add FlashInfer tests for strict MXFP8 masked grouped GEMM on SM100/SM103:

- a TDD root-cause test that fails on the current kernel before the implementation proceeds
- valid rows match a dequantized FP32 grouped-matmul reference, cast to the requested output dtype, when padded activation rows contain random stale FP8 data
- valid rows match the same reference when padded scale rows contain random bytes or `0xFF`
- empty experts and non-uniform `expert_num_tokens`
- row-major fixed-slot wrapper conversion, if the wrapper is added in the same change
- strict mode rejects unsupported dtype or scale-vector combinations
- swapped-A/B strict-mode behavior is either tested or explicitly rejected

Add a regression test mirroring the vLLM DeepEP fixed-slot shape: `(E, M_max, K)` activations, per-expert token counts, MXFP8 weights, and stale padded scale rows.

Reference tolerances should match existing MXFP8 grouped GEMM tests unless the new fixed-slot path requires a tighter local oracle. The root-cause test must compare valid rows separately from masked rows so it can identify whether the necessary fix is input strict masking or output sanitation.

## Performance Validation

Measure strict mode against the current vLLM workaround path:

- current vLLM cleanup plus existing FlashInfer kernel
- new FlashInfer strict mode without vLLM cleanup

Report total GEMM-call region time and kernel breakdown. The expected win is removal of pre-pass cleanup kernels and fewer extra memory writes. If strict predicates slow the GEMM enough to erase that win, keep strict mode opt-in and consider a generated MXFP8-only specialization.

Use geomean, worst-case, and outlier reporting over representative DeepEP-LL shapes. Treat strict mode as successful if it preserves valid-row accuracy, removes the relevant vLLM cleanup kernels, and improves or is neutral on total region time for the target shapes. If strict mode regresses total region time while output sanitation alone solves the real issue, prefer the narrower output-sanitizing wrapper.

## Open Risks

- The current CuTeDSL scale layout may make row-level masking non-local for scale tiles. If so, strict masking may need tile-level finite-scale handling plus payload predication rather than simple row predicates.
- Strict masking must not accidentally change combine-fusion or swapped-AB modes.
- vLLM's full-workspace activation reads may be the actual root cause. If so, strict input masking is not the right primary implementation.
