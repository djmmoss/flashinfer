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

"""Benchmark GDN Prefill SM100 CuTe DSL kernel."""

import argparse
import sys

import numpy as np
import torch
import torch.nn.functional as F

from flashinfer.utils import is_sm100a_supported
from flashinfer.testing.utils import bench_gpu_time


def make_inputs(B, T, H, D, device="cuda", dtype=torch.float16):
    torch.manual_seed(42)
    q = F.normalize(
        torch.randn(B, T, H, D, device=device, dtype=torch.float32), p=2, dim=-1
    ).to(dtype)
    k = F.normalize(
        torch.randn(B, T, H, D, device=device, dtype=torch.float32), p=2, dim=-1
    ).to(dtype)
    v = torch.randn(B, T, H, D, device=device, dtype=dtype)
    g = F.logsigmoid(torch.randn(B, T, H, device=device, dtype=dtype))
    beta = torch.rand(B, T, H, device=device, dtype=dtype).sigmoid()
    h0 = torch.zeros(B, H, D, D, device=device, dtype=torch.float32)
    return q, k, v, g, beta, h0


def bench_sm100(B, T, H, D, dtype=torch.float16):
    from flashinfer.gdn_kernels.gdn_prefill_sm100 import chunk_gated_delta_rule_sm100

    q, k, v, g, beta, h0 = make_inputs(B, T, H, D, dtype=dtype)
    o = torch.zeros_like(v)
    state_out = torch.zeros(B, H, D, D, device="cuda", dtype=torch.float32)

    def fn():
        chunk_gated_delta_rule_sm100(
            q, k, v, g, beta, o, None, h0.clone(), None, state_out
        )

    times = bench_gpu_time(
        fn, dry_run_time_ms=100, repeat_time_ms=1000, enable_cupti=True
    )
    return np.median(times)


def main():
    parser = argparse.ArgumentParser(description="Benchmark GDN Prefill SM100")
    parser.add_argument("--batch-sizes", type=str, default="1,2,4,8,16,32,64")
    parser.add_argument(
        "--seq-lens", type=str, default="128,256,512,1024,2048,4096,8192"
    )
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument(
        "--dtype", type=str, default="float16", choices=["float16", "bfloat16"]
    )
    args = parser.parse_args()

    if not torch.cuda.is_available() or not is_sm100a_supported(torch.device("cuda")):
        print("Requires SM100a (Blackwell)")
        sys.exit(1)

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    seq_lens = [int(x) for x in args.seq_lens.split(",")]
    H, D = args.num_heads, args.head_dim
    dtype = getattr(torch, args.dtype)

    print(f"GDN Prefill SM100 Benchmark: H={H}, D={D}, dtype={args.dtype}")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"{'Batch':>6} {'SeqLen':>8} {'Tokens':>10} {'Time(ms)':>10} {'TFLOPS':>10}")
    print("-" * 50)

    for T in seq_lens:
        for B in batch_sizes:
            try:
                t_ms = bench_sm100(B, T, H, D, dtype)
                tokens = B * T
                flops = 4 * tokens * H * D * D  # 2 matmuls of d^2 each
                tflops = flops / t_ms / 1e9
                print(f"{B:>6} {T:>8} {tokens:>10,} {t_ms:>10.3f} {tflops:>10.2f}")
            except Exception as e:
                print(f"{B:>6} {T:>8} {'ERROR':>10} {str(e)[:30]}")


if __name__ == "__main__":
    main()
