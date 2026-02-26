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
KDA (Key-Driven Attention) Kernels - CuTe DSL Implementations
==============================================================

Per-K-dimension gating variant of GDN. Gate g[B,T,HV,K] applied per-lane
instead of GDN's scalar broadcast.

Exported Kernels:
- recurrent_kda: Low-level KDA decode kernel dispatch (T=1)
- cutedsl_kda_decode: fla-compatible wrapper with state management
- RecurrentKDAKernel: Kernel class for advanced usage
"""

from typing import Optional, Type

try:
    from .kda_decode_bf16_state import (
        recurrent_kda,
        cutedsl_kda_decode,
        RecurrentKDAKernel,
    )

    _has_cute_dsl = True
except ImportError:
    _has_cute_dsl = False
    recurrent_kda = None  # type: ignore
    cutedsl_kda_decode = None  # type: ignore
    RecurrentKDAKernel: Optional[Type] = None  # type: ignore

__all__ = [
    "recurrent_kda",
    "cutedsl_kda_decode",
    "RecurrentKDAKernel",
]
