"""cast_back 06 (torch). See doc/quant/cast_back/06_fp4_e2m1.md"""

import torch

from harness.consts import CANONICAL_G
from harness.math_ops import unpack_e2m1_bytes


def torch_cast_back_fp4(q_packed: torch.Tensor, sf: torch.Tensor,
                        group_size: int = CANONICAL_G):
    """Dequantize packed-FP4 input.

    ``q_packed`` is (M, K/2) int8; the logical width is K = 2 * q_packed.shape[1].
    """
    # --- BEGIN SOLUTION hint="unpack_e2m1_bytes(q_packed) gives (M, K) float32; then scale per group exactly as variant 01"
    values = unpack_e2m1_bytes(q_packed)
    m, k = values.shape
    assert k % group_size == 0
    grouped = values.view(m, k // group_size, group_size)
    return (grouped * sf.unsqueeze(-1)).view(m, k).to(torch.bfloat16)
    # --- END SOLUTION
