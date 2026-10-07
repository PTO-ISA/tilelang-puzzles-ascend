"""cast_back 07 (torch). See doc/quant/cast_back/07_col_major_compose.md"""

import torch

from common.consts import CANONICAL_G
from common.math_ops import decode_packed_ue8m0, unpack_e2m1_bytes


def torch_cast_back_compose(q_packed: torch.Tensor, sf_cm: torch.Tensor,
                            group_size: int = CANONICAL_G):
    """Dequantize FP4 values with column-major packed-UE8M0 scales.

    ``q_packed``: (M, K/2) int8, two e2m1 nibbles per byte.
    ``sf_cm``    : (K/group_size/2, M) int16 -- transposed *and* byte-packed.
    """
    # --- BEGIN SOLUTION hint="transpose sf_cm back to row-major with .T, decode_packed_ue8m0 it, unpack the FP4 values, then scale per group"
    scale = decode_packed_ue8m0(sf_cm.T.contiguous())
    values = unpack_e2m1_bytes(q_packed)
    m, k = values.shape
    assert k % group_size == 0
    grouped = values.view(m, k // group_size, group_size)
    return (grouped * scale.unsqueeze(-1)).view(m, k).to(torch.bfloat16)
    # --- END SOLUTION
