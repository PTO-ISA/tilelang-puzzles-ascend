"""cast_back 02 (torch). See doc/quant/cast_back/02_fp32_out.md"""

import torch

from common.consts import CANONICAL_G


def torch_cast_back_f32(q: torch.Tensor, sf: torch.Tensor, group_size: int = CANONICAL_G):
    """Dequantize to float32 instead of bfloat16."""
    # --- BEGIN SOLUTION hint="same as variant 01, but return float32 (no .to(bfloat16))"
    m, k = q.shape
    assert k % group_size == 0
    grouped = q.float().view(m, k // group_size, group_size)
    return (grouped * sf.unsqueeze(-1)).view(m, k)
    # --- END SOLUTION
