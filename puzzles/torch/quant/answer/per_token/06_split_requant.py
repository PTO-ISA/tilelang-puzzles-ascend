"""per_token 06 (torch). See doc/quant/per_token/06_split_requant.md"""

import torch

from harness import oracle
from harness.consts import CANONICAL_G, E4M3_CLAMP_MIN, E4M3_MAX


def torch_sf_only(x: torch.Tensor, group_size: int = CANONICAL_G):
    """Compute only the scale factors."""
    # --- BEGIN SOLUTION hint="amax over each group, clamp, divide by E4M3_MAX; return just sf"
    m, k = x.shape
    grouped = x.float().view(m, k // group_size, group_size)
    amax = grouped.abs().amax(dim=-1).clamp(min=E4M3_CLAMP_MIN)
    return amax / E4M3_MAX
    # --- END SOLUTION


def torch_cast_only(x: torch.Tensor, sf: torch.Tensor, group_size: int = CANONICAL_G):
    """Quantize using scales that are given, with no amax pass.

    Mirror the kernel: take the reciprocal of the stored scale and multiply.
    """
    # --- BEGIN SOLUTION hint="sf_inv = 1.0 / sf (not E4M3_MAX/amax -- we only have sf); multiply each group by sf_inv.unsqueeze(-1) and cast to float8_e4m3fn"
    m, k = x.shape
    grouped = x.float().view(m, k // group_size, group_size)
    sf_inv = 1.0 / sf
    return (grouped * sf_inv.unsqueeze(-1)).view(m, k).to(torch.float8_e4m3fn)
    # --- END SOLUTION


def torch_requant(q_in: torch.Tensor, sf_in: torch.Tensor,
                  group_size: int = CANONICAL_G):
    """Dequantize an already-quantized input, then quantize it again."""
    # --- BEGIN SOLUTION hint="dequantize with oracle.cast_back(q_in, sf_in, (1, group_size), out_dtype=float32), then run the ordinary variant-01 quantize on the result"
    x = oracle.cast_back(q_in, sf_in, (1, group_size), out_dtype=torch.float32)
    m, k = x.shape
    grouped = x.view(m, k // group_size, group_size)
    amax = grouped.abs().amax(dim=-1).clamp(min=E4M3_CLAMP_MIN)
    sf = amax / E4M3_MAX
    q = (grouped * (E4M3_MAX / amax).unsqueeze(-1)).view(m, k).to(torch.float8_e4m3fn)
    return q, sf
    # --- END SOLUTION
