"""per_channel 01 (torch). See doc/quant/per_channel/01_raw_32tokens.md"""

import torch

from harness.consts import BLOCK_MN, E4M3_CLAMP_MIN, E4M3_MAX


def torch_per_channel_cast(x: torch.Tensor, group_tokens: int = BLOCK_MN):
    """Return ``(q, sf)`` with one scale per channel per group of tokens."""
    # --- BEGIN SOLUTION hint="view x as (M//group_tokens, group_tokens, K); amax over dim=1 (the token axis, not the last axis) and clamp; sf = amax/E4M3_MAX; broadcast with unsqueeze(1)"
    m, k = x.shape
    assert m % group_tokens == 0, f"M={m} must be a multiple of {group_tokens}"
    grouped = x.float().view(m // group_tokens, group_tokens, k)
    amax = grouped.abs().amax(dim=1).clamp(min=E4M3_CLAMP_MIN)
    sf = amax / E4M3_MAX
    q = (grouped * (E4M3_MAX / amax).unsqueeze(1)).view(m, k).to(torch.float8_e4m3fn)
    return q, sf
    # --- END SOLUTION
