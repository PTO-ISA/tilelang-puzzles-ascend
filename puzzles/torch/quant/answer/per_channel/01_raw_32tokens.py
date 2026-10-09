"""per_channel 01 (torch). See doc/quant/per_channel/01_raw_32tokens.md"""

import torch

from harness.consts import BLOCK_MN, E4M3_CLAMP_MIN, E4M3_MAX


def torch_per_channel_cast(x: torch.Tensor, group_tokens: int = BLOCK_MN):
    """Quantize with one scale per channel, reduced across ``group_tokens`` rows.

    Args:
        x: ``(M, K)`` **bfloat16** -- the values to quantize. Row 0 is all zeros
            in the tests, which is what exercises the clamp.
        group_tokens: rows sharing one scale. 32 on Ascend; M must be a
            multiple of it.

    Returns:
        ``q``: ``(M, K)`` **float8_e4m3fn**.
        ``sf``: ``(M/group_tokens, K)`` **float32** -- one scale per *channel*,
        so this array is wide where per_token's was narrow.
    """
    # --- BEGIN SOLUTION hint="view x as (M//group_tokens, group_tokens, K); amax over dim=1 (the token axis, not the last axis) and clamp; sf = amax/E4M3_MAX; broadcast with unsqueeze(1)"
    m, k = x.shape
    assert m % group_tokens == 0, f"M={m} must be a multiple of {group_tokens}"
    grouped = x.float().view(m // group_tokens, group_tokens, k)
    amax = grouped.abs().amax(dim=1).clamp(min=E4M3_CLAMP_MIN)
    sf = amax / E4M3_MAX
    q = (grouped * (E4M3_MAX / amax).unsqueeze(1)).view(m, k).to(torch.float8_e4m3fn)
    return q, sf
    # --- END SOLUTION

if __name__ == "__main__":        # not a script -- see the module docstring
    raise SystemExit("Run it through the harness:  python -m harness.check torch/per_channel/01")
