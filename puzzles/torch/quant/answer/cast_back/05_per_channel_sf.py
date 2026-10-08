"""cast_back 05 (torch). See doc/quant/cast_back/05_per_channel_sf.md"""

import torch

from harness.consts import BLOCK_MN


def torch_cast_back_per_channel(q: torch.Tensor, sf: torch.Tensor,
                                group_tokens: int = BLOCK_MN):
    """Dequantize with per-channel scales shared over ``group_tokens`` rows."""
    # --- BEGIN SOLUTION hint="view q as (M/group_tokens, group_tokens, K); sf is (M/group_tokens, K) so unsqueeze at dim 1 to broadcast over the token axis"
    m, k = q.shape
    assert m % group_tokens == 0
    grouped = q.float().view(m // group_tokens, group_tokens, k)
    out = grouped * sf.unsqueeze(1)
    return out.view(m, k).to(torch.bfloat16)
    # --- END SOLUTION

if __name__ == "__main__":        # not a script -- see the module docstring
    raise SystemExit("Run it through the harness:  python -m harness.check torch/cast_back/05")
