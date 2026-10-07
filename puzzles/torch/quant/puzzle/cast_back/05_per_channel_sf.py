"""cast_back 05 (torch). See doc/quant/cast_back/05_per_channel_sf.md"""

import torch

from harness.consts import BLOCK_MN


def torch_cast_back_per_channel(q: torch.Tensor, sf: torch.Tensor,
                                group_tokens: int = BLOCK_MN):
    """Dequantize with per-channel scales shared over ``group_tokens`` rows."""
    # TODO: view q as (M/group_tokens, group_tokens, K); sf is (M/group_tokens, K)
    #       so unsqueeze at dim 1 to broadcast over the token axis
    raise NotImplementedError("torch/cast_back/05_per_channel_sf: implement torch_cast_back_per_channel")
