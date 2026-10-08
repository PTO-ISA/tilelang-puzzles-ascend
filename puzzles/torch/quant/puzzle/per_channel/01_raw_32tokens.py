"""per_channel 01 (torch). See doc/quant/per_channel/01_raw_32tokens.md"""

import torch

from harness.consts import BLOCK_MN, E4M3_CLAMP_MIN, E4M3_MAX


def torch_per_channel_cast(x: torch.Tensor, group_tokens: int = BLOCK_MN):
    """Return ``(q, sf)`` with one scale per channel per group of tokens."""
    # TODO: view x as (M//group_tokens, group_tokens, K); amax over dim=1 (the
    #       token axis, not the last axis) and clamp; sf = amax/E4M3_MAX;
    #       broadcast with unsqueeze(1)
    raise NotImplementedError("torch/per_channel/01_raw_32tokens: implement torch_per_channel_cast")

if __name__ == "__main__":        # not a script -- see the module docstring
    raise SystemExit("Run it through the harness:  python -m harness.check torch/per_channel/01")
