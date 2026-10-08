"""per_channel 04 (torch). See doc/quant/per_channel/04_compose.md"""

import torch

from harness.consts import BLOCK_MN, E4M3_CLAMP_MIN, E4M3_MAX, PACK_FACTOR
from harness.math_ops import ceil_log2_exp, inv_pow2_from_exp, pack_ue8m0_along_m


def torch_per_channel_compose(x: torch.Tensor, group_tokens: int = BLOCK_MN):
    """The fully composed per_channel kernel. Returns ``(q, sf_packed)``."""
    # TODO: combine variants 01-03: reduce amax along dim=1 in bfloat16, widen,
    #       exp = ceil_log2_exp(amax/E4M3_MAX), apply
    #       inv_pow2_from_exp(exp).unsqueeze(1), and pack (exp+127) with
    #       pack_ue8m0_along_m
    raise NotImplementedError("torch/per_channel/04_compose: implement torch_per_channel_compose")

if __name__ == "__main__":        # not a script -- see the module docstring
    raise SystemExit("Run it through the harness:  python -m harness.check torch/per_channel/04")
