"""per_channel 02 (torch). See doc/quant/per_channel/02_round_packed_m.md"""

import torch

from harness.consts import BLOCK_MN, E4M3_CLAMP_MIN, E4M3_MAX, PACK_FACTOR
from harness.math_ops import ceil_log2_exp, inv_pow2_from_exp, pack_ue8m0_along_m


def torch_per_channel_cast_packed(x: torch.Tensor, group_tokens: int = BLOCK_MN):
    """Return ``(q, sf_packed)`` with power-of-two scales packed along M.

    ``sf_packed`` is (M/group_tokens/PACK_FACTOR, K) int16.
    """
    # TODO: reduce along dim=1 as in variant 01; exp =
    #       ceil_log2_exp(amax/E4M3_MAX); multiply by
    #       inv_pow2_from_exp(exp).unsqueeze(1); return
    #       pack_ue8m0_along_m((exp+127).to(uint8))
    raise NotImplementedError("torch/per_channel/02_round_packed_m: implement torch_per_channel_cast_packed")
