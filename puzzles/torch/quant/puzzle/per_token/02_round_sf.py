"""per_token 02 (torch). See doc/quant/per_token/02_round_sf.md"""

import torch

from harness.consts import CANONICAL_G, E4M3_CLAMP_MIN, E4M3_MAX
from harness.math_ops import ceil_log2_exp, inv_pow2_from_exp, pow2_from_exp


def torch_per_token_cast_round(x: torch.Tensor, group_size: int = CANONICAL_G):
    """Quantize with a power-of-two scale. Returns ``(q, sf)``, sf still float32."""
    # TODO: amax as in variant 01; exp = ceil_log2_exp(amax/E4M3_MAX); sf =
    #       pow2_from_exp(exp); multiply by inv_pow2_from_exp(exp) instead of
    #       dividing
    raise NotImplementedError("torch/per_token/02_round_sf: implement torch_per_token_cast_round")
