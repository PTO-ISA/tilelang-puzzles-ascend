"""per_token 07 (torch). See doc/quant/per_token/07_bf16_fast_compose.md"""

import torch

from harness.consts import CANONICAL_G, E4M3_CLAMP_MIN, E4M3_MAX, PACK_FACTOR
from harness.math_ops import ceil_log2_exp, inv_pow2_from_exp, pack_ue8m0_row_major


def torch_per_token_bf16_compose(x: torch.Tensor, group_size: int = CANONICAL_G):
    """The fully composed variant: bf16 compute, pow2 packed scale, col-major.

    Returns ``(q, sf_packed)``: FP8 values and the scales as packed UE8M0
    int16, shape ``(M, K/group_size/PACK_FACTOR)``.
    """
    # TODO: reduce amax in bfloat16 (cast grouped to bfloat16 before
    #       .abs().amax()), then widen to float32 for the exponent math; exp =
    #       ceil_log2_exp(amax/E4M3_MAX); multiply by inv_pow2_from_exp(exp); pack
    #       (exp+127) with pack_ue8m0_row_major
    raise NotImplementedError("torch/per_token/07_bf16_fast_compose: implement torch_per_token_bf16_compose")
