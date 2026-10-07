"""per_token 04 (torch). See doc/quant/per_token/04_fp32_in_fp4_out.md"""

import torch

from harness.consts import CANONICAL_G, E2M1_CLAMP_MIN, E2M1_MAX
from harness.math_ops import pack_e2m1_from_fp32


def torch_per_token_cast_fp4(x: torch.Tensor, group_size: int = CANONICAL_G):
    """Quantize a float32 input to packed FP4. Returns ``(q_packed, sf)``."""
    # TODO: same shape as variant 01 but with E2M1_MAX / E2M1_CLAMP_MIN instead of
    #       the e4m3 constants, and pack_e2m1_from_fp32(quant) instead of
    #       .to(float8_e4m3fn)
    raise NotImplementedError("torch/per_token/04_fp32_in_fp4_out: implement torch_per_token_cast_fp4")
