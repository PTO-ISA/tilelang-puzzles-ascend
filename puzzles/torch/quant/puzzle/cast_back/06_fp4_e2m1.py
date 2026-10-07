"""cast_back 06 (torch). See doc/quant/cast_back/06_fp4_e2m1.md"""

import torch

from common.consts import CANONICAL_G
from common.math_ops import unpack_e2m1_bytes


def torch_cast_back_fp4(q_packed: torch.Tensor, sf: torch.Tensor,
                        group_size: int = CANONICAL_G):
    """Dequantize packed-FP4 input.

    ``q_packed`` is (M, K/2) int8; the logical width is K = 2 * q_packed.shape[1].
    """
    # TODO: unpack_e2m1_bytes(q_packed) gives (M, K) float32; then scale per group
    #       exactly as variant 01
    raise NotImplementedError("torch/cast_back/06_fp4_e2m1: implement torch_cast_back_fp4")
