"""per_token 03 (torch). See doc/quant/per_token/03_packed_ue8m0.md"""

import torch

from harness.consts import CANONICAL_G, E4M3_CLAMP_MIN, E4M3_MAX, PACK_FACTOR
from harness.math_ops import ceil_log2_exp, inv_pow2_from_exp, pack_ue8m0_row_major


def torch_per_token_cast_packed(x: torch.Tensor, group_size: int = CANONICAL_G):
    """Quantize with packed-UE8M0 scales. Returns ``(q, sf_packed)``.

    ``sf_packed`` is (M, K/group_size/PACK_FACTOR) int16.
    """
    # TODO: as variant 02, but instead of pow2_from_exp store (exp + 127) as uint8
    #       and call pack_ue8m0_row_major on it
    raise NotImplementedError("torch/per_token/03_packed_ue8m0: implement torch_per_token_cast_packed")
