"""cast_back 03 (torch). See doc/quant/cast_back/03_packed_ue8m0.md"""

import torch

from harness.consts import CANONICAL_G
from harness.math_ops import decode_packed_ue8m0


def torch_cast_back_packed(q: torch.Tensor, sf_packed: torch.Tensor,
                           group_size: int = CANONICAL_G):
    """Dequantize with packed-UE8M0 scales.

    ``sf_packed`` is (M, K/group_size/2) int16; each word holds two exponent
    bytes, low byte first.
    """
    # --- BEGIN SOLUTION hint="decode_packed_ue8m0(sf_packed) gives (M, K/G) float32 scales; then dequantize exactly as variant 01"
    m, k = q.shape
    assert k % group_size == 0
    scale = decode_packed_ue8m0(sf_packed)
    grouped = q.float().view(m, k // group_size, group_size)
    return (grouped * scale.unsqueeze(-1)).view(m, k).to(torch.bfloat16)
    # --- END SOLUTION

if __name__ == "__main__":        # not a script -- see the module docstring
    raise SystemExit("Run it through the harness:  python -m harness.check torch/cast_back/03")
