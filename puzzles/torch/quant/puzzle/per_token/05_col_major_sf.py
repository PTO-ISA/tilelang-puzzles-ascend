"""per_token 05 (torch). See doc/quant/per_token/05_col_major_sf.md"""

import torch

from harness.consts import CANONICAL_G, E4M3_CLAMP_MIN, E4M3_MAX


def torch_per_token_cast_col_major(x: torch.Tensor, group_size: int = CANONICAL_G):
    """Quantize, returning the scales in the kernel-native column-major layout.

    Returns ``(q, sf_cm)`` where ``sf_cm`` is (K/group_size, M).
    """
    # TODO: compute (q, sf) with power-of-two scales exactly as variant 02 (bits =
    #       (amax/E4M3_MAX).view(torch.int32); exp = ((bits - 1) >> 23) + 1 - 127;
    #       sf = ((127 + exp) << 23).view(torch.float32); multiply by ((127 - exp)
    #       << 23).view(torch.float32)), then return sf.T.contiguous() instead of
    #       sf, shape (K/group_size, M). In torch that transpose is one call; on
    #       the NPU the scales sit in vector lanes and the same change needs an
    #       index vector and a gather -- see the variant page.
    raise NotImplementedError("torch/per_token/05_col_major_sf: implement torch_per_token_cast_col_major")

if __name__ == "__main__":        # not a script -- see the module docstring
    raise SystemExit("Run it through the harness:  python -m harness.check torch/per_token/05")
