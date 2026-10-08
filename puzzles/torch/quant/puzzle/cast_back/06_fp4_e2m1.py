"""cast_back 06 (torch). See doc/quant/cast_back/06_fp4_e2m1.md"""

import torch

from harness.consts import CANONICAL_G


def torch_cast_back_fp4(q_packed: torch.Tensor, sf: torch.Tensor,
                        group_size: int = CANONICAL_G):
    """Dequantize packed-FP4 input.

    ``q_packed`` is (M, K/2) int8; the logical width is K = 2 * q_packed.shape[1].
    """
    # TODO: decode the nibbles yourself, then scale per group as variant 01. Each
    #       byte holds two values, low nibble first: lo = q_packed.to(torch.int16)
    #       & 0x0F and hi = (q_packed.to(torch.int16) >> 4) & 0x0F. A nibble is
    #       sign|exp(2)|mant(1): s = (n >> 3) & 1, e = (n >> 1) & 3, mant = n & 1.
    #       With e == 0 the value is subnormal -- mant * 0.5, no implicit leading
    #       one -- otherwise it is (1 + mant/2) * 2**(e - 1), bias 1. Apply the
    #       sign, then torch.stack([decode(lo), decode(hi)], dim=-1).reshape to
    #       (M, K) so the two nibbles of a byte stay adjacent.
    raise NotImplementedError("torch/cast_back/06_fp4_e2m1: implement torch_cast_back_fp4")

if __name__ == "__main__":        # not a script -- see the module docstring
    raise SystemExit("Run it through the harness:  python -m harness.check torch/cast_back/06")
