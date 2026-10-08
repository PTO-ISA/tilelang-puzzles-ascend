"""cast_back 05 (torch). See doc/quant/cast_back/04_fp4_e2m1.md"""

import torch

from harness.consts import CANONICAL_G


def torch_cast_back_fp4(q_packed: torch.Tensor, sf: torch.Tensor,
                        group_size: int = CANONICAL_G):
    """Dequantize packed-FP4 input.

    ``q_packed`` is (M, K/2) int8; the logical width is K = 2 * q_packed.shape[1].
    """
    # --- BEGIN SOLUTION hint="decode the nibbles yourself, then scale per group as variant 01. Each byte holds two values, low nibble first: lo = q_packed.to(torch.int16) & 0x0F and hi = (q_packed.to(torch.int16) >> 4) & 0x0F. A nibble is sign|exp(2)|mant(1): s = (n >> 3) & 1, e = (n >> 1) & 3, mant = n & 1. With e == 0 the value is subnormal -- mant * 0.5, no implicit leading one -- otherwise it is (1 + mant/2) * 2**(e - 1), bias 1. Apply the sign, then torch.stack([decode(lo), decode(hi)], dim=-1).reshape to (M, K) so the two nibbles of a byte stay adjacent."
    # Decode the 16 e2m1 codes. Only 8 distinct magnitudes exist
    # -- 0, 0.5, 1, 1.5, 2, 3, 4, 6 -- so this table is the whole format.
    lo = q_packed.to(torch.int16) & 0x0F
    hi = (q_packed.to(torch.int16) >> 4) & 0x0F

    def decode(n: torch.Tensor) -> torch.Tensor:
        s = (n >> 3) & 0x1          # sign bit
        e = (n >> 1) & 0x3          # 2 exponent bits
        mant = n & 0x1              # 1 mantissa bit
        sign = torch.where(s == 1, -1.0, 1.0)
        # e == 0 is the subnormal row: value is 0 or 0.5, no implicit leading 1.
        sub = mant.to(torch.float32) * 0.5
        # Otherwise the usual (1 + m/2) * 2^(e - 1), bias 1.
        norm = (1.0 + mant.to(torch.float32) * 0.5) * torch.pow(
            torch.tensor(2.0, device=n.device), (e - 1).to(torch.float32)
        )
        return torch.where(e == 0, sub, norm) * sign

    values = torch.stack([decode(lo), decode(hi)], dim=-1).reshape(
        *q_packed.shape[:-1], q_packed.shape[-1] * 2
    )

    m, k = values.shape
    assert k % group_size == 0
    grouped = values.view(m, k // group_size, group_size)
    return (grouped * sf.unsqueeze(-1)).view(m, k).to(torch.bfloat16)
    # --- END SOLUTION

if __name__ == "__main__":        # not a script -- see the module docstring
    raise SystemExit("Run it through the harness:  python -m harness.check torch/cast_back/05")
