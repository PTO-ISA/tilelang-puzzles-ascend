"""cast_back 02 (torch). See doc/quant/cast_back/02_packed_ue8m0.md"""

import torch

from harness.consts import CANONICAL_G


def torch_cast_back_packed(q: torch.Tensor, sf_packed: torch.Tensor,
                           group_size: int = CANONICAL_G):
    """Dequantize with packed-UE8M0 scales.

    ``sf_packed`` is (M, K/group_size/2) int16; each word holds two exponent
    bytes, low byte first.
    """
    # --- BEGIN SOLUTION hint="unpack the scales yourself, then dequantize as variant 01. Split each int16 into two bytes -- wide = sf_packed.to(torch.int32); lo = (wide & 0xFF).to(torch.uint8); hi = ((wide >> 8) & 0xFF).to(torch.uint8) -- and interleave them low-byte-first with torch.stack([lo, hi], dim=-1).reshape(M, K/group_size). A UE8M0 byte is a bare exponent, so it decodes as (e8m0.to(torch.int32) << 23).view(torch.float32)."
    m, k = q.shape
    assert k % group_size == 0

    # Split each int16 into its two exponent bytes, low byte first, and
    # interleave them back to one scale per group. A UE8M0 byte is a bare
    # exponent, so `e << 23` reassembles the float32 directly -- mantissa zero,
    # which is why the format can carry a power-of-two scale in one byte.
    wide = sf_packed.to(torch.int32)
    lo = (wide & 0xFF).to(torch.uint8)
    hi = ((wide >> 8) & 0xFF).to(torch.uint8)
    e8m0 = torch.stack([lo, hi], dim=-1).reshape(*sf_packed.shape[:-1],
                                                 sf_packed.shape[-1] * 2)
    scale = (e8m0.to(torch.int32) << 23).view(torch.float32)

    grouped = q.float().view(m, k // group_size, group_size)
    return (grouped * scale.unsqueeze(-1)).view(m, k).to(torch.bfloat16)
    # --- END SOLUTION

if __name__ == "__main__":        # not a script -- see the module docstring
    raise SystemExit("Run it through the harness:  python -m harness.check torch/cast_back/02")
