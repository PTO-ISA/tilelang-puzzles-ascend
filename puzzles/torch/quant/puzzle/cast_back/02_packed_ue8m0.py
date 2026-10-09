"""cast_back 02 (torch). See doc/quant/cast_back/02_packed_ue8m0.md"""

import torch

from harness.consts import CANONICAL_G


def torch_cast_back_packed(q: torch.Tensor, sf_packed: torch.Tensor,
                           group_size: int = CANONICAL_G):
    """Dequantize with packed-UE8M0 scales.

    Args:
        q: ``(M, K)`` **float8_e4m3fn** -- the quantized values.
        sf_packed: ``(M, K/group_size/2)`` **int16** -- each word holds two
            UE8M0 exponent bytes, low byte first. A byte ``e`` means the scale
            ``2^(e-127)``, so there is no mantissa to store.
        group_size: channels sharing one scale. Fixed at 32 on Ascend.

    Returns:
        ``(M, K)`` **bfloat16**.
    """
    # TODO: unpack the scales yourself, then dequantize as variant 01. Split each
    #       int16 into two bytes -- wide = sf_packed.to(torch.int32); lo = (wide &
    #       0xFF).to(torch.uint8); hi = ((wide >> 8) & 0xFF).to(torch.uint8) --
    #       and interleave them low-byte-first with torch.stack([lo, hi],
    #       dim=-1).reshape(M, K/group_size). A UE8M0 byte is a bare exponent, so
    #       it decodes as (e8m0.to(torch.int32) << 23).view(torch.float32).
    raise NotImplementedError("torch/cast_back/02_packed_ue8m0: implement torch_cast_back_packed")

if __name__ == "__main__":        # not a script -- see the module docstring
    raise SystemExit("Run it through the harness:  python -m harness.check torch/cast_back/02")
