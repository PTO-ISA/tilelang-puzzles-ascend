"""per_token 03 (torch). See doc/quant/per_token/03_packed_ue8m0.md"""

import torch

from harness.consts import CANONICAL_G, E4M3_CLAMP_MIN, E4M3_MAX


def torch_per_token_cast_packed(x: torch.Tensor, group_size: int = CANONICAL_G):
    """Quantize, storing each power-of-two scale as one UE8M0 exponent byte.

    Args:
        x: ``(M, K)`` **bfloat16** -- the activations to quantize. Row 0 is all
            zeros in the tests, which is what exercises the clamp.
        group_size: channels sharing one scale. Fixed at 32 on Ascend.

    Returns:
        ``q``: ``(M, K)`` **float8_e4m3fn**.
        ``sf_packed``: ``(M, K/group_size/2)`` **int16** -- two exponent bytes
        per word, low byte first. 4x less scale memory than variant 02.
    """
    # TODO: as variant 02 for the exponent (bits =
    #       (amax/E4M3_MAX).view(torch.int32); exp = ((bits - 1) >> 23) + 1 -
    #       127), but do not rebuild a float32 scale. Store the biased byte
    #       instead: e8m0 = (exp + 127).to(torch.uint8), then pack two bytes per
    #       int16 with lo = e8m0[..., 0::2].to(torch.int16), hi = e8m0[...,
    #       1::2].to(torch.int16), packed = lo | (hi << 8). Quantize by
    #       multiplying by ((127 - exp) << 23).view(torch.float32).
    raise NotImplementedError("torch/per_token/03_packed_ue8m0: implement torch_per_token_cast_packed")

if __name__ == "__main__":        # not a script -- see the module docstring
    raise SystemExit("Run it through the harness:  python -m harness.check torch/per_token/03 --role puzzle")
