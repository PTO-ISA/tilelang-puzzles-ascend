"""per_token 03 (torch). See doc/quant/per_token/03_packed_ue8m0.md"""

import torch

from harness.consts import CANONICAL_G, E4M3_CLAMP_MIN, E4M3_MAX


def torch_per_token_cast_packed(x: torch.Tensor, group_size: int = CANONICAL_G):
    """Quantize with packed-UE8M0 scales. Returns ``(q, sf_packed)``.

    ``sf_packed`` is (M, K/group_size/2) int16 -- two exponent bytes per word.
    """
    # --- BEGIN SOLUTION hint="as variant 02 for the exponent (bits = (amax/E4M3_MAX).view(torch.int32); exp = ((bits - 1) >> 23) + 1 - 127), but do not rebuild a float32 scale. Store the biased byte instead: e8m0 = (exp + 127).to(torch.uint8), then pack two bytes per int16 with lo = e8m0[..., 0::2].to(torch.int16), hi = e8m0[..., 1::2].to(torch.int16), packed = lo | (hi << 8). Quantize by multiplying by ((127 - exp) << 23).view(torch.float32)."
    m, k = x.shape
    assert k % group_size == 0
    grouped = x.float().view(m, k // group_size, group_size)
    amax = grouped.abs().amax(dim=-1).clamp(min=E4M3_CLAMP_MIN)

    # ceil(log2(v)) from the float32 exponent field: `bits >> 23` gives
    # floor(log2(v)) + 127, and subtracting 1 first turns the floor into a
    # ceiling, so a power of two stays put and anything else rounds up.
    bits = (amax / E4M3_MAX).view(torch.int32)
    exp_sf = ((bits - 1) >> 23) + 1 - 127

    sf_inv = ((127 - exp_sf) << 23).view(torch.float32)
    q = (grouped * sf_inv.unsqueeze(-1)).view(m, k).to(torch.float8_e4m3fn)

    # Pack two exponent bytes per int16, low byte first -- the layout the
    # public API presents. Along K the bytes are already adjacent, so this is
    # just a strided pair of reads.
    e8m0 = (exp_sf + 127).to(torch.uint8)
    lo = e8m0[..., 0::2].to(torch.int16)
    hi = e8m0[..., 1::2].to(torch.int16)
    packed = lo | (hi << 8)
    return q, packed
    # --- END SOLUTION

if __name__ == "__main__":        # not a script -- see the module docstring
    raise SystemExit("Run it through the harness:  python -m harness.check torch/per_token/03")
