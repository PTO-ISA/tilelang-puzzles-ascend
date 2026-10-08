"""per_channel 02 (torch). See doc/quant/per_channel/02_round_packed_m.md"""

import torch

from harness.consts import BLOCK_MN, E4M3_CLAMP_MIN, E4M3_MAX


def torch_per_channel_cast_packed(x: torch.Tensor, group_tokens: int = BLOCK_MN):
    """Return ``(q, sf_packed)`` with power-of-two scales packed along M.

    ``sf_packed`` is (M/group_tokens/2, K) int16 -- two exponent bytes per word,
    packed along M.
    """
    # --- BEGIN SOLUTION hint="reduce along dim=1 as in variant 01, then the exponent trick from per_token/02: bits = (amax/E4M3_MAX).view(torch.int32); exp = ((bits - 1) >> 23) + 1 - 127; multiply by ((127 - exp) << 23).view(torch.float32).unsqueeze(1). Then pack along M rather than K: e8m0 = (exp + 127).to(torch.uint8), packed = e8m0[0::2].to(torch.int16) | (e8m0[1::2].to(torch.int16) << 8) -- note the slice is on dim 0, so the two bytes of a word come from different scale rows."
    m, k = x.shape
    assert m % group_tokens == 0
    grouped = x.float().view(m // group_tokens, group_tokens, k)
    amax = grouped.abs().amax(dim=1).clamp(min=E4M3_CLAMP_MIN)

    # ceil(log2(v)) from the float32 exponent field: `bits >> 23` gives
    # floor(log2(v)) + 127, and subtracting 1 first turns the floor into a
    # ceiling, so a power of two stays put and anything else rounds up.
    bits = (amax / E4M3_MAX).view(torch.int32)
    exp_sf = ((bits - 1) >> 23) + 1 - 127

    sf_inv = ((127 - exp_sf) << 23).view(torch.float32)
    q = (grouped * sf_inv.unsqueeze(1)).view(m, k).to(torch.float8_e4m3fn)

    # Pack along M: word [i, c] holds m-group 2i's exponent in the low byte and
    # m-group 2i+1's in the high byte. These two bytes are K apart in memory,
    # not adjacent -- which is why the NPU needs a real interleave instruction.
    e8m0 = (exp_sf + 127).to(torch.uint8)
    lo = e8m0[0::2].to(torch.int16)
    hi = e8m0[1::2].to(torch.int16)
    packed = lo | (hi << 8)
    return q, packed
    # --- END SOLUTION

if __name__ == "__main__":        # not a script -- see the module docstring
    raise SystemExit("Run it through the harness:  python -m harness.check torch/per_channel/02")
