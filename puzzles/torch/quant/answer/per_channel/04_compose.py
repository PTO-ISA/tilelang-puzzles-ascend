"""per_channel 04 (torch). See doc/quant/per_channel/04_compose.md"""

import torch

from harness.consts import BLOCK_MN, E4M3_CLAMP_MIN, E4M3_MAX, PACK_FACTOR


def torch_per_channel_compose(x: torch.Tensor, group_tokens: int = BLOCK_MN):
    """The composed per_channel kernel: bfloat16 reduction, packed along M.

    Args:
        x: ``(M, K)`` **bfloat16** -- the values to quantize. Row 0 is all zeros
            in the tests, which is what exercises the clamp.
            M must be a multiple of ``2 * group_tokens`` (64 by default).
        group_tokens: rows sharing one scale. 32 on Ascend.

    Returns:
        ``q``: ``(M, K)`` **float8_e4m3fn**.
        ``sf_packed``: ``(M/group_tokens/2, K)`` **int16** -- packed UE8M0 along
        M, as in variant 02.
    """
    # --- BEGIN SOLUTION hint="combine variants 01-02: reduce amax along dim=1 in bfloat16 then widen to float32 (bfloat16 keeps the full exponent range, so the chosen power of two is unchanged); bits = (amax/E4M3_MAX).view(torch.int32); exp = ((bits - 1) >> 23) + 1 - 127; apply ((127 - exp) << 23).view(torch.float32).unsqueeze(1); and pack (exp + 127).to(torch.uint8) along M with e8m0[0::2] | (e8m0[1::2] << 8) as in variant 02."
    m, k = x.shape
    assert m % group_tokens == 0
    assert (m // group_tokens) % PACK_FACTOR == 0, (
        f"packing along M needs an even number of token groups; "
        f"M={m} gives {m // group_tokens}"
    )
    grouped = x.view(m // group_tokens, group_tokens, k)
    amax = grouped.to(torch.bfloat16).abs().amax(dim=1).float().clamp(min=E4M3_CLAMP_MIN)

    # ceil(log2(v)) from the float32 exponent field: `bits >> 23` gives
    # floor(log2(v)) + 127, and subtracting 1 first turns the floor into a
    # ceiling, so a power of two stays put and anything else rounds up.
    bits = (amax / E4M3_MAX).view(torch.int32)
    exp_sf = ((bits - 1) >> 23) + 1 - 127

    sf_inv = ((127 - exp_sf) << 23).view(torch.float32)
    q = (grouped.float() * sf_inv.unsqueeze(1)).view(m, k)

    # Pack along M: word [i, c] holds m-group 2i's exponent in the low byte and
    # m-group 2i+1's in the high byte. These two bytes are K apart in memory,
    # not adjacent -- which is why the NPU needs a real interleave instruction.
    e8m0 = (exp_sf + 127).to(torch.uint8)
    lo = e8m0[0::2].to(torch.int16)
    hi = e8m0[1::2].to(torch.int16)
    packed = lo | (hi << 8)
    return q.to(torch.float8_e4m3fn), packed
    # --- END SOLUTION

if __name__ == "__main__":        # not a script -- see the module docstring
    raise SystemExit("Run it through the harness:  python -m harness.check torch/per_channel/04")
