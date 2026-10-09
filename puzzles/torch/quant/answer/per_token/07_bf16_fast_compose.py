"""per_token 07 (torch). See doc/quant/per_token/07_bf16_fast_compose.md"""

import torch

from harness.consts import CANONICAL_G, E4M3_CLAMP_MIN, E4M3_MAX


def torch_per_token_bf16_compose(x: torch.Tensor, group_size: int = CANONICAL_G):
    """The composed fast path: bfloat16 compute, power-of-two packed scales.

    Args:
        x: ``(M, K)`` **bfloat16** with **K a multiple of 256** -- the fast path
            steps 256 values at a time, which is why this variant runs at
            K=256 where the rest of the ladder uses 128.
        group_size: channels sharing one scale. Fixed at 32 on Ascend.

    Returns:
        ``q``: ``(M, K)`` **float8_e4m3fn**.
        ``sf_packed``: ``(M, K/group_size/2)`` **int16** -- packed UE8M0, as in
        variant 03. The reduction runs in bfloat16 but the scale uses only the
        exponent, which bfloat16 keeps exactly.
    """
    # --- BEGIN SOLUTION hint="reduce amax in bfloat16 (cast grouped to bfloat16 before .abs().amax()), then widen to float32 for the exponent math -- bfloat16 keeps float32 full exponent range, so the chosen power of two is unaffected. Then exactly variant 03: bits = (amax/E4M3_MAX).view(torch.int32); exp = ((bits - 1) >> 23) + 1 - 127; multiply by ((127 - exp) << 23).view(torch.float32); pack (exp + 127).to(torch.uint8) two bytes per int16 with lo | (hi << 8)."
    m, k = x.shape
    assert k % group_size == 0
    assert k % 256 == 0, "the bf16 fast path steps 256 values at a time"
    grouped = x.view(m, k // group_size, group_size)
    # The reduction itself runs in bfloat16 -- this is the part that doubles the
    # lane count on the NPU. Widen only afterwards, for the exponent math.
    amax = grouped.to(torch.bfloat16).abs().amax(dim=-1).float().clamp(min=E4M3_CLAMP_MIN)
    # ceil(log2(v)) from the float32 exponent field: `bits >> 23` gives
    # floor(log2(v)) + 127, and subtracting 1 first turns the floor into a
    # ceiling, so a power of two stays put and anything else rounds up.
    bits = (amax / E4M3_MAX).view(torch.int32)
    exp_sf = ((bits - 1) >> 23) + 1 - 127

    # Multiplying by a power of two is exact, so this is safe in bfloat16.
    sf_inv = ((127 - exp_sf) << 23).view(torch.float32)
    q = (grouped.float() * sf_inv.unsqueeze(-1)).view(m, k).to(torch.float8_e4m3fn)

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
    raise SystemExit("Run it through the harness:  python -m harness.check torch/per_token/07")
