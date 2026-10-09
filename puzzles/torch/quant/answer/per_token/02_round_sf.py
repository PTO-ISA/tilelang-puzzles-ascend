"""per_token 02 (torch). See doc/quant/per_token/02_round_sf.md"""

import torch

from harness.consts import CANONICAL_G, E4M3_CLAMP_MIN, E4M3_MAX


def torch_per_token_cast_round(x: torch.Tensor, group_size: int = CANONICAL_G):
    """Quantize with the scale rounded up to a power of two.

    Args:
        x: ``(M, K)`` **bfloat16** -- the activations to quantize. Row 0 is all
            zeros in the tests, which is what exercises the clamp.
        group_size: channels sharing one scale. Fixed at 32 on Ascend.

    Returns:
        ``q``: ``(M, K)`` **float8_e4m3fn**.
        ``sf``: ``(M, K/group_size)`` **float32**, still a full float but now
        always a power of two -- every mantissa bit is zero, which is what
        lets variant 03 store it in a single byte.
    """
    # --- BEGIN SOLUTION hint="amax as in variant 01, then round the scale up to a power of two with the float32 exponent trick. For v = amax/E4M3_MAX: bits = v.view(torch.int32); exp = ((bits - 1) >> 23) + 1 - 127 is ceil(log2(v)). Rebuild both scales by writing that exponent back into the exponent field: sf = ((127 + exp) << 23).view(torch.float32) and sf_inv = ((127 - exp) << 23).view(torch.float32). Multiply by sf_inv instead of dividing -- negating an exponent is exact."
    m, k = x.shape
    assert k % group_size == 0
    grouped = x.float().view(m, k // group_size, group_size)
    amax = grouped.abs().amax(dim=-1).clamp(min=E4M3_CLAMP_MIN)

    # ceil(log2(v)) from the exponent field. `bits >> 23` alone gives
    # floor(log2(v)) + 127; subtracting 1 first turns the floor into a ceiling,
    # so an exact power of two stays put and anything else rounds up.
    bits = (amax / E4M3_MAX).view(torch.int32)
    exp_sf = ((bits - 1) >> 23) + 1 - 127

    # Write the exponent back into a float32. Both of these are exact: they set
    # an exponent field and leave the mantissa zero, so sf_inv is exactly 1/sf.
    sf = ((127 + exp_sf) << 23).view(torch.float32)
    sf_inv = ((127 - exp_sf) << 23).view(torch.float32)

    q = (grouped * sf_inv.unsqueeze(-1)).view(m, k).to(torch.float8_e4m3fn)
    return q, sf
    # --- END SOLUTION

if __name__ == "__main__":        # not a script -- see the module docstring
    raise SystemExit("Run it through the harness:  python -m harness.check torch/per_token/02")
