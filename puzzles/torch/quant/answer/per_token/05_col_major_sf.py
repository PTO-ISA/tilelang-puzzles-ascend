"""per_token 05 (torch). See doc/quant/per_token/05_col_major_sf.md"""

import torch

from harness.consts import CANONICAL_G, E4M3_CLAMP_MIN, E4M3_MAX


def torch_per_token_cast_col_major(x: torch.Tensor, group_size: int = CANONICAL_G):
    """Quantize, returning the scales in the kernel-native column-major layout.

    Args:
        x: ``(M, K)`` **bfloat16** -- the activations to quantize. Row 0 is all
            zeros in the tests, which is what exercises the clamp.
        group_size: channels sharing one scale. Fixed at 32 on Ascend.

    Returns:
        ``q``: ``(M, K)`` **float8_e4m3fn**.
        ``sf_cm``: ``(K/group_size, M)`` **float32** -- the transpose of
        variant 02's ``(M, K/group_size)``, so the consuming GEMM can fetch one
        tile's scales contiguously. In torch this is one ``.T``; on the NPU the
        scales live in vector lanes and the same change needs a gather.
    """
    # --- BEGIN SOLUTION hint="compute (q, sf) with power-of-two scales exactly as variant 02 (bits = (amax/E4M3_MAX).view(torch.int32); exp = ((bits - 1) >> 23) + 1 - 127; sf = ((127 + exp) << 23).view(torch.float32); multiply by ((127 - exp) << 23).view(torch.float32)), then return sf.T.contiguous() instead of sf, shape (K/group_size, M). In torch that transpose is one call; on the NPU the scales sit in vector lanes and the same change needs an index vector and a gather -- see the variant page."
    m, k = x.shape
    assert k % group_size == 0
    grouped = x.float().view(m, k // group_size, group_size)
    amax = grouped.abs().amax(dim=-1).clamp(min=E4M3_CLAMP_MIN)

    # The power-of-two scale, as in variant 02.
    bits = (amax / E4M3_MAX).view(torch.int32)
    exp_sf = ((bits - 1) >> 23) + 1 - 127
    sf = ((127 + exp_sf) << 23).view(torch.float32)
    sf_inv = ((127 - exp_sf) << 23).view(torch.float32)

    q = (grouped * sf_inv.unsqueeze(-1)).view(m, k).to(torch.float8_e4m3fn)

    # The whole config, and in torch it really is just a transpose. The NPU
    # cannot restride a register, so this same line becomes an index vector and
    # a gather there -- which is what makes this the hardest variant.
    return q, sf.T.contiguous()
    # --- END SOLUTION

if __name__ == "__main__":        # not a script -- see the module docstring
    raise SystemExit("Run it through the harness:  python -m harness.check torch/per_token/05")
