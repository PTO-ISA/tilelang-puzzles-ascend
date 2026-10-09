"""per_token 06 (torch). See doc/quant/per_token/06_split_requant.md"""

import torch

from harness.consts import CANONICAL_G, E4M3_CLAMP_MIN, E4M3_MAX


def torch_sf_only(x: torch.Tensor, group_size: int = CANONICAL_G):
    """Compute only the scale factors -- no quantized output.

    Args:
        x: ``(M, K)`` **bfloat16**.
        group_size: channels sharing one scale. Fixed at 32 on Ascend.

    Returns:
        ``(M, K/group_size)`` **float32** -- ``amax/448`` per group.
    """
    # --- BEGIN SOLUTION hint="amax over each group, clamp, divide by E4M3_MAX; return just sf"
    m, k = x.shape
    grouped = x.float().view(m, k // group_size, group_size)
    amax = grouped.abs().amax(dim=-1).clamp(min=E4M3_CLAMP_MIN)
    return amax / E4M3_MAX
    # --- END SOLUTION


def torch_cast_only(x: torch.Tensor, sf: torch.Tensor, group_size: int = CANONICAL_G):
    """Quantize using scales that are given, with no amax pass.

    Args:
        x: ``(M, K)`` **bfloat16**.
        sf: ``(M, K/group_size)`` **float32** -- the scales, already computed.
        group_size: channels sharing one scale. Fixed at 32 on Ascend.

    Returns:
        ``(M, K)`` **float8_e4m3fn**.

    Mirror the kernel: take the reciprocal of the stored scale and multiply.
    That is not bit-identical to the fused path, which forms ``448/amax``
    directly -- see the variant page for the measured difference.
    """
    # --- BEGIN SOLUTION hint="sf_inv = 1.0 / sf (not E4M3_MAX/amax -- we only have sf); multiply each group by sf_inv.unsqueeze(-1) and cast to float8_e4m3fn"
    m, k = x.shape
    grouped = x.float().view(m, k // group_size, group_size)
    sf_inv = 1.0 / sf
    return (grouped * sf_inv.unsqueeze(-1)).view(m, k).to(torch.float8_e4m3fn)
    # --- END SOLUTION


def torch_requant(q_in: torch.Tensor, sf_in: torch.Tensor,
                  group_size: int = CANONICAL_G):
    """Dequantize an already-quantized input, then quantize it again.

    Args:
        q_in: ``(M, K)`` **float8_e4m3fn** -- note the first argument is the
            *quantized* tensor here, not the activations.
        sf_in: ``(M, K/group_size)`` **float32** -- the scales it was made with.
        group_size: channels sharing one scale. Fixed at 32 on Ascend.

    Returns:
        ``q``: ``(M, K)`` **float8_e4m3fn**.
        ``sf``: ``(M, K/group_size)`` **float32**.
    """
    # --- BEGIN SOLUTION hint="dequantize the input exactly as cast_back/01 did -- group q_in by group_size and multiply by sf_in.unsqueeze(-1), in float32 -- then run the ordinary variant-01 quantize on the result. The kernel needs a scratch buffer and two barriers for this because the new amax cannot be known until the whole group is dequantized."
    # Dequantize -- cast_back/01 with one scale per group_size channels.
    mi, ki = q_in.shape
    x = (q_in.float().view(mi, ki // group_size, group_size)
         * sf_in.unsqueeze(-1)).view(mi, ki)

    m, k = x.shape
    grouped = x.view(m, k // group_size, group_size)
    amax = grouped.abs().amax(dim=-1).clamp(min=E4M3_CLAMP_MIN)
    sf = amax / E4M3_MAX
    q = (grouped * (E4M3_MAX / amax).unsqueeze(-1)).view(m, k).to(torch.float8_e4m3fn)
    return q, sf
    # --- END SOLUTION

if __name__ == "__main__":        # not a script -- see the module docstring
    raise SystemExit("Run it through the harness:  python -m harness.check torch/per_token/06")
