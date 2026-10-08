"""per_token 07 (torch). See doc/quant/per_token/07_bf16_fast_compose.md"""

import torch

from harness.consts import CANONICAL_G, E4M3_CLAMP_MIN, E4M3_MAX


def torch_per_token_bf16_compose(x: torch.Tensor, group_size: int = CANONICAL_G):
    """The fully composed variant: bf16 compute, pow2 packed scale, col-major.

    Returns ``(q, sf_packed)``: FP8 values and the scales as packed UE8M0
    int16, shape ``(M, K/group_size/2)``.
    """
    # TODO: reduce amax in bfloat16 (cast grouped to bfloat16 before
    #       .abs().amax()), then widen to float32 for the exponent math --
    #       bfloat16 keeps float32 full exponent range, so the chosen power of two
    #       is unaffected. Then exactly variant 03: bits =
    #       (amax/E4M3_MAX).view(torch.int32); exp = ((bits - 1) >> 23) + 1 - 127;
    #       multiply by ((127 - exp) << 23).view(torch.float32); pack (exp +
    #       127).to(torch.uint8) two bytes per int16 with lo | (hi << 8).
    raise NotImplementedError("torch/per_token/07_bf16_fast_compose: implement torch_per_token_bf16_compose")

if __name__ == "__main__":        # not a script -- see the module docstring
    raise SystemExit("Run it through the harness:  python -m harness.check torch/per_token/07")
