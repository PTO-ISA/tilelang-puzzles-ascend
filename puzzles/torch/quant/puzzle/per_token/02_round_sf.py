"""per_token 02 (torch). See doc/quant/per_token/02_round_sf.md"""

import torch

from harness.consts import CANONICAL_G, E4M3_CLAMP_MIN, E4M3_MAX


def torch_per_token_cast_round(x: torch.Tensor, group_size: int = CANONICAL_G):
    """Quantize with a power-of-two scale. Returns ``(q, sf)``, sf still float32."""
    # TODO: amax as in variant 01, then round the scale up to a power of two with
    #       the float32 exponent trick. For v = amax/E4M3_MAX: bits =
    #       v.view(torch.int32); exp = ((bits - 1) >> 23) + 1 - 127 is
    #       ceil(log2(v)). Rebuild both scales by writing that exponent back into
    #       the exponent field: sf = ((127 + exp) << 23).view(torch.float32) and
    #       sf_inv = ((127 - exp) << 23).view(torch.float32). Multiply by sf_inv
    #       instead of dividing -- negating an exponent is exact.
    raise NotImplementedError("torch/per_token/02_round_sf: implement torch_per_token_cast_round")

if __name__ == "__main__":        # not a script -- see the module docstring
    raise SystemExit("Run it through the harness:  python -m harness.check torch/per_token/02")
