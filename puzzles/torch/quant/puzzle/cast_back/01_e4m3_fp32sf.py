"""cast_back 01 (torch). See doc/quant/cast_back/01_e4m3_fp32sf.md"""

import torch

from harness.consts import CANONICAL_G


def torch_cast_back(q: torch.Tensor, sf: torch.Tensor,
                    group_size: int = CANONICAL_G, out_dtype: str = "bfloat16"):
    """Dequantize: out[m, k] = float(q[m, k]) * sf[m, k // group_size].

    Args:
        q: ``(M, K)`` **float8_e4m3fn** -- the quantized values.
        sf: ``(M, K/group_size)`` **float32** -- one positive scale per group
            of ``group_size`` consecutive channels.
        group_size: channels sharing one scale. Fixed at 32 on Ascend.
        out_dtype: ``'bfloat16'`` or ``'float32'``.

    Returns:
        ``(M, K)`` **bfloat16** by default, or **float32** when ``out_dtype`` is
        ``'float32'``.

    ``out_dtype`` selects the output type, as it does on the NPU tiers. In torch
    it is one ``.to()`` argument; there it picks the *store instruction*, which
    is why this config is worth carrying at all.
    """
    # TODO: view q as (M, K//G, G), multiply by sf.unsqueeze(-1), flatten back,
    #       and cast to out_dtype -- torch.bfloat16 or torch.float32. In torch the
    #       dtype is one argument; look at the ASC and PTO files for the same
    #       switch to see why it is worth a config at all.
    raise NotImplementedError("torch/cast_back/01_e4m3_fp32sf: implement torch_cast_back")

if __name__ == "__main__":        # not a script -- see the module docstring
    raise SystemExit("Run it through the harness:  python -m harness.check torch/cast_back/01")
