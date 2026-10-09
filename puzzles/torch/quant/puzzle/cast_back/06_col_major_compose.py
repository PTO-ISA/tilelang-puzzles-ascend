"""cast_back 06 (torch). See doc/quant/cast_back/05_col_major_compose.md"""

import torch

from harness.consts import CANONICAL_G


def _unpack_e2m1_bytes(q_packed: torch.Tensor) -> torch.Tensor:
    """Packed e2m1 int8 (M, K/2) -> float32 (M, K), two nibbles per byte.

    The same explicit code derived in ``cast_back/05``, repeated here so this
    variant's solution can be about composing the two layouts.
    """
    lo = q_packed.to(torch.int16) & 0x0F
    hi = (q_packed.to(torch.int16) >> 4) & 0x0F

    def decode(n: torch.Tensor) -> torch.Tensor:
        s = (n >> 3) & 0x1
        e = (n >> 1) & 0x3
        mant = n & 0x1
        sign = torch.where(s == 1, -1.0, 1.0)
        sub = mant.to(torch.float32) * 0.5
        norm = (1.0 + mant.to(torch.float32) * 0.5) * torch.pow(
            torch.tensor(2.0, device=n.device), (e - 1).to(torch.float32)
        )
        return torch.where(e == 0, sub, norm) * sign

    return torch.stack([decode(lo), decode(hi)], dim=-1).reshape(
        *q_packed.shape[:-1], q_packed.shape[-1] * 2
    )


def torch_cast_back_compose(q_packed: torch.Tensor, sf_cm: torch.Tensor,
                            group_size: int = CANONICAL_G):
    """Dequantize FP4 values with column-major packed-UE8M0 scales.

    Args:
        q_packed: ``(M, K/2)`` **int8** -- two e2m1 nibbles per byte, low first.
        sf_cm: ``(K/group_size/2, M)`` **int16** -- transposed *and* byte-packed,
            so both layout changes apply at once. Transpose it back before
            unpacking.
        group_size: channels sharing one scale. Fixed at 32 on Ascend.

    Returns:
        ``(M, K)`` **bfloat16**.
    """
    # TODO: undo both layouts, then scale per group. Transpose sf_cm back with
    #       .T.contiguous(); unpack its two exponent bytes per int16 as in variant
    #       02 (lo = wide & 0xFF, hi = (wide >> 8) & 0xFF, stacked low-byte-first,
    #       then e << 23 viewed as float32); and call _unpack_e2m1_bytes for the
    #       values -- that is variant 05 work, given back to you here. Note a
    #       broadcast load does not care about stride, so consuming the transposed
    #       layout costs nothing.
    raise NotImplementedError("torch/cast_back/06_col_major_compose: implement torch_cast_back_compose")

if __name__ == "__main__":        # not a script -- see the module docstring
    raise SystemExit("Run it through the harness:  python -m harness.check torch/cast_back/06 --role puzzle")
