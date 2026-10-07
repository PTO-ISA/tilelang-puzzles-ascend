"""Scale-factor and FP4 helpers shared by torch references (Ascend pack_factor=2)."""

import torch
import torch.nn.functional as F

from .consts import PACK_FACTOR


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def align_up(a: int, b: int) -> int:
    return ceil_div(a, b) * b


def ceil_log2_exp(sf: torch.Tensor) -> torch.Tensor:
    """Kernel ``ceil(log2(sf))``: ``((bits - 1) >> 23) + 1 - 127``."""
    bits = sf.view(torch.int32)
    return ((bits - 1) >> 23) + 1 - 127


def pow2_from_exp(exp_sf: torch.Tensor) -> torch.Tensor:
    return ((127 + exp_sf) << 23).view(torch.float32)


def inv_pow2_from_exp(exp_sf: torch.Tensor) -> torch.Tensor:
    return ((127 - exp_sf) << 23).view(torch.float32)


def e8m0_from_pow2_sf(sf_pow2: torch.Tensor) -> torch.Tensor:
    return (sf_pow2.view(torch.int32) >> 23).to(torch.uint8)


def decode_ue8m0(e8m0: torch.Tensor) -> torch.Tensor:
    """uint8 exponent byte -> float32 ``2^(e-127)``."""
    return (e8m0.to(torch.int32) << 23).view(torch.float32)


def pack_ue8m0_row_major(e8m0: torch.Tensor) -> torch.Tensor:
    """``(M, G) uint8`` -> Ascend public ``(M, G/2) int16`` (pack_factor=2)."""
    assert e8m0.shape[-1] % PACK_FACTOR == 0
    lo = e8m0[..., 0::2].to(torch.int16)
    hi = e8m0[..., 1::2].to(torch.int16)
    return lo | (hi << 8)


def unpack_ue8m0_row_major(packed: torch.Tensor) -> torch.Tensor:
    """``(M, G/2) int16`` -> ``(M, G) uint8``."""
    lo = (packed.to(torch.int32) & 0xFF).to(torch.uint8)
    hi = ((packed.to(torch.int32) >> 8) & 0xFF).to(torch.uint8)
    return torch.stack([lo, hi], dim=-1).reshape(*packed.shape[:-1], packed.shape[-1] * 2)


def decode_packed_ue8m0(packed: torch.Tensor) -> torch.Tensor:
    return decode_ue8m0(unpack_ue8m0_row_major(packed))


def unpack_e2m1_bytes(packed: torch.Tensor) -> torch.Tensor:
    """Decode packed int8 e2m1 (two nibbles per byte) to float32. (M, K/2) -> (M, K)."""
    lo = packed.to(torch.int16) & 0x0F
    hi = (packed.to(torch.int16) >> 4) & 0x0F

    def decode(n: torch.Tensor) -> torch.Tensor:
        s = (n >> 3) & 0x1
        e = (n >> 1) & 0x3
        m = n & 0x1
        sign = torch.where(s == 1, -1.0, 1.0)
        sub = m.to(torch.float32) * 0.5
        norm = (1.0 + m.to(torch.float32) * 0.5) * torch.pow(
            torch.tensor(2.0, device=n.device), (e - 1).to(torch.float32)
        )
        return torch.where(e == 0, sub, norm) * sign

    return torch.stack([decode(lo), decode(hi)], dim=-1).reshape(*packed.shape[:-1], packed.shape[-1] * 2)


def pack_e2m1_from_fp32(quant: torch.Tensor) -> torch.Tensor:
    """Bit-exact e2m1 pack. ``quant`` FP32 (M, K) even K -> int8 (M, K/2)."""
    device = quant.device
    q = quant.contiguous().view(torch.int32)
    signs = q & 0x80000000
    exps = (q >> 23) & 0xFF
    mant = q & 0x7FFFFF
    e8, e2 = 127, 1
    is_sub = exps < e8
    shift = e8 - exps - 1
    mant_pre = 0x400000 | ((mant >> 1) & ((1 << 31) - 1))
    sticky = is_sub & ((mant & 1) != 0)
    mask = (1 << shift.clamp(max=31)) - 1
    sticky = sticky | (is_sub & ((mant_pre & mask) != 0))
    mant = torch.where(is_sub, mant_pre >> shift, mant)
    exps = torch.maximum(exps, torch.tensor(e8 - e2, device=device)) - (e8 - e2)
    m2 = (mant >> 21) & 3
    guard = m2 & 1
    lsb = (m2 >> 1) & 1
    sticky = sticky | ((mant & ((1 << 21) - 1)) != 0)
    tmp = (((exps << 2) | m2) + (guard & (sticky.to(torch.int32) | lsb))) >> 1
    nibbles = (((signs >> 28) & 0xF) | torch.minimum(tmp, torch.tensor(7, device=device))).to(torch.uint8)
    h, w = quant.shape
    pair = nibbles.view(h, w // 2, 2)
    return (pair[..., 0] | (pair[..., 1] << 4)).view(torch.int8)
