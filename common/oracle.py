"""Compact torch oracles used by puzzle checkers (Ascend G=32 defaults)."""

import torch
import torch.nn.functional as F

from .consts import E2M1_CLAMP_MIN, E2M1_MAX, E4M3_CLAMP_MIN, E4M3_MAX
from .math_ops import (
    ceil_div,
    ceil_log2_exp,
    decode_packed_ue8m0,
    decode_ue8m0,
    inv_pow2_from_exp,
    pack_e2m1_from_fp32,
    pack_ue8m0_row_major,
    pow2_from_exp,
    unpack_e2m1_bytes,
)


def _fmt_consts(fmt: str) -> tuple[float, float]:
    if fmt == "e4m3":
        return E4M3_MAX, E4M3_CLAMP_MIN
    if fmt == "e2m1":
        return E2M1_MAX, E2M1_CLAMP_MIN
    raise ValueError(fmt)


def _scale_from_amax(amax: torch.Tensor, fmt: str, *, round_sf: bool):
    max_v, clamp = _fmt_consts(fmt)
    amax = torch.clamp(amax, min=clamp)
    if not round_sf:
        return amax / max_v, max_v / amax
    exp_sf = ceil_log2_exp(amax / max_v)
    return pow2_from_exp(exp_sf), inv_pow2_from_exp(exp_sf)


def _cast_quant(quant: torch.Tensor, fmt: str) -> torch.Tensor:
    if fmt == "e4m3":
        return quant.to(torch.float8_e4m3fn)
    return pack_e2m1_from_fp32(quant)


def per_token(
    x: torch.Tensor,
    group_size: int = 32,
    *,
    fmt: str = "e4m3",
    round_sf: bool = False,
    packed: bool = False,
):
    m, k = x.shape
    assert k % group_size == 0
    g = group_size
    grouped = x.view(m, k // g, g)
    amax = grouped.abs().amax(dim=-1).float()
    sf, sf_inv = _scale_from_amax(amax, fmt, round_sf=round_sf)
    quant = (grouped.float() * sf_inv.unsqueeze(-1)).view(m, k)
    out = _cast_quant(quant, fmt)
    if packed:
        e8m0 = (ceil_log2_exp(torch.clamp(amax, min=_fmt_consts(fmt)[1]) / _fmt_consts(fmt)[0]) + 127).to(
            torch.uint8
        )
        return out, pack_ue8m0_row_major(e8m0)
    return out, sf


def per_block(
    x: torch.Tensor,
    block: tuple = (32, 32),
    *,
    fmt: str = "e4m3",
    round_sf: bool = False,
    packed: bool = False,
):
    m, k = x.shape
    bm, bk = block
    assert m % bm == 0 and k % bk == 0
    tiles = x.view(m // bm, bm, k // bk, bk).permute(0, 2, 1, 3)
    amax = tiles.abs().amax(dim=(-1, -2)).float()
    sf, sf_inv = _scale_from_amax(amax, fmt, round_sf=round_sf)
    quant = tiles.float() * sf_inv.unsqueeze(-1).unsqueeze(-1)
    quant = quant.permute(0, 2, 1, 3).contiguous().view(m, k)
    out = _cast_quant(quant, fmt)
    if packed:
        e8m0 = (ceil_log2_exp(torch.clamp(amax, min=_fmt_consts(fmt)[1]) / _fmt_consts(fmt)[0]) + 127).to(
            torch.uint8
        )
        return out, pack_ue8m0_row_major(e8m0)
    return out, sf


def per_channel(x: torch.Tensor, group_tokens: int = 32, *, round_sf: bool = False):
    m, k = x.shape
    assert m % group_tokens == 0
    grouped = x.view(m // group_tokens, group_tokens, k)
    amax = grouped.abs().amax(dim=1).float()
    sf, sf_inv = _scale_from_amax(amax, "e4m3", round_sf=round_sf)
    quant = grouped.float() * sf_inv.unsqueeze(1)
    out = quant.view(m, k).to(torch.float8_e4m3fn)
    return out, sf


def cast_back(
    q: torch.Tensor,
    sf: torch.Tensor,
    sf_block: tuple = (1, 32),
    *,
    packed: bool = False,
    col_major: bool = False,
    out_dtype=torch.bfloat16,
    fp4: bool = False,
):
    qf = unpack_e2m1_bytes(q) if fp4 else q.float()
    m, k = qf.shape
    bm, bk = sf_block
    if packed:
        scale = decode_packed_ue8m0(sf) if sf.dtype == torch.int16 else decode_ue8m0(sf)
    elif col_major:
        scale = sf.T if sf.shape[0] != ceil_div(m, bm) else sf
    else:
        scale = sf
    scale = scale[: ceil_div(m, bm), : ceil_div(k, bk)]
    assert m % bm == 0 and k % bk == 0
    q_blocks = qf.view(ceil_div(m, bm), bm, ceil_div(k, bk), bk)
    out = q_blocks * scale.unsqueeze(1).unsqueeze(-1)
    return out.permute(0, 2, 1, 3).contiguous().view(m, k).to(out_dtype)
