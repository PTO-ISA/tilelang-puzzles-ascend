"""Torch oracles: the single definition of numerical truth for this repo.

Every ASC and PTO kernel variant is checked against the function here that
carries the same config. The torch tier of the ladder is where a student
implements these by hand first; these versions are the reference answers.

Ascend defaults throughout: quant group / block size 32, UE8M0 pack factor 2.
"""

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
    pack_ue8m0_along_m,
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


def per_channel(
    x: torch.Tensor,
    group_tokens: int = 32,
    *,
    round_sf: bool = False,
    packed: bool = False,
):
    """One scale per channel, shared across ``group_tokens`` tokens.

    With ``packed``, the UE8M0 bytes are fused along **M** rather than along the
    channel axis -- see ``pack_ue8m0_along_m`` for why.
    """
    m, k = x.shape
    assert m % group_tokens == 0
    grouped = x.view(m // group_tokens, group_tokens, k)
    amax = grouped.abs().amax(dim=1).float()
    sf, sf_inv = _scale_from_amax(amax, "e4m3", round_sf=round_sf)
    quant = grouped.float() * sf_inv.unsqueeze(1)
    out = quant.view(m, k).to(torch.float8_e4m3fn)
    if packed:
        clamped = torch.clamp(amax, min=E4M3_CLAMP_MIN)
        e8m0 = (ceil_log2_exp(clamped / E4M3_MAX) + 127).to(torch.uint8)
        return out, pack_ue8m0_along_m(e8m0)
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
    # (M/bm, bm, K/bk, bk) already has its axes in memory order, so the scale
    # broadcasts straight in and the result flattens back with no permute.
    # (A trailing permute(0, 2, 1, 3) here is a layout bug: it is invisible when
    # bm == 1, because permuting a size-1 axis cannot change the linear order,
    # but it transposes the tile interior for every coarser block.)
    q_blocks = qf.view(ceil_div(m, bm), bm, ceil_div(k, bk), bk)
    out = q_blocks * scale.unsqueeze(1).unsqueeze(-1)
    return out.reshape(m, k).to(out_dtype)

# ---------------------------------------------------------------------------
# scale-factor layout
# ---------------------------------------------------------------------------

def to_col_major(sf: torch.Tensor) -> torch.Tensor:
    """Row-major ``(num_m, num_k)`` scales -> the kernel-native transposed layout.

    The GEMM that consumes these scales wants them TMA-aligned, which means the
    kernel writes ``sf[k_block, m]`` and the host restores the public
    ``(m, k_block)`` view with a ``.T``. Kernels that set
    ``use_tma_aligned_col_major_sf`` produce this layout directly, which is why
    they need an in-register transpose (see the per_token col-major variant).
    """
    return sf.T.contiguous()


# ---------------------------------------------------------------------------
# the sf_only / cast_only split
# ---------------------------------------------------------------------------

def per_token_sf_only(x, group_size=32, *, fmt="e4m3", round_sf=False, packed=False):
    """Compute the scale factors and nothing else.

    Production splits the kernel this way so a caller that only needs scales
    (to size a later pass, say) does not pay for the quantized output.
    """
    _, sf = per_token(x, group_size, fmt=fmt, round_sf=round_sf, packed=packed)
    return sf


def per_token_cast_only(x, sf, group_size=32, *, fmt="e4m3", packed=False):
    """Quantize using scales that are *given*, not computed.

    There is no amax pass at all: the kernel loads the scale, takes a reciprocal,
    and multiplies. That reciprocal is why ``cast_only`` is not always bit-identical
    to the fused kernel on the same input. The fused path forms the inverse
    directly from amax as ``max_value / amax``; ``cast_only`` only has the rounded
    stored scale ``sf``, so it computes ``1 / sf``. Those two differ in the last
    bit or two, which can flip an occasional FP8 code.

    With ``round_sf`` the difference vanishes: a power-of-two scale is exact, and
    its reciprocal is exact too (the kernel just negates the exponent field).
    That is one practical reason production prefers power-of-two scales.
    """
    m, k = x.shape
    g = group_size
    assert k % g == 0
    scale = decode_packed_ue8m0(sf) if packed else sf
    sf_inv = 1.0 / scale
    quant = (x.view(m, k // g, g).float() * sf_inv.unsqueeze(-1)).view(m, k)
    return _cast_quant(quant, fmt)


# ---------------------------------------------------------------------------
# requant: dequantize by an input scale, then quantize again
# ---------------------------------------------------------------------------

def requant_per_token(
    q_in,
    sf_in,
    group_size=32,
    *,
    in_fmt="e4m3",
    in_packed=False,
    fmt="e4m3",
    round_sf=False,
    packed=False,
):
    """Already-quantized input + its scales -> freshly quantized output.

    This is production's ``in_config.with_sf`` path. The kernel does it in two
    stages inside the VF: dequantize into a scratch UB buffer, then run the
    ordinary amax/scale/quantize over that buffer. Doing it in one pass is not
    possible because the amax of the dequantized values is not known until the
    whole group has been dequantized.
    """
    x = cast_back(
        q_in, sf_in, (1, group_size),
        packed=in_packed, fp4=(in_fmt == "e2m1"), out_dtype=torch.float32,
    )
    return per_token(x, group_size, fmt=fmt, round_sf=round_sf, packed=packed)
