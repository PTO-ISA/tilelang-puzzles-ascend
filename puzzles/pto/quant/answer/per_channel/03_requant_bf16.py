"""per_channel 03 (PTO). See doc/quant/per_channel/03_requant_bf16.md"""

import torch
import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import vmi as V

from harness import status
from harness.consts import BLOCK_MN, CANONICAL_G, E4M3_CLAMP_MIN, E4M3_MAX

LANES = 128


@tilelang.jit(target="pto")
def compile_kernel(hidden: int, group_tokens: int = BLOCK_MN,
                   in_group: int = CANONICAL_G):
    """Per-token-quantized input -> per-channel-quantized output."""
    assert hidden % 128 == 0 and group_tokens == 32 and in_group == 32
    num_tokens = T.dynamic("num_tokens")
    num_groups = T.ceildiv(num_tokens, group_tokens)
    num_in_groups = hidden // in_group
    num_col_tiles = hidden // LANES

    @T.prim_func
    def per_channel_requant(
        Q_in: T.Tensor((num_tokens, hidden), T.float8_e4m3fn),
        Sf_in: T.Tensor((num_tokens, num_in_groups), T.float32),
        Q: T.Tensor((num_tokens, hidden), T.float8_e4m3fn),
        Sf: T.Tensor((num_groups, hidden), T.float32),
    ):
        with T.Kernel(1):
            q_in_ub = T.alloc_shared((group_tokens, hidden), T.float8_e4m3fn)
            sf_in_ub = T.alloc_shared((group_tokens, LANES), T.float32)
            val_ub = T.alloc_shared((group_tokens, hidden), T.float32)
            q_ub = T.alloc_shared((group_tokens, hidden), T.float8_e4m3fn)
            sf_ub = T.alloc_shared((hidden,), T.float32)
            inv_ub = T.alloc_shared((hidden,), T.float32)

            for mg in T.serial(num_groups):
                T.copy(Q_in[mg * group_tokens, 0], q_in_ub)
                for row in T.serial(group_tokens):
                    T.copy(Sf_in[mg * group_tokens + row, 0],
                           sf_in_ub[row, 0:num_in_groups])
                # --- BEGIN SOLUTION hint="stage 1 dequantize: per row, per 128-lane tile, the INPUT scale varies along K so fetch it with one V.vload(..., size=128, stride=1, dist_mode='brc', group=4); multiply the converted FP8 and store float32 into val_ub. Barrier. stage 2 is variant 01 unchanged, reading val_ub with a plain V.vload: the OUTPUT scale varies along M, one value per lane."
                with T.SimdVF():
                    mask = V.create_mask(LANES, size=LANES)
                    qmax = V.vbrc(T.float32(E4M3_MAX), size=LANES)
                    groups_per_tile = LANES // in_group

                    # ---- stage 1: dequantize. Scale varies along K -> one brc
                    # load with group=, where ASC needs two loads and a select.
                    for row in T.serial(group_tokens):
                        for ct in T.serial(num_col_tiles):
                            col = ct * LANES
                            g = ct * groups_per_tile
                            scale = V.vload(sf_in_ub[row, g], size=LANES, stride=1,
                                            dist_mode="brc", group=groups_per_tile)
                            raw = V.vcvt(V.vload(q_in_ub[row, col], size=LANES),
                                         "float32")
                            V.vstore(V.vmul(raw, scale, mask), val_ub[row, col])
                    T.simd.mem_bar("VST_VLD")

                    # ---- stage 2: quantize. Scale varies along M: no broadcast.
                    for ct in T.serial(num_col_tiles):
                        col = ct * LANES
                        acc = V.alloc_local((1,), V.vreg(LANES, T.float32))
                        acc[0] = V.vbrc(T.float32(E4M3_CLAMP_MIN), size=LANES)
                        for row in T.serial(group_tokens):
                            acc[0] = V.vmax(
                                acc[0],
                                V.vabs(V.vload(val_ub[row, col], size=LANES), mask),
                                mask)
                        # precision="exact" is the 0-ULP divide, as production uses.
                        V.vstore(V.vdiv(acc[0], qmax, mask, precision="exact"),
                                 sf_ub[col])
                        V.vstore(V.vdiv(qmax, acc[0], mask, precision="exact"),
                                 inv_ub[col])
                    T.simd.mem_bar("VST_VLD")

                    for ct in T.serial(num_col_tiles):
                        col = ct * LANES
                        inv = V.vload(inv_ub[col], size=LANES)   # one plain load
                        for row in T.serial(group_tokens):
                            V.vstore(V.vcvt(
                                V.vmul(V.vload(val_ub[row, col], size=LANES), inv,
                                       mask), "float8_e4m3fn", rounding="R",
                                saturate="SAT"), q_ub[row, col])
                # --- END SOLUTION
                T.copy(q_ub, Q[mg * group_tokens, 0])
                T.copy(sf_ub, Sf[mg, 0])

    return per_channel_requant


def launch(q_in: torch.Tensor, sf_in: torch.Tensor):
    m, hidden = q_in.shape
    q = torch.empty((m, hidden), dtype=torch.float8_e4m3fn, device=q_in.device)
    sf = torch.empty((m // BLOCK_MN, hidden), dtype=torch.float32, device=q_in.device)
    compile_kernel(hidden)(q_in, sf_in, q, sf)
    status.assert_on_device("per_channel 03", q, sf)
    return q, sf

if __name__ == "__main__":        # not a script -- see the module docstring
    raise SystemExit("Run it through the harness:  python -m harness.check pto/per_channel/03")
