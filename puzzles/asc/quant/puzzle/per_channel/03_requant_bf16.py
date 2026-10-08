"""per_channel 03 (ASC). See doc/quant/per_channel/03_requant_bf16.md"""

import torch
import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S

from harness import status
from harness.consts import BLOCK_MN, CANONICAL_G, E4M3_CLAMP_MIN, E4M3_MAX

LANES = 64


@tilelang.jit(target="ascend")
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
                # TODO: stage 1 dequantize: per row, per 64-lane tile, the INPUT
                #       scale varies along K so it needs two S.vld(...,
                #       dist='BRC_B32') plus S.vsel(lo, hi, mask_low) as in
                #       per_token/01; multiply the unpacked FP8 and store float32
                #       into val_ub. Barrier. stage 2 is variant 01 unchanged,
                #       reading val_ub with a plain S.vld: the OUTPUT scale varies
                #       along M, one value per lane.
                raise NotImplementedError("asc/per_channel/03_requant_bf16: implement per_channel_requant")
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
    raise SystemExit("Run it through the harness:  python -m harness.check asc/per_channel/03")
