"""per_channel 01 (ASC). See doc/quant/per_channel/01_raw_32tokens.md"""

import torch
import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S

from harness import status
from harness.consts import BLOCK_MN, E4M3_CLAMP_MIN, E4M3_MAX

LANES = 64


@tilelang.jit(target="ascend")
def compile_kernel(hidden: int, group_tokens: int = BLOCK_MN):
    """Quantize bfloat16 -> FP8 with one FP32 scale per channel per token group."""
    assert hidden % LANES == 0, "channels are processed 64 at a time"
    assert group_tokens == 32
    num_tokens = T.dynamic("num_tokens")
    num_groups = T.ceildiv(num_tokens, group_tokens)
    num_col_tiles = hidden // LANES

    @T.prim_func
    def per_channel_cast(
        X: T.Tensor((num_tokens, hidden), T.bfloat16),
        Q: T.Tensor((num_tokens, hidden), T.float8_e4m3fn),
        Sf: T.Tensor((num_groups, hidden), T.float32),
    ):
        with T.Kernel(1):
            # a whole token group lives in UB, because the reduction spans it
            x_ub = T.alloc_shared((group_tokens, hidden), T.bfloat16)
            q_ub = T.alloc_shared((group_tokens, hidden), T.float8_e4m3fn)
            sf_ub = T.alloc_shared((hidden,), T.float32)
            inv_ub = T.alloc_shared((hidden,), T.float32)

            for mg in T.serial(num_groups):
                T.copy(X[mg * group_tokens, 0], x_ub)
                # --- BEGIN SOLUTION hint="no cross-lane reduction here. For each 64-channel tile: acc = S.alloc_local((1,), T.float32) seeded with S.vdup(E4M3_CLAMP_MIN, T.float32), then loop over the 32 rows doing acc[0] = S.vmax(acc[0], S.vabs(S.vcvt(S.vld(x_ub[row, col], dist='UNPK_B16'), T.float32))). Store sf and its inverse contiguously. After a barrier, apply with a plain S.vld(inv_ub[col]) -- no broadcast."
                with T.SimdVF():
                    qmax = S.vdup(E4M3_MAX, T.float32)
                    for ct in T.serial(num_col_tiles):
                        col = ct * LANES
                        # immutable SIMD values: the accumulator is a register array
                        acc = S.alloc_local((1,), T.float32)
                        # seeding with the clamp floor makes the clamp free
                        acc[0] = S.vdup(E4M3_CLAMP_MIN, T.float32)
                        for row in T.serial(group_tokens):
                            v = S.vabs(S.vcvt(S.vld(x_ub[row, col], dist="UNPK_B16"),
                                              T.float32))
                            # lane c holds the max over tokens seen so far
                            acc[0] = S.vmax(acc[0], v)
                        S.vsts(sf_ub[col], S.vdiv(acc[0], qmax))
                        S.vsts(inv_ub[col], S.vdiv(qmax, acc[0]))
                    S.mem_bar("VST_VLD")

                    for ct in T.serial(num_col_tiles):
                        col = ct * LANES
                        # the scale is already one value per lane
                        inv = S.vld(inv_ub[col])
                        for row in T.serial(group_tokens):
                            v = S.vcvt(S.vld(x_ub[row, col], dist="UNPK_B16"),
                                       T.float32)
                            S.vsts(q_ub[row, col],
                                   S.vcvt(S.vmul(v, inv), T.float8_e4m3fn),
                                   dist="PK4_B32")
                # --- END SOLUTION
                T.copy(q_ub, Q[mg * group_tokens, 0])
                T.copy(sf_ub, Sf[mg, 0])

    return per_channel_cast


def launch(x: torch.Tensor):
    m, hidden = x.shape
    q = torch.empty((m, hidden), dtype=torch.float8_e4m3fn, device=x.device)
    sf = torch.empty((m // BLOCK_MN, hidden), dtype=torch.float32, device=x.device)
    compile_kernel(hidden)(x, q, sf)
    status.assert_on_device("per_channel 01", q, sf)
    return q, sf

if __name__ == "__main__":        # not a script -- see the module docstring
    raise SystemExit("Run it through the harness:  python -m harness.check asc/per_channel/01")
