"""per_channel 01 (PTO). See doc/quant/per_channel/01_raw_32tokens.md"""

import torch
import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import vmi as V

from harness import status
from harness.consts import BLOCK_MN, E4M3_CLAMP_MIN, E4M3_MAX

LANES = 128


@tilelang.jit(target="pto")
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
                # --- BEGIN SOLUTION hint="no cross-lane reduction here. For each 128-channel tile: acc = V.alloc_local((1,), V.vreg(128, T.float32)) seeded with V.vbrc(T.float32(E4M3_CLAMP_MIN), size=128), then loop over the 32 rows doing acc[0] = V.vmax(acc[0], V.vabs(V.vcvt(V.vload(x_ub[row, col], size=128), 'float32'), mask), mask). Store sf and its inverse contiguously. After a barrier, apply with a plain V.vload(inv_ub[col], size=128) -- no broadcast."
                with T.SimdVF():
                    mask = V.create_mask(LANES, size=LANES)
                    qmax = V.vbrc(T.float32(E4M3_MAX), size=LANES)
                    for ct in T.serial(num_col_tiles):
                        col = ct * LANES
                        # immutable SIMD values: the accumulator is a register
                        # array, and its vector type is explicit.
                        acc = V.alloc_local((1,), V.vreg(LANES, T.float32))
                        # seeding with the clamp floor makes the clamp free
                        acc[0] = V.vbrc(T.float32(E4M3_CLAMP_MIN), size=LANES)
                        for row in T.serial(group_tokens):
                            v = V.vabs(V.vcvt(V.vload(x_ub[row, col], size=LANES),
                                              "float32"), mask)
                            # lane c holds the max over tokens seen so far
                            acc[0] = V.vmax(acc[0], v, mask)
                        V.vstore(V.vdiv(acc[0], qmax, mask), sf_ub[col])
                        V.vstore(V.vdiv(qmax, acc[0], mask), inv_ub[col])
                    T.simd.mem_bar("VST_VLD")

                    for ct in T.serial(num_col_tiles):
                        col = ct * LANES
                        # the scale is already one value per lane
                        inv = V.vload(inv_ub[col], size=LANES)
                        for row in T.serial(group_tokens):
                            v = V.vcvt(V.vload(x_ub[row, col], size=LANES), "float32")
                            V.vstore(V.vcvt(V.vmul(v, inv, mask), "float8_e4m3fn",
                                            rounding="R", saturate="SAT"),
                                     q_ub[row, col])
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
