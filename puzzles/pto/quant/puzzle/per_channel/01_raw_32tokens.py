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
                # TODO: no cross-lane reduction here. For each 128-channel tile:
                #       acc = V.alloc_local((1,), V.vreg(128, T.float32)) seeded
                #       with V.vbrc(T.float32(E4M3_CLAMP_MIN), size=128), then
                #       loop over the 32 rows doing acc[0] = V.vmax(acc[0],
                #       V.vabs(V.vcvt(V.vload(x_ub[row, col], size=128),
                #       'float32'), mask), mask). Store sf and its inverse
                #       contiguously. After a barrier, apply with a plain
                #       V.vload(inv_ub[col], size=128) -- no broadcast.
                raise NotImplementedError("pto/per_channel/01_raw_32tokens: implement per_channel_cast")
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
    raise SystemExit("Run it through the harness:  python -m harness.check pto/per_channel/01 --role puzzle")
