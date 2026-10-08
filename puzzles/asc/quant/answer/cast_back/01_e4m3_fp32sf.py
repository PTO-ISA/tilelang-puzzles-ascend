"""cast_back 01 (ASC). See doc/quant/cast_back/01_e4m3_fp32sf.md"""

import torch
import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S

from harness import status
from harness.consts import CANONICAL_G

LANES = 64          # float32 lanes in one 256-byte vector register
SF_PAD = 64         # pad the scale buffer out to a whole register


@tilelang.jit(target="ascend", out_idx=[2])
def compile_kernel(hidden: int, out_dtype: str = "bfloat16",
                   group_size: int = CANONICAL_G):
    """Dequantize FP8 with one FP32 scale per `group_size` channels.

    `out_dtype` is a Python string, so the branch on it below is taken at trace
    time and only the chosen store is emitted -- the same mechanism as
    `per_token/06`'s `mode`. (Not the same as branching on a `T.unroll` loop
    variable, which silently takes the first arm; see doc/known-issues.md.)
    """
    assert hidden % 128 == 0, "this teaching kernel steps two 64-lane strips at a time"
    assert group_size == 32, "Ascend quant granularity is fixed at 32"
    assert out_dtype in ("bfloat16", "float32"), out_dtype
    out_t = T.bfloat16 if out_dtype == "bfloat16" else T.float32
    num_groups = hidden // group_size
    num_tokens = T.dynamic("num_tokens")

    @T.prim_func
    def cast_back(
        Q: T.Tensor((num_tokens, hidden), T.float8_e4m3fn),
        Sf: T.Tensor((num_tokens, num_groups), T.float32),
        Out: T.Tensor((num_tokens, hidden), out_t),
    ):
        # One vector core. Production runs T.Persistent over many cores; that is
        # scheduling, and it would make the simulator far slower without
        # teaching anything new about the vector unit.
        with T.Kernel(1):
            q_ub = T.alloc_shared((hidden,), T.float8_e4m3fn)
            sf_ub = T.alloc_shared((SF_PAD,), T.float32)
            out_ub = T.alloc_shared((hidden,), out_t)

            for token in T.serial(num_tokens):
                # GM -> UB, on the DMA engine.
                T.copy(Q[token, 0], q_ub)
                T.copy(Sf[token, 0], sf_ub[0:num_groups])

                # --- BEGIN SOLUTION hint="open `with T.SimdVF():`; make a mask with S.pset(32, 'PAT_VL32'); loop strip over hidden//64; load 64 FP8 values with S.vld(q_ub[col], dist='UNPK4_B8') and S.vcvt to float32; build the scale with two S.vld(..., dist='BRC_B32') and S.vsel(lo, hi, mask_low); S.vmul; then branch on out_dtype for the store: bfloat16 needs S.vsts(..., S.vcvt(scaled, T.bfloat16), dist='PK_B32') -- a *narrowing* store that packs 64 lanes of 32 bits into 64 contiguous 16-bit values -- while float32 is the plain contiguous S.vsts(..., scaled, dist='NORM_B32'). The wider dtype takes the simpler instruction."
                with T.SimdVF():
                    # Lanes 0-31 on, 32-63 off. Used to pick between the two
                    # scales that a 64-lane strip straddles.
                    mask_low = S.pset(32, "PAT_VL32")
                    for strip in T.serial(hidden // LANES):
                        col = strip * LANES
                        group = strip * (LANES // group_size)   # 2 groups per strip

                        # 64 FP8 bytes -> 64 float32 lanes, then convert.
                        raw = S.vld(q_ub[col], dist="UNPK4_B8")
                        values = S.vcvt(raw, T.float32)

                        # Build the scale vector: lanes 0-31 get group's scale,
                        # lanes 32-63 get the next group's.
                        lo = S.vld(sf_ub[group], dist="BRC_B32")
                        hi = S.vld(sf_ub[group + 1], dist="BRC_B32")
                        scale = S.vsel(lo, hi, mask_low)

                        scaled = S.vmul(values, scale)

                        # The output dtype picks the store *instruction*, and
                        # the intuition is inverted: float32 is the plain
                        # contiguous store, while bfloat16 needs a narrowing
                        # one that packs as it writes. The cheaper dtype to
                        # emit is the wider one.
                        if out_dtype == "bfloat16":
                            S.vsts(out_ub[col], S.vcvt(scaled, T.bfloat16),
                                   dist="PK_B32")      # 64 lanes -> 64 x 2 bytes
                        else:
                            S.vsts(out_ub[col], scaled,
                                   dist="NORM_B32")    # 64 lanes -> 64 x 4 bytes
                # --- END SOLUTION

                # UB -> GM.
                T.copy(out_ub, Out[token, 0])

    return cast_back


def launch(q: torch.Tensor, sf: torch.Tensor,
           out_dtype: str = "bfloat16") -> torch.Tensor:
    """Run the kernel. Returns a device tensor -- never a host-computed answer."""
    kernel = compile_kernel(q.shape[1], out_dtype)
    out = kernel(q, sf)
    status.assert_on_device(f"cast_back 01 {out_dtype}", out)
    return out

if __name__ == "__main__":        # not a script -- see the module docstring
    raise SystemExit("Run it through the harness:  python -m harness.check asc/cast_back/01")
