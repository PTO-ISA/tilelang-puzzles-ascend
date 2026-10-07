"""cast_back 02 (ASC) -- float32 output instead of bfloat16.

Only the store changes. In torch this was one `.to()` argument
(puzzles/torch/.../02_fp32_out.py); here it is a different distribution mode, and
the reason is worth a paragraph.

A vector register is 256 bytes = 64 float32 lanes. The *values* in those lanes are
already float32, so writing float32 is the direct case:

    S.vsts(out_ub[col], values, dist="NORM_B32")     # 64 lanes -> 64 x 4 bytes

Writing bfloat16 (variant 01) is the one that needs work: 64 lanes of 32 bits have
to become 64 contiguous 16-bit values, so the store *packs* as it writes:

    S.vsts(out_ub[col], S.vcvt(values, T.bfloat16), dist="PK_B32")

So the "cheaper" output dtype is the wider one, which is the opposite of the
bandwidth intuition. `NORM_B32` is a plain contiguous store; `PK_B32` is a
narrowing one. The output buffer's size changes with it.

Run:  python puzzles/asc/quant/answer/cast_back/02_fp32_out.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[5]))

import torch
import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S

from common import oracle, sim, status
from common.check import assert_fp32_ulps
from common.consts import CANONICAL_G

VARIANT = "asc/cast_back/02_fp32_out"
LANES = 64
SF_PAD = 64


@tilelang.jit(target="ascend", out_idx=[2])
def compile_kernel(hidden: int, group_size: int = CANONICAL_G):
    """Dequantize FP8 -> float32."""
    assert hidden % 128 == 0 and group_size == 32
    num_groups = hidden // group_size
    num_tokens = T.dynamic("num_tokens")

    @T.prim_func
    def cast_back(
        Q: T.Tensor((num_tokens, hidden), T.float8_e4m3fn),
        Sf: T.Tensor((num_tokens, num_groups), T.float32),
        Out: T.Tensor((num_tokens, hidden), T.float32),
    ):
        with T.Kernel(1):
            q_ub = T.alloc_shared((hidden,), T.float8_e4m3fn)
            sf_ub = T.alloc_shared((SF_PAD,), T.float32)
            out_ub = T.alloc_shared((hidden,), T.float32)
            for token in T.serial(num_tokens):
                T.copy(Q[token, 0], q_ub)
                T.copy(Sf[token, 0], sf_ub[0:num_groups])
                # TODO: same as variant 01, except the store is
                #       S.vsts(out_ub[col], scaled, dist='NORM_B32') with no vcvt
                #       -- the lanes are already float32
                raise NotImplementedError("asc/cast_back/02_fp32_out: implement cast_back")
                T.copy(out_ub, Out[token, 0])

    return cast_back


def launch(q: torch.Tensor, sf: torch.Tensor) -> torch.Tensor:
    out = compile_kernel(q.shape[1])(q, sf)
    status.assert_on_device("cast_back 02", out)
    return out


def demo_numbers() -> None:
    print("[demo] store widths for one 256-byte register:")
    print("[demo]   float32  out: 64 lanes x 4 B = 256 B  -> NORM_B32 (plain store)")
    print("[demo]   bfloat16 out: 64 lanes x 2 B = 128 B  -> PK_B32   (narrowing store)")
    print("[demo] the wider output dtype needs the simpler instruction")


def test_correctness() -> None:
    m, k = sim.sim_shapes()
    torch.manual_seed(0)
    q = (torch.randn(m, k) * 100).to(torch.float8_e4m3fn)
    sf = torch.rand(m, k // CANONICAL_G) * 0.01 + 1e-4
    ref = oracle.cast_back(q, sf, (1, CANONICAL_G), out_dtype=torch.float32)
    got = launch(q.npu(), sf.npu()).cpu()
    assert_fp32_ulps(got, ref, f"cast_back_f32({m},{k})", max_ulps=0)
    print(f"[check] shape=({m},{k}) bit-exact vs the torch oracle")


def main() -> int:
    m, k = sim.sim_shapes()
    if status.unimplemented(VARIANT, lambda: compile_kernel(k)):
        return 0
    sim.maybe_reexec()
    sim.print_banner("asc", "cast_back", "02_fp32_out")

    def body():
        demo_numbers()
        test_correctness()

    return status.run_variant(VARIANT, body)


if __name__ == "__main__":
    sys.exit(main())
