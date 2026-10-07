"""cast_back 02 (PTO) -- float32 output instead of bfloat16.

Only the store changes.

### PTO vs ASC

This variant is where VMI's handling of width stops being cosmetic. ASC picks the
store *instruction* by output dtype -- `NORM_B32` for a plain 32-bit store,
`PK_B32` for a narrowing one -- so the kernel text differs between the two output
types:

    ASC, bfloat16 out:  S.vsts(out_ub[col], S.vcvt(v, T.bfloat16), dist="PK_B32")
    ASC, float32  out:  S.vsts(out_ub[col], v,                     dist="NORM_B32")

VMI infers it from the destination buffer's dtype, so the store is spelled the
same either way and only the convert appears or disappears:

    PTO, bfloat16 out:  V.vstore(V.vcvt(v, "bfloat16"), out_ub[col])
    PTO, float32  out:  V.vstore(v,                     out_ub[col])

The useful consequence is that a kernel parameterised over its output dtype needs
no branch in PTO, while in ASC the distribution mode has to be selected -- which
is exactly what production's `store_out` helper does.

The hardware fact underneath is unchanged and worth knowing either way: a 256-byte
register is 64 float32 lanes, so a float32 store is the direct case and a
bfloat16 store is the one that has to narrow and pack. The wider output dtype
needs the simpler instruction.

Run:  python puzzles/pto/quant/answer/cast_back/02_fp32_out.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[5]))

import torch
import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import vmi as V

from common import oracle, sim, status
from common.check import assert_fp32_ulps
from common.consts import CANONICAL_G

VARIANT = "pto/cast_back/02_fp32_out"
LANES = 64
SF_PAD = 64


@tilelang.jit(target="pto", out_idx=[2])
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
                # --- BEGIN SOLUTION hint="same as variant 01, except V.vstore(scaled, out_ub[col]) with no vcvt -- the destination buffer is float32 so VMI stores 32-bit lanes directly"
                with T.SimdVF():
                    mask = V.create_mask(LANES, size=LANES)
                    groups_per_strip = LANES // group_size
                    for strip in T.serial(hidden // LANES):
                        col = strip * LANES
                        group = strip * groups_per_strip
                        values = V.vcvt(V.vload(q_ub[col], size=LANES), "float32")
                        scale = V.vload(sf_ub[group], size=LANES, stride=1,
                                        dist_mode="brc", group=groups_per_strip)
                        scaled = V.vmul(values, scale, mask)
                        # Identical spelling to variant 01's store; only the
                        # convert is gone, because out_ub is already float32.
                        V.vstore(scaled, out_ub[col])
                # --- END SOLUTION
                T.copy(out_ub, Out[token, 0])

    return cast_back


def launch(q: torch.Tensor, sf: torch.Tensor) -> torch.Tensor:
    out = compile_kernel(q.shape[1])(q, sf)
    status.assert_on_device("cast_back 02", out)
    return out


def demo_numbers() -> None:
    print("[demo] store widths for one 256-byte register:")
    print("[demo]   float32  out: 64 lanes x 4 B = 256 B")
    print("[demo]   bfloat16 out: 64 lanes x 2 B = 128 B (must narrow and pack)")
    print("[demo] ASC picks the instruction (NORM_B32 vs PK_B32); VMI infers it")
    print("[demo] from the destination buffer dtype, so the store text is the same")


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
    sim.print_banner("pto", "cast_back", "02_fp32_out")

    def body():
        demo_numbers()
        test_correctness()

    return status.run_variant(VARIANT, body)


if __name__ == "__main__":
    sys.exit(main())
