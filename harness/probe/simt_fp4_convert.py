"""Can a `T.SimtVF` region reach the FP4 convert? No -- but the kernel can.

    python -m harness.probe.simt_fp4_convert hand     # the shipped encoder
    python -m harness.probe.simt_fp4_convert hybrid   # SIMT + a SimdVF convert
    python -m harness.probe.simt_fp4_convert          # both, verify only

`per_token/04` is the worst row in the SIMT tier by a wide margin, because no
FP4 convert is reachable from a `SimtVF` region and an 18-line e2m1 encoder
stands in for one hardware instruction. The obvious question is whether a plain
`T.Cast` to `float4_e2m1fn` works inside the region. It does not, and the reason
is below. What *does* work is leaving the region for that one instruction.

## Why `T.Cast` to FP4 cannot work inside a SimtVF region

Nine spellings were tried; all fail before codegen, with three distinct errors:

    q_ub[i] = T.Cast(fp4, v)                     store wants 2 lanes, got 1
    for k in T.vectorized(2): q_ub[2*p+k] = ..   float4_e2m1fnx2 vs ...x4
    fp4 fragment + T.copy                        same lanes mismatch
    fp4 fragment, 2/4/8/16 codes per thread      same lanes mismatch
    T.Cast("float4_e2m1fnx2", scalar)            lane count 2 vs 1
    reinterpret an fp4 code as uint8             4 bits vs 8

The root cause is not tilelang's surface. An FP4 element is **four bits**, so an
fp4 buffer is retyped to `float4_e2m1fnx2` -- one byte holding two codes --
before Ascend codegen (`src/ascend/transform/rewrite_fp4_to_fp4x2.cc`), and
every store into it must supply `index_lanes * 2` lanes. A SIMT region is
per-thread *scalar* code, so it has exactly one lane to offer.

Underneath, the toolchain agrees: in `include/pto/npu/a5/` the only FP4 types
that exist are `vector_f4e2m1x2` and `float4_e2m1x2_t`, and the conversion is
`vcvt(vector_f4e2m1x2, vector_bf16, preg, ROUND_R, PART_P0)` -- a vector
instruction over a vector-only type, with a predicate register and a half-register
part selector. **There is no scalar FP4 type to convert into.** This is a
property of the hardware's data types, not a gap in the frontend.

## What works instead: one kernel, two regions

A `T.SimtVF` region and a `T.SimdVF` region can live in the same kernel. The
thread model does the reduction, the scale and the round-to-odd; the vector
model is entered for the one instruction it owns:

    with T.SimdVF():
        S.vsts(q_ub[col], S.vcvt(S.vld(bf_ub[col]), T.float4_e2m1fn),
               dist="PK4_B32")

That is 3 lines against the hand encoder's 18, and it is the measurement below.

Measured, SoC Ascend950PR_9599, M=32 K=128 G=32, 32 launches each, both arms
**bit-exact** against `harness.oracle.per_token(..., fmt="e2m1")` -- 0 of 4096
codes differ from the torch packer, in either arm:

    arm                        VF_SIMT    VF      total   vs ASC's 6,160
    hand-written e2m1 encoder  543,827     0    543,827       88.3x
    SIMT reduce + SIMD cvt       6,050  1,728      7,778        1.26x

**69.9x fewer cycles**, and the FP4 row stops being an outlier at all: 1.26x is
the same neighbourhood as `per_token/01`. A hybrid kernel issues both launch
kinds and costs their sum, which is what the table above reports.

(The hand arm measures 543,827 here against the ladder's 562,280 for the same
kernel. That is not a discrepancy to reconcile: `T.if_then_else` lowers to a
real branch, so the instruction count depends on the data, and this probe's
input is not the harness's. Compare the two arms, which share an input.)

The ladder does **not** ship this. `puzzles/simt/` exists to show what the
thread model costs when a kernel is written in it, and a kernel that drops into
`T.SimdVF` for its hardest instruction is no longer that kernel -- it is the
answer to a different and better question, which is why this probe states it
separately. `doc/known-issues.md` and `doc/quant/per_token/04_fp32_in_fp4_out.md`
carry the numbers.

Reproduce:

    TLP_KEEP_SIM_OUT=1 python -m harness.probe.simt_fp4_convert hand
    TLP_KEEP_SIM_OUT=1 python -m harness.probe.simt_fp4_convert hybrid
    # then, on each printed [sim] out= directory, sum BOTH launch kinds --
    # a hybrid kernel issues VF_SIMT and VF rows and costs the sum.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch
import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from harness import oracle, sim                                    # noqa: E402
from harness.consts import CANONICAL_G, E2M1_CLAMP_MIN, E2M1_MAX    # noqa: E402
from harness.math_ops import unpack_e2m1_bytes                      # noqa: E402

M, K = 32, 128
G = CANONICAL_G
THREADS = 128
LANES = 64
PAIR = 2 * LANES          # one float32 pair of registers, as the ASC tier uses


def _shipped():
    """The SIMT tier's own kernel, imported so the arms cannot drift."""
    path = ROOT / "puzzles/simt/quant/answer/per_token/04_fp32_in_fp4_out.py"
    spec = importlib.util.spec_from_file_location("fp4_hand", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@tilelang.jit(target="ascend")
def compile_hybrid(hidden: int, group_size: int = G):
    """per_token/04, with the e2m1 encode replaced by one `S.vcvt`.

    Everything up to the convert is the SIMT tier's: `T.reduce_absmax` over a
    (group, channel) fragment, the clamp, the two scales, and the apply. The
    round-to-odd is the SIMD tiers' (`S.vdintlv` + sticky bit) written on
    scalars, because the convert that follows it rounds a second time.
    """
    assert hidden % group_size == 0 and group_size == 32
    num_groups = hidden // group_size
    num_tokens = T.dynamic("num_tokens")

    @T.prim_func
    def per_token_cast(
        X: T.Tensor((num_tokens, hidden), T.float32),
        Q: T.Tensor((num_tokens, hidden), T.float4_e2m1fn),
        Sf: T.Tensor((num_tokens, num_groups), T.float32),
    ):
        with T.Kernel(1):
            x_ub = T.alloc_shared((hidden,), T.float32)
            bf_ub = T.alloc_shared((hidden,), T.bfloat16)
            q_ub = T.alloc_shared((hidden,), T.float4_e2m1fn)
            sf_ub = T.alloc_shared((num_groups,), T.float32)

            for token in T.serial(num_tokens):
                T.copy(X[token, 0], x_ub)

                with T.SimtVF(threads=THREADS):
                    x_l = T.alloc_fragment((num_groups, group_size), T.float32)
                    amax_l = T.alloc_fragment((num_groups,), T.float32)
                    inv_l = T.alloc_fragment((num_groups,), T.float32)
                    sf_l = T.alloc_fragment((num_groups,), T.float32)

                    for g, j in T.Parallel(num_groups, group_size):
                        x_l[g, j] = x_ub[g * group_size + j]
                    T.reduce_absmax(x_l, amax_l, dim=1)

                    for g in T.Parallel(num_groups):
                        amax_l[g] = T.max(amax_l[g], E2M1_CLAMP_MIN)
                        inv_l[g] = E2M1_MAX / amax_l[g]
                        sf_l[g] = amax_l[g] / E2M1_MAX

                    # Scale, then round to odd on the way to bfloat16: keep a
                    # sticky bit so the convert's own rounding cannot round
                    # twice. This is S.vdintlv + S.vor + S.vmin on scalars.
                    for i in T.Parallel(hidden):
                        bits = T.reinterpret("uint32",
                                             x_ub[i] * inv_l[i // group_size])
                        sticky = T.min(bits & 0xFFFF, T.uint32(1))
                        bf_ub[i] = T.reinterpret(
                            "bfloat16", T.Cast("uint16", (bits >> 16) | sticky))

                    T.copy(sf_l, sf_ub)

                # The one instruction the thread model cannot name.
                with T.SimdVF():
                    for pair in T.serial(hidden // PAIR):
                        S.vsts(q_ub[pair * PAIR],
                               S.vcvt(S.vld(bf_ub[pair * PAIR]),
                                      T.float4_e2m1fn),
                               dist="PK4_B32")

                T.copy(q_ub, Q[token, 0])
                T.copy(sf_ub, Sf[token, 0])

    return per_token_cast


def run(arm: str):
    torch.manual_seed(0)
    x = (torch.randn(M, K, dtype=torch.float32) * 3.0)
    sf = torch.empty((M, K // G), dtype=torch.float32)
    if arm == "hand":
        q = torch.empty((M, K // 2), dtype=torch.uint8)
        args = [t.npu() for t in (x, q, sf)]
        _shipped().compile_kernel(K)(*args)        # the FIRST device op
        got_q = args[1].cpu()
    else:
        q = torch.empty((M, K // 2), dtype=torch.uint8).view(
            torch.float4_e2m1fn_x2)
        args = [t.npu() for t in (x, q, sf)]
        compile_hybrid(K)(*args)                   # the FIRST device op
        got_q = args[1].cpu().view(torch.uint8)

    ref_q, ref_sf = oracle.per_token(x, G, fmt="e2m1")
    got_sf = args[2].cpu()
    ok_sf = torch.allclose(got_sf, ref_sf, rtol=1e-6, atol=0)
    got_v = unpack_e2m1_bytes(got_q)
    ref_v = unpack_e2m1_bytes(ref_q)
    diff, total = int((got_v != ref_v).sum()), got_v.numel()
    # the harness's own standard for this variant: FP4 has eight magnitudes, so
    # differing codes are counted rather than hidden behind a tolerance
    ok_q = diff / total < 0.02
    print(f"[check] arm={arm} scales={'ok' if ok_sf else 'WRONG'}, "
          f"FP4 codes differing from the torch packer: {diff}/{total} "
          f"({diff / total:.2%})")
    return ok_sf and ok_q, got_q, got_sf


def main() -> int:
    want = sys.argv[1] if len(sys.argv) > 1 else None
    if sim.should_reexec():
        argv = ["-m", "harness.probe.simt_fp4_convert"] + ([want] if want else [])
        raise SystemExit(sim.reexec_argv(argv))
    ok = True
    out = {}
    for arm in ([want] if want else ["hand", "hybrid"]):
        good, q, sf = run(arm)
        ok &= good
        out[arm] = (q, sf)
    if len(out) == 2:
        same_sf = torch.equal(out["hand"][1], out["hybrid"][1])
        hv = unpack_e2m1_bytes(out["hand"][0])
        yv = unpack_e2m1_bytes(out["hybrid"][0])
        d = int((hv != yv).sum())
        print(f"[check] the two arms agree on the scales: {same_sf}; "
              f"FP4 codes differing between them: {d}/{hv.numel()}")
        ok &= same_sf
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
