"""Does `T.reduce_max` beat the explicit serial scan in a SimtVF region?

    python -m harness.probe.simt_reduce_max scan        # one spelling
    python -m harness.probe.simt_reduce_max reduce
    python -m harness.probe.simt_reduce_max             # both, verify only

Every SIMT variant in `puzzles/simt/quant/` finds its group maximum with one
thread per group and an inner `T.serial`:

    for g in T.Parallel(num_groups):
        amax[g] = E4M3_CLAMP_MIN
        for j in T.serial(group_size):
            amax[g] = T.max(amax[g], T.abs(x_ub[g * group_size + j]))

tilelang also has `T.reduce_max`, which is shorter and looks like the natural
thing to reach for. `harness/probe/simt_limits.py` already established that it
compiles and returns the right answer on this shape; what was never measured is
what it costs. This probe runs both spellings of `per_token/01`'s reduction as
the first device op of their process, so `msprof op simulator` profiles the
kernel itself, and `tools/vf_cycles.py` sums the VF_SIMT rows of each trace.

Both arms compute the same thing and are checked against the same torch oracle
before any cycle number is reported.

Measured, SoC Ascend950PR_9599, M=32 K=128 G=32, one VF launch per token in
both arms (`call_count=32`), both within the harness's FP8 tolerance and
byte-identical to each other:

    spelling            VF_SIMT cycles   vs ASC's 6,176   issued abs/reduce ops
    serial scan             71,606           11.59x               4,096
    T.reduce_max             8,012            1.30x                 256

**The reduction spelling is worth 8.94x on this variant**, and the shipped
`puzzles/simt/quant/answer/per_token/01_raw_fp32sf.py` uses the slower one.
The mechanism is thread occupancy, not the library call: `T.Parallel(num_groups)`
with an inner `T.serial(group_size)` keeps 4 of 128 threads busy and issues the
inner body 32 times over, while `T.Parallel(num_groups, group_size)` plus
`T.reduce_max` has every thread take one element and leaves the fold to the
backend. `tools/vf_trace_audit.py` on the two traces shows the difference
directly: 4,096 reduce/abs instructions against 256.

Reproduce:

    TLP_KEEP_SIM_OUT=1 python -m harness.probe.simt_reduce_max scan
    TLP_KEEP_SIM_OUT=1 python -m harness.probe.simt_reduce_max reduce
    # then, on each printed [sim] out= directory:
    python -c "import sys; sys.path.insert(0,'tools'); from pathlib import Path; \
               from vf_cycles import sum_vf_cycles; print(sum_vf_cycles(Path(d)))"
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import tilelang
import tilelang.ascend.language as T

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from harness import sim                                        # noqa: E402
from harness import oracle                                      # noqa: E402
from harness.asserts import assert_fp8_near                     # noqa: E402
from harness.consts import CANONICAL_G, E4M3_CLAMP_MIN, E4M3_MAX   # noqa: E402

M, K = 32, 128
G = CANONICAL_G
THREADS = 128


@tilelang.jit(target="ascend")
def compile_kernel(hidden: int, spelling: str, group_size: int = G):
    """per_token/01's algorithm, with the reduction written one of two ways."""
    assert spelling in ("scan", "reduce")
    num_groups = hidden // group_size
    num_tokens = T.dynamic("num_tokens")

    @T.prim_func
    def per_token_cast(
        X: T.Tensor((num_tokens, hidden), T.bfloat16),
        Q: T.Tensor((num_tokens, hidden), T.float8_e4m3fn),
        Sf: T.Tensor((num_tokens, num_groups), T.float32),
    ):
        with T.Kernel(1):
            x_ub = T.alloc_shared((hidden,), T.bfloat16)
            q_ub = T.alloc_shared((hidden,), T.float8_e4m3fn)
            sf_ub = T.alloc_shared((num_groups,), T.float32)

            for token in T.serial(num_tokens):
                T.copy(X[token, 0], x_ub)
                with T.SimtVF(threads=THREADS):
                    q_l = T.alloc_fragment((hidden,), T.float32)
                    amax_l = T.alloc_fragment((num_groups,), T.float32)
                    inv_l = T.alloc_fragment((num_groups,), T.float32)
                    sf_l = T.alloc_fragment((num_groups,), T.float32)

                    if spelling == "scan":
                        # the shipped tier: one thread per group, seeded with
                        # the clamp so the clamp costs nothing
                        for g in T.Parallel(num_groups):
                            amax_l[g] = E4M3_CLAMP_MIN
                            for j in T.serial(group_size):
                                v = T.Cast("float32", x_ub[g * group_size + j])
                                amax_l[g] = T.max(amax_l[g], T.abs(v))
                            inv_l[g] = E4M3_MAX / amax_l[g]
                            sf_l[g] = amax_l[g] / E4M3_MAX
                    else:
                        # T.reduce_max over a (num_groups, group_size) fragment.
                        # The clamp cannot be folded into a seed here, so it is
                        # a separate max afterwards.
                        a_l = T.alloc_fragment((num_groups, group_size), T.float32)
                        for g, j in T.Parallel(num_groups, group_size):
                            a_l[g, j] = T.abs(
                                T.Cast("float32", x_ub[g * group_size + j]))
                        T.reduce_max(a_l, amax_l, dim=1)
                        for g in T.Parallel(num_groups):
                            amax_l[g] = T.max(amax_l[g], E4M3_CLAMP_MIN)
                            inv_l[g] = E4M3_MAX / amax_l[g]
                            sf_l[g] = amax_l[g] / E4M3_MAX

                    for i in T.Parallel(hidden):
                        q_l[i] = T.Cast("float32", x_ub[i]) * inv_l[i // group_size]

                    T.copy(q_l, q_ub)
                    T.copy(sf_l, sf_ub)

                T.copy(q_ub, Q[token, 0])
                T.copy(sf_ub, Sf[token, 0])

    return per_token_cast


def run(spelling: str):
    torch.manual_seed(0)
    x = torch.randn(M, K, dtype=torch.float32).to(torch.bfloat16)
    q = torch.empty((M, K), dtype=torch.float8_e4m3fn)
    sf = torch.empty((M, K // G), dtype=torch.float32)
    args = [t.npu() for t in (x, q, sf)]
    compile_kernel(K, spelling)(*args)          # the FIRST device op
    gq, gsf = oracle.per_token(x, G)
    got_q, got_sf = args[1].cpu(), args[2].cpu()
    ok_sf = torch.allclose(got_sf, gsf, rtol=1e-6, atol=0)
    bad = int((got_q.view(torch.uint8) != gq.view(torch.uint8)).sum())
    # the harness's own standard for FP8 codes -- a handful of boundary ties
    # may flip under the simulator's rounding; see harness/asserts.py
    assert_fp8_near(got_q, gq, f"q({M},{K}) {spelling}")
    print(f"[check] spelling={spelling} scales={'ok' if ok_sf else 'WRONG'}, "
          f"codes within the harness tolerance ({bad}/{M * K} differ by a tie)")
    return ok_sf, got_q, got_sf


def main() -> int:
    want = sys.argv[1] if len(sys.argv) > 1 else None
    if sim.should_reexec():
        argv = ["-m", "harness.probe.simt_reduce_max"] + ([want] if want else [])
        raise SystemExit(sim.reexec_argv(argv))
    ok = True
    out = {}
    for spelling in ([want] if want else ["scan", "reduce"]):
        good, q, sf = run(spelling)
        ok &= good
        out[spelling] = (q, sf)
    if len(out) == 2:
        same = (torch.equal(out["scan"][0].view(torch.uint8),
                            out["reduce"][0].view(torch.uint8))
                and torch.equal(out["scan"][1], out["reduce"][1]))
        print(f"[check] the two spellings agree byte for byte: {same}")
        ok &= same
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
