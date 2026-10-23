"""Probe what a ``T.SimtVF`` region can and cannot lower.

Compile-only by default (fast, no simulator). With --run it also launches the
bodies that compiled and checks what they actually computed, which is how the
two *silent* failures below were found -- they compile and return wrong numbers.

    python harness/probe/simt_limits.py          # compile-only, no simulator
    python harness/probe/simt_limits.py --run    # + numerical check, needs the
                                                 # simulator or a device

Every claim in the SIMT section of ``doc/known-issues.md`` comes from this file.
See README.md in this directory for the last recorded results.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import torch
import torch_npu  # noqa: F401
import tilelang
import tilelang.ascend.language as T

THREADS = 128
K = 128          # channels
G = 32           # quant group
NG = K // G      # groups
ROWS, COLS = 4, 32


# --------------------------------------------------------------------------
# bodies under test
# --------------------------------------------------------------------------

def body_fp4_convert():
    """float32 -> e2m1 via T.copy into an FP4 buffer, inside a SimtVF region.

    Expected to FAIL: the SIMD tiers reach a packed-FP4 convert through
    V.vcvt / S.vcvt, and nothing in the SIMT surface does.
    """
    @T.prim_func
    def kern(X: T.Tensor((K,), T.float32), Q: T.Tensor((K,), T.float4_e2m1fn)):
        with T.Kernel(1):
            x_ub = T.alloc_shared((K,), T.float32)
            q_ub = T.alloc_shared((K,), T.float4_e2m1fn)
            T.copy(X, x_ub)
            with T.SimtVF(threads=THREADS):
                v = T.alloc_fragment((K,), T.float32)
                for i in T.Parallel(K):
                    v[i] = x_ub[i]
                T.copy(v, q_ub)          # float32 -> e2m1: no such path
            T.copy(q_ub, Q)
    return kern


def body_bf16_reduce():
    """A cross-thread reduction whose accumulator is bfloat16."""

    @T.prim_func
    def kern(X: T.Tensor((K,), T.bfloat16), O: T.Tensor((NG,), T.bfloat16)):
        with T.Kernel(1):
            x_ub = T.alloc_shared((K,), T.bfloat16)
            o_ub = T.alloc_shared((NG,), T.bfloat16)
            T.copy(X, x_ub)
            with T.SimtVF(threads=THREADS):
                acc = T.alloc_fragment((NG,), T.bfloat16)
                for g in T.Parallel(NG):
                    acc[g] = T.bfloat16(0)
                    for j in T.serial(G):
                        acc[g] = T.max(acc[g], T.abs(x_ub[g * G + j]))
                T.copy(acc, o_ub)
            T.copy(o_ub, O)
    return kern


def body_bf16_reduce_no_abs():
    """The same bfloat16 reduction with T.abs removed, to isolate the failure."""
    @T.prim_func
    def kern(X: T.Tensor((K,), T.bfloat16), O: T.Tensor((NG,), T.bfloat16)):
        with T.Kernel(1):
            x_ub = T.alloc_shared((K,), T.bfloat16)
            o_ub = T.alloc_shared((NG,), T.bfloat16)
            T.copy(X, x_ub)
            with T.SimtVF(threads=THREADS):
                acc = T.alloc_fragment((NG,), T.bfloat16)
                for g in T.Parallel(NG):
                    acc[g] = T.bfloat16(0)
                    for j in T.serial(G):
                        acc[g] = T.max(acc[g], x_ub[g * G + j])
                T.copy(acc, o_ub)
            T.copy(o_ub, O)
    return kern


def body_f32_abs():
    """T.abs on float32, for contrast: the float32 path is the one we use."""
    @T.prim_func
    def kern(X: T.Tensor((K,), T.float32), O: T.Tensor((K,), T.float32)):
        with T.Kernel(1):
            x_ub = T.alloc_shared((K,), T.float32)
            o_ub = T.alloc_shared((K,), T.float32)
            T.copy(X, x_ub)
            with T.SimtVF(threads=THREADS):
                v = T.alloc_fragment((K,), T.float32)
                for i in T.Parallel(K):
                    v[i] = T.abs(x_ub[i])
                T.copy(v, o_ub)
            T.copy(o_ub, O)
    return kern


def body_reduce_max_2d():
    """T.reduce_max(dim=1) on a (num_groups, group_size) fragment, unannotated.

    This case exists to KEEP A CLAIM HONEST. An earlier draft of the SIMT tier
    asserted that this shape needs an explicit ``T.annotate_layout`` and writes
    only lane 0 without one. The probe does not reproduce that: it compiles and
    returns the right answer. The real cause of the wrong numbers that prompted
    the claim was ``body_copy_2d_region`` below, which was present in the same
    kernel at the time. The claim was retracted; this case is the regression
    test for the retraction.
    """

    @T.prim_func
    def kern(X: T.Tensor((K,), T.float32), O: T.Tensor((NG,), T.float32)):
        with T.Kernel(1):
            x_ub = T.alloc_shared((K,), T.float32)
            o_ub = T.alloc_shared((NG,), T.float32)
            T.copy(X, x_ub)
            with T.SimtVF(threads=THREADS):
                v = T.alloc_fragment((NG, G), T.float32)
                amax = T.alloc_fragment((NG,), T.float32)
                for g, j in T.Parallel(NG, G):
                    v[g, j] = T.abs(x_ub[g * G + j])
                T.reduce_max(v, amax, dim=1)
                T.copy(amax, o_ub)
            T.copy(o_ub, O)
    return kern


def body_serial_scan():
    """The workaround: one thread per group, inner T.serial. Should be right."""

    @T.prim_func
    def kern(X: T.Tensor((K,), T.float32), O: T.Tensor((NG,), T.float32)):
        with T.Kernel(1):
            x_ub = T.alloc_shared((K,), T.float32)
            o_ub = T.alloc_shared((NG,), T.float32)
            T.copy(X, x_ub)
            with T.SimtVF(threads=THREADS):
                amax = T.alloc_fragment((NG,), T.float32)
                for g in T.Parallel(NG):
                    amax[g] = T.float32(0)
                    for j in T.serial(G):
                        amax[g] = T.max(amax[g], T.abs(x_ub[g * G + j]))
                T.copy(amax, o_ub)
            T.copy(o_ub, O)
    return kern


def body_two_patterns_fragment():
    """Two different index expressions on ONE fragment in one Parallel body.

    The documented constraint (src/op/parallel.cc, RecordBufferAccess).
    """
    @T.prim_func
    def kern(X: T.Tensor((K,), T.float32), O: T.Tensor((K // 2,), T.float32)):
        with T.Kernel(1):
            x_ub = T.alloc_shared((K,), T.float32)
            o_ub = T.alloc_shared((K // 2,), T.float32)
            T.copy(X, x_ub)
            with T.SimtVF(threads=THREADS):
                v = T.alloc_fragment((K,), T.float32)
                o = T.alloc_fragment((K // 2,), T.float32)
                for i in T.Parallel(K):
                    v[i] = x_ub[i]
                for p in T.Parallel(K // 2):
                    o[p] = v[2 * p] + v[2 * p + 1]      # two patterns on `v`
                T.copy(o, o_ub)
            T.copy(o_ub, O)
    return kern


def body_two_patterns_shared():
    """The same thing on a SHARED buffer, which is exempt. Should compile."""
    @T.prim_func
    def kern(X: T.Tensor((K,), T.float32), O: T.Tensor((K // 2,), T.float32)):
        with T.Kernel(1):
            x_ub = T.alloc_shared((K,), T.float32)
            o_ub = T.alloc_shared((K // 2,), T.float32)
            T.copy(X, x_ub)
            with T.SimtVF(threads=THREADS):
                o = T.alloc_fragment((K // 2,), T.float32)
                for p in T.Parallel(K // 2):
                    o[p] = x_ub[2 * p] + x_ub[2 * p + 1]
                T.copy(o, o_ub)
            T.copy(o_ub, O)
    return kern


def body_copy_2d_region():
    """`T.copy(X[token, 0], ub)` into a 2-D UB buffer.

    Compiles. Copies the 2-D REGION X[token:token+rows, 0:cols], not the token's
    K contiguous elements -- a SILENT wrong answer. --run shows which.
    """
    @T.prim_func
    def kern(X: T.Tensor((8, K), T.float32), O: T.Tensor((ROWS, COLS), T.float32)):
        with T.Kernel(1):
            ub = T.alloc_shared((ROWS, COLS), T.float32)
            T.copy(X[1, 0], ub)             # intent: row 1's first rows*cols values
            T.copy(ub, O)
    return kern


# name, builder, how to check the numbers when --run is given.
# A `check` of None means "compile-only": the body is expected not to compile,
# or there is nothing numeric to say about it.
def _group_amax(x):
    return x.abs().view(NG, G).amax(dim=-1)


def _pairs(x):
    return x[0::2] + x[1::2]


CASES = [
    ("fp4 convert inside SimtVF", body_fp4_convert, None),
    ("bfloat16 reduce with T.abs", body_bf16_reduce, ("bf16", NG, _group_amax)),
    ("bfloat16 reduce without T.abs", body_bf16_reduce_no_abs,
     ("bf16", NG, lambda x: x.view(NG, G).amax(dim=-1))),
    ("float32 T.abs, for contrast", body_f32_abs, ("f32", K, lambda x: x.abs())),
    # expected to PASS -- see the docstring: this is a retracted claim's test
    ("reduce_max(dim=1), unannotated", body_reduce_max_2d, ("f32", NG, _group_amax)),
    ("one thread per group + serial scan", body_serial_scan,
     ("f32", NG, _group_amax)),
    ("two index patterns on a fragment", body_two_patterns_fragment, None),
    ("two index patterns on shared", body_two_patterns_shared,
     ("f32", K // 2, _pairs)),
    ("T.copy(X[token, 0], 2-D ub)", body_copy_2d_region, ("copy2d", 0, None)),
]


# --------------------------------------------------------------------------
# checks
# --------------------------------------------------------------------------

def check(name: str, kern, spec, run: bool) -> None:
    try:
        jit = tilelang.compile(kern, target="ascend")
    except Exception as exc:                                  # noqa: BLE001
        first = str(exc).strip().splitlines()[0][:150]
        print(f"[check] COMPILE-FAIL  {name}")
        print(f"[check]               {type(exc).__name__}: {first}")
        return
    print(f"[check] compiles      {name}")
    if not run or spec is None:
        return

    kind, n_out, ref_fn = spec
    torch.manual_seed(0)
    try:
        if kind == "copy2d":
            # Numbered input, so the result says which elements arrived.
            x = torch.arange(8 * K, dtype=torch.float32).reshape(8, K)
            od = torch.empty((ROWS, COLS), dtype=torch.float32).npu()
            jit(x.npu(), od)
            got = od.cpu()
            as_row = torch.equal(got, x[1, :ROWS * COLS].reshape(ROWS, COLS))
            as_region = torch.equal(got, x[1:1 + ROWS, 0:COLS])
            print(f"[check]               row 1's first {ROWS * COLS} values? {as_row}")
            print(f"[check]               the 2-D region X[1:{1 + ROWS}, 0:{COLS}]? "
                  f"{as_region}")
            print(f"[check]               got[0][:4] = {got[0][:4].tolist()}")
            return

        dt = torch.bfloat16 if kind == "bf16" else torch.float32
        x = torch.randn(K, dtype=torch.float32)
        if kind == "bf16":
            x = x.to(torch.bfloat16)
        ref = ref_fn(x.float() if kind == "bf16" else x)
        od = torch.empty((n_out,), dtype=dt).npu()
        jit(x.npu(), od)
        got = od.cpu().float()
        ok = torch.allclose(got, ref.float(), rtol=1e-2, atol=1e-2)
        print(f"[check]               numerically correct: {ok}")
        if not ok:
            print(f"[check]               got  {[round(v, 4) for v in got.tolist()[:8]]}")
            print(f"[check]               want {[round(v, 4) for v in ref.float().tolist()[:8]]}")
    except Exception as exc:                                  # noqa: BLE001
        print(f"[check]               RUN-FAIL {type(exc).__name__}: "
              f"{str(exc).strip().splitlines()[0][:130]}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="store_true",
                    help="also launch what compiled and check the numbers")
    args = ap.parse_args()
    if args.run:
        # The numerical half has to execute on a device; with none visible, hand
        # off to the CPU simulator exactly as every variant file does.
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
        from harness import sim
        if sim.should_reexec():
            raise SystemExit(sim.reexec_argv([str(pathlib.Path(__file__).resolve()),
                                              "--run"]))
    print("probing T.SimtVF limits "
          f"({'compile + run' if args.run else 'compile only'})\n")
    for name, body, spec in CASES:
        check(name, body(), spec, args.run)
    print("\nResults are recorded in harness/probe/README.md. Two of these compile")
    print("and return wrong numbers, which is the whole reason --run exists.")
    # A probe reports what it found; finding a limit is a successful probe, so the
    # verdict sim.py looks for is unconditional.
    print("PASS probe finished (read the lines above; a COMPILE-FAIL here is data)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
