"""Does `T.if_then_else` lower to a select, or to a divergent branch?

    python harness/probe/simt_divergence.py            # both arms, ~16 min
    python harness/probe/simt_divergence.py --mode sel # one arm

It lowers to a **branch**, and on `simt/per_token/04` that costs 1.75x.

This probe exists because that variant measured 99.6x ASC's simulated VF
cycles, far outside the rest of the SIMT tier (3.2x-21.2x), and "the hand-written
FP4 encoder is just expensive" turned out to be only part of the answer. It is
the only variant in the ladder with a large number of `SIMT_START_DVG` /
`SIMT_END_DVG` markers in its instruction trace, so the hypothesis was testable:
rewrite the three conditionals inside the encoder with mask arithmetic, change
nothing else, and see what moves.

Both arms are bit-exact against the torch packer, so this is a like-for-like
comparison of two spellings of one algorithm.

Measured 2026-10-12, tilelang 0.1.15 + CANN 9.2.0-beta.2, Ascend950PR_9599,
M=32 K=128:

| spelling | cycles | instructions | cycles/instr | divergence | SIMT_SEL |
|---|---:|---:|---:|---:|---:|
| `T.if_then_else` (shipped) | 615,170 | 124,294 | 4.95 | 5,864 | 1,764 |
| mask arithmetic | **351,956** | 179,908 | **1.96** | **0** | 8,192 |

The branch-free version issues **45% more instructions and runs 1.75x fewer
cycles**. Its cycles-per-instruction of 1.96 is exactly what every other SIMT
`per_token` variant achieves (1.90-2.01); the shipped spelling's 4.95 is the
divergence penalty and nothing else.

Worth being clear about what this is and is not. On a GPU, `cond ? a : b` over
cheap arithmetic is predicated into a select by the compiler, so a GPU
programmer writing the natural conditional does not pay for divergence. Here it
is a real branch. That is a **codegen quality gap, not a property of the thread
model**, and it is why the ladder's headline SIMT figures quote the range with
`per_token/04` set aside.

The shipped kernel keeps `T.if_then_else` deliberately: the brief asked for the
spelling a GPU programmer reaches for first, not the tuned one. The faster
spelling is below, and `doc/quant/per_token/04_fp32_in_fp4_out.md` says so.
"""

from __future__ import annotations

import argparse
import collections
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch                                                   # noqa: E402
import tilelang                                                # noqa: E402
import tilelang.ascend.language as T                           # noqa: E402

from harness import sim                                        # noqa: E402
from harness.consts import CANONICAL_G, E2M1_CLAMP_MIN, E2M1_MAX   # noqa: E402
from harness.math_ops import pack_e2m1_from_fp32               # noqa: E402

THREADS, M, K, G = 128, 32, 128, CANONICAL_G
NG = K // G
OUT_RE = re.compile(r"^\[sim\] out=(.+)$", re.M)
LINE = re.compile(r"\) (RVEC\w+|VEC|PUSHQ) *:.*?\(ID: \d+\) ([A-Za-z0-9_]+)")


def nibble(v, mode: str):
    """float32 -> e2m1 nibble. `mode` changes only how the conditionals are spelled."""
    bits = T.reinterpret("int32", v)
    sign_nib = (bits >> 28) & 0x8
    exps = (bits >> 23) & 0xFF
    mant = bits & 0x7FFFFF
    sh = T.min(126 - exps, 31)
    mant_pre = 0x400000 | (mant >> 1)
    low = (1 << sh) - 1

    if mode == "sel":
        is_sub = exps < 127
        sticky_sub = T.if_then_else(is_sub, (mant & 1) | (mant_pre & low), 0)
        mant_eff = T.if_then_else(is_sub, mant_pre >> sh, mant)
    else:
        # -(exps < 127) is 0 or all-ones. Both arms are always safe to evaluate
        # because `sh` is already clamped to 31.
        m = -T.Cast("int32", exps < 127)
        sticky_sub = ((mant & 1) | (mant_pre & low)) & m
        mant_eff = ((mant_pre >> sh) & m) | (mant & ~m)

    exp_eff = T.max(exps, 126) - 126
    m2 = (mant_eff >> 21) & 3
    guard, lsb = m2 & 1, (m2 >> 1) & 1
    sticky = sticky_sub | (mant_eff & 0x1FFFFF)
    # `sticky` is a non-negative bit field, so min(.,1) is its "is non-zero".
    nz = T.if_then_else(sticky != 0, 1, 0) if mode == "sel" else T.min(sticky, 1)
    tmp = (((exp_eff << 2) | m2) + (guard & (nz | lsb))) >> 1
    return sign_nib | T.min(tmp, 7)


@tilelang.jit(target="ascend")
def build(mode: str):
    @T.prim_func
    def kern(X: T.Tensor((M, K), T.float32), Q: T.Tensor((M, K // 2), T.uint8),
             Sf: T.Tensor((M, NG), T.float32)):
        with T.Kernel(1):
            x_ub = T.alloc_shared((K,), T.float32)
            q_ub = T.alloc_shared((K // 2,), T.uint8)
            sf_ub = T.alloc_shared((NG,), T.float32)
            for token in T.serial(M):
                T.copy(X[token, 0], x_ub)
                with T.SimtVF(threads=THREADS):
                    inv_l = T.alloc_fragment((NG,), T.float32)
                    sf_l = T.alloc_fragment((NG,), T.float32)
                    for g in T.Parallel(NG):
                        amax = T.alloc_var(T.float32)
                        amax = E2M1_CLAMP_MIN
                        for j in T.serial(G):
                            amax = T.max(amax, T.abs(x_ub[g * G + j]))
                        inv_l[g] = E2M1_MAX / amax
                        sf_l[g] = amax / E2M1_MAX
                    for p in T.Parallel(K // 2):
                        inv = inv_l[p // (G // 2)]
                        lo = nibble(x_ub[2 * p] * inv, mode)
                        hi = nibble(x_ub[2 * p + 1] * inv, mode)
                        q_ub[p] = T.Cast("uint8", lo | (hi << 4))
                    T.copy(sf_l, sf_ub)
                T.copy(q_ub, Q[token, 0])
                T.copy(sf_ub, Sf[token, 0])
    return kern


def run_one(mode: str) -> None:
    """The inner process: build, launch, and check against the torch packer."""
    torch.manual_seed(0)
    x = torch.randn(M, K, dtype=torch.float32) * 3.0
    x[0] = 0                                   # exercises the clamp and zero path
    grouped = x.view(M, NG, G)
    amax = grouped.abs().amax(-1).clamp(min=E2M1_CLAMP_MIN)
    quant = (grouped * (E2M1_MAX / amax).unsqueeze(-1)).reshape(M, K).contiguous()
    ref = pack_e2m1_from_fp32(quant).view(torch.uint8)

    q = torch.empty((M, K // 2), dtype=torch.uint8).npu()
    sf = torch.empty((M, NG), dtype=torch.float32).npu()
    build(mode)(x.npu(), q, sf)
    diff = int((q.cpu() != ref).sum())
    print(f"[check] mode={mode}: {diff}/{ref.numel()} bytes differ from the "
          f"torch packer")
    print(f"{'PASS' if diff == 0 else 'FAIL'} mode={mode}")


def measure(mode: str) -> dict | None:
    """Outer process: run one arm under the simulator and read its trace."""
    proc = subprocess.run(
        [sys.executable, "-B", str(Path(__file__).resolve()), "--mode", mode],
        cwd=str(ROOT), text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env={**__import__("os").environ, "TLP_KEEP_SIM_OUT": "1"})
    ok = "PASS mode=" in proc.stdout
    om = OUT_RE.search(proc.stdout)
    if not om:
        print(f"  {mode}: no [sim] out= line; run with --mode {mode} to see why")
        return None
    sys.path.insert(0, str(ROOT / "tools"))
    from vf_cycles import sum_vf_cycles
    rec = sum_vf_cycles(Path(om.group(1).strip()))
    hist: collections.Counter = collections.Counter()
    dump = Path(rec["csv"]).parents[2] / "dump" / "core0.veccore0.instr_log.dump"
    if dump.exists():
        for line in dump.open(errors="replace"):
            m = LINE.search(line)
            if m:
                hist[m.group(2)] += 1
    total = sum(hist.values())
    return {
        "mode": mode, "bit_exact": ok, "cycles": rec["cycles"], "instrs": total,
        "per_instr": (rec["cycles"] / total) if total else 0,
        "dvg": hist.get("SIMT_START_DVG", 0) + hist.get("SIMT_END_DVG", 0),
        "sel": hist.get("SIMT_SEL", 0), "branch": hist.get("SIMT_BRANCH", 0),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("sel", "mask"),
                    help="run just this arm (used internally, and for debugging)")
    args = ap.parse_args()

    if args.mode:
        if sim.should_reexec():
            raise SystemExit(sim.reexec_argv(
                [str(Path(__file__).resolve()), "--mode", args.mode]))
        run_one(args.mode)
        return 0

    print("comparing two spellings of the same e2m1 encoder "
          "(each arm needs its own simulator run)\n")
    rows = [r for r in (measure("sel"), measure("mask")) if r]
    if len(rows) != 2:
        return 1
    hdr = (f"  {'spelling':26} {'cycles':>9} {'instrs':>8} {'cyc/ins':>8} "
           f"{'DVG':>6} {'SEL':>6} {'BRANCH':>7}  bit-exact")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    label = {"sel": "T.if_then_else (shipped)", "mask": "mask arithmetic"}
    for r in rows:
        print(f"  {label[r['mode']]:26} {r['cycles']:>9,.0f} {r['instrs']:>8} "
              f"{r['per_instr']:>8.2f} {r['dvg']:>6} {r['sel']:>6} "
              f"{r['branch']:>7}  {r['bit_exact']}")
    a, b = rows[0], rows[1]
    print()
    print(f"  mask arithmetic issues {b['instrs'] / a['instrs']:.2f}x the "
          f"instructions and takes {b['cycles'] / a['cycles']:.2f}x the cycles")
    print(f"  -- i.e. {a['cycles'] / b['cycles']:.2f}x FASTER with more work, "
          f"because divergence goes {a['dvg']} -> {b['dvg']}.")
    if not (a["bit_exact"] and b["bit_exact"]):
        print("\n  WARNING: an arm was not bit-exact; the comparison is void.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
