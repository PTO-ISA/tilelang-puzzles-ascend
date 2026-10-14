"""Does the declared SIMT thread count change cycles on a small-group reduction?

    python harness/probe/simt_thread_count.py              # 32 / 64 / 128, ~12 min
    python harness/probe/simt_thread_count.py --threads 32 # one arm

Every SIMT kernel in this repo declares `THREADS = 128`. On `per_token/01` at
`K=128, group_size=32` the passes inside the `T.SimtVF` region want different
widths:

    for g, j in T.Parallel(num_groups, group_size):  # extent 128 -- the fold
    for g in T.Parallel(num_groups):                 # extent 4   -- the scales
    for i in T.Parallel(hidden):                     # extent 128 -- the apply

So one short pass remains, and the question is whether declaring fewer threads
helps it. This probe measures it: same kernel, same results, only
`T.SimtVF(threads=...)` changes.

It is measured against the **shipped** spelling, which folds with
`T.reduce_absmax`. The sweep was first run against the serial scan this tier
used to ship -- one thread per group, extent 4 -- and found 32/64/128 threads
within 4% of each other (71,278 / 69,047 / 71,606 cycles). That was the right
answer to the wrong question: the way to recover those threads was to widen the
`T.Parallel`, not to narrow the region. See
`harness/probe/simt_reduce_max.py`, which measures the 8.73x.

`T.SimtVF` accepts any positive thread count (tilelang 0.1.15,
`threads: int | list | tuple = 128`), so lowering it is legal; whether it is
faster is the question.

Correctness is checked against `harness/oracle.py` on every arm, so a thread
count that compiles but computes the wrong answer cannot be reported as fast.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch                                                       # noqa: E402
import tilelang                                                    # noqa: E402
import tilelang.ascend.language as T                               # noqa: E402

from harness import oracle, sim                                    # noqa: E402
from harness.asserts import assert_fp8_near, assert_fp32_ulps      # noqa: E402
from harness.consts import CANONICAL_G, E4M3_CLAMP_MIN, E4M3_MAX   # noqa: E402
from harness.demo import randn_with_zero_row                       # noqa: E402

M, K = 32, 128
OUT_RE = re.compile(r"^\[sim\] out=(.+)$", re.M)
ARMS = (32, 64, 128)


@tilelang.jit(target="ascend")
def build(hidden: int, threads: int, group_size: int = CANONICAL_G):
    """puzzles/simt/quant/answer/per_token/01_raw_fp32sf.py, with `threads` free."""
    assert hidden % group_size == 0 and group_size == 32
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
                with T.SimtVF(threads=threads):
                    x_l = T.alloc_fragment((num_groups, group_size), T.float32)
                    q_l = T.alloc_fragment((hidden,), T.float32)
                    amax_l = T.alloc_fragment((num_groups,), T.float32)
                    inv_l = T.alloc_fragment((num_groups,), T.float32)
                    sf_l = T.alloc_fragment((num_groups,), T.float32)

                    for g, j in T.Parallel(num_groups, group_size):
                        x_l[g, j] = T.Cast("float32", x_ub[g * group_size + j])
                    T.reduce_absmax(x_l, amax_l, dim=1)

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


def run_one(threads: int) -> None:
    """The inner process: build at this width and check against the oracle."""
    torch.manual_seed(0)
    x = randn_with_zero_row(M, K, torch.device("cpu"), dtype=torch.bfloat16)
    ref_q, ref_sf = oracle.per_token(x, CANONICAL_G)

    q = torch.empty((M, K), dtype=torch.float8_e4m3fn).npu()
    sf = torch.empty((M, K // CANONICAL_G), dtype=torch.float32).npu()
    build(K, threads)(x.npu(), q, sf)
    assert_fp32_ulps(sf.cpu(), ref_sf, f"sf threads={threads}", max_ulps=1)
    assert_fp8_near(q.cpu(), ref_q, f"q threads={threads}")
    print(f"[check] threads={threads}: matches the oracle")
    print(f"PASS threads={threads}")


def measure(threads: int) -> dict | None:
    proc = subprocess.run(
        [sys.executable, "-B", str(Path(__file__).resolve()),
         "--threads", str(threads)],
        cwd=str(ROOT), text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env={**os.environ, "TLP_KEEP_SIM_OUT": "1"})
    ok = f"PASS threads={threads}" in proc.stdout
    om = OUT_RE.search(proc.stdout)
    if not om:
        tail = "\n".join(proc.stdout.strip().splitlines()[-4:])
        print(f"  threads={threads}: no simulator output\n{tail}")
        return None
    sys.path.insert(0, str(ROOT / "tools"))
    from vf_cycles import sum_vf_cycles
    rec = sum_vf_cycles(Path(om.group(1).strip()))
    return {"threads": threads, "ok": ok, "cycles": rec["cycles"],
            "calls": rec["call_count"]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--threads", type=int, help="run just this width (internal)")
    args = ap.parse_args()

    if args.threads:
        if sim.should_reexec():
            raise SystemExit(sim.reexec_argv(
                [str(Path(__file__).resolve()), "--threads", str(args.threads)]))
        run_one(args.threads)
        return 0

    print(f"per_token/01 (SIMT), M={M} K={K} group_size={CANONICAL_G}: "
          f"fold extent {K}, scale extent 4, apply extent {K}\n")
    rows = [r for r in (measure(t) for t in ARMS) if r]
    if not rows:
        return 1
    base = next((r for r in rows if r["threads"] == 128), rows[-1])
    print(f"  {'threads':>8} {'cycles':>10} {'vs 128':>8} {'launches':>9}  correct")
    print("  " + "-" * 50)
    for r in rows:
        rel = r["cycles"] / base["cycles"] if base["cycles"] else float("nan")
        print(f"  {r['threads']:>8} {r['cycles']:>10,.0f} {rel:>7.2f}x "
              f"{r['calls']:>9}  {r['ok']}")
    if not all(r["ok"] for r in rows):
        print("\n  WARNING: an arm did not match the oracle; its cycles mean nothing.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
