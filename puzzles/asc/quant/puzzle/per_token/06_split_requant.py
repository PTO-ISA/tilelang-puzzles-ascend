"""per_token 06 (ASC) -- sf_only, cast_only, and requantization.

Three related configs, all about *not* doing the whole job. Production compiles
one kernel with these as compile-time flags, and this file does the same: `mode`
selects which passes get emitted.

    mode="full"       amax -> scale -> quantize          (variants 01-05)
    mode="sf_only"    amax -> scale, no quantized output
    mode="cast_only"  scales are given; apply them, no amax pass
    mode="requant"    input is already quantized; dequantize, then quantize again

### Why flags rather than four kernels

Because the passes are the same code. `sf_only` is the full kernel with pass 3
deleted; `cast_only` is the full kernel with pass 1 deleted and pass 2 reduced to a
reciprocal. Expressing that as `if` statements in the kernel builder -- evaluated
at trace time, so they cost nothing at runtime -- is how production keeps one
source for a dozen configurations.

This is also the first variant where the Python-level `if` is doing real work. Note
it is a *trace-time* branch: `mode` is a Python string, so the condition is
evaluated while the kernel is being built and only the taken branch is emitted.
(Contrast `T.unroll`, where the loop variable is symbolic -- see the gotcha
documented in cast_back/07.)

### cast_only cannot reproduce the fused path exactly

With `cast_only` the kernel only has the stored scale, so it must compute `1/sf`
and multiply. The full path forms `448/amax` directly. Those differ in the last
bit or two, which flips the occasional FP8 code -- the torch variant measures it:
about 75 codes in 131072 for bfloat16 input, and **zero** when the scale is a power
of two, because then both the scale and its reciprocal are exact.

That is a practical argument for `round_sf` beyond memory: it makes the split
kernels bit-compatible with the fused one.

### requant needs a scratch buffer

The new amax cannot be known until the whole group has been dequantized, so the
dequantized values have to live somewhere between the two stages. That is
`dequant_ub`, with a barrier on each side of it -- the same shape as production's
`in_config.with_sf` path.

Run:  python puzzles/asc/quant/answer/per_token/06_split_requant.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[5]))

import torch
import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S

from common import oracle, sim, status
from common.check import assert_fp8_near, assert_fp32_ulps
from common.consts import CANONICAL_G, E4M3_CLAMP_MIN, E4M3_MAX
from common.demo import randn_with_zero_row

VARIANT = "asc/per_token/06_split_requant"
LANES = 64
PAIR = 128
SF_PAD = 64


@tilelang.jit(target="ascend")
def compile_kernel(hidden: int, mode: str = "full", group_size: int = CANONICAL_G):
    """mode in {"full", "sf_only", "cast_only", "requant"}."""
    assert hidden % PAIR == 0 and group_size == 32
    assert mode in ("full", "sf_only", "cast_only", "requant")
    num_groups = hidden // group_size
    num_tokens = T.dynamic("num_tokens")
    groups_per_pair = PAIR // group_size

    # Trace-time choices: only the selected branches are ever emitted.
    need_amax = mode in ("full", "sf_only", "requant")
    need_quant = mode in ("full", "cast_only", "requant")
    sf_is_input = mode == "cast_only"
    x_dtype = T.float8_e4m3fn if mode == "requant" else T.bfloat16

    @T.prim_func
    def per_token_cast(
        X: T.Tensor((num_tokens, hidden), x_dtype),
        XSf: T.Tensor((num_tokens, num_groups), T.float32),
        Q: T.Tensor((num_tokens, hidden), T.float8_e4m3fn),
        Sf: T.Tensor((num_tokens, num_groups), T.float32),
    ):
        with T.Kernel(1):
            x_ub = T.alloc_shared((hidden,), x_dtype)
            q_ub = T.alloc_shared((hidden,), T.float8_e4m3fn)
            val_ub = T.alloc_shared((hidden,), T.float32)   # dequantized scratch
            amax_ub = T.alloc_shared((SF_PAD,), T.float32)
            sf_ub = T.alloc_shared((SF_PAD,), T.float32)
            inv_ub = T.alloc_shared((SF_PAD,), T.float32)
            xsf_ub = T.alloc_shared((SF_PAD,), T.float32)

            for token in T.serial(num_tokens):
                T.copy(X[token, 0], x_ub)
                if mode in ("cast_only", "requant"):
                    T.copy(XSf[token, 0], xsf_ub[0:num_groups])
                # TODO: gate the three passes on the mode. requant first
                #       dequantizes x_ub into val_ub using the input scales
                #       (broadcast + vsel as in cast_back/01) with a barrier
                #       after. sf_only emits passes 1-2 only. cast_only skips pass
                #       1 and sets inv = S.vdiv(1.0, given sf). Everything else is
                #       variants 01/02.
                raise NotImplementedError("asc/per_token/06_split_requant: implement per_token_cast")
                if need_quant:
                    T.copy(q_ub, Q[token, 0])
                if need_amax:
                    T.copy(sf_ub[0:num_groups], Sf[token, 0])

    return per_token_cast


def launch(x: torch.Tensor, mode: str = "full", x_sf: torch.Tensor | None = None):
    m, hidden = x.shape
    ng = hidden // CANONICAL_G
    dev = x.device
    xsf = x_sf if x_sf is not None else torch.zeros((m, ng), dtype=torch.float32, device=dev)
    q = torch.empty((m, hidden), dtype=torch.float8_e4m3fn, device=dev)
    sf = torch.empty((m, ng), dtype=torch.float32, device=dev)
    compile_kernel(hidden, mode)(x, xsf, q, sf)
    status.assert_on_device(f"per_token 06 {mode}", q, sf)
    return q, sf


def demo_numbers() -> None:
    print("[demo] which passes each mode emits (decided at trace time):")
    rows = [("full", "amax", "scale", "quantize"),
            ("sf_only", "amax", "scale", "-"),
            ("cast_only", "-", "1/sf", "quantize"),
            ("requant", "dequant+amax", "scale", "quantize")]
    for name, p1, p2, p3 in rows:
        print(f"[demo]   {name:10} pass1={p1:13} pass2={p2:6} pass3={p3}")
    print("[demo] cast_only must compute 1/sf because it never sees amax, so it")
    print("[demo] diverges from the fused path by the odd FP8 code -- unless the")
    print("[demo] scale is a power of two, when both are exact. See the torch")
    print("[demo] variant, which measures it.")


def test_correctness() -> None:
    m, k = sim.sim_shapes()
    torch.manual_seed(0)
    x = randn_with_zero_row(m, k, torch.device("cpu"))
    ref_q, ref_sf = oracle.per_token(x, CANONICAL_G)

    _, sf = launch(x.npu(), "sf_only")
    assert_fp32_ulps(sf.cpu(), ref_sf, "sf_only", max_ulps=1)
    print("[check] sf_only matches the oracle's scales")

    q, _ = launch(x.npu(), "cast_only", x_sf=ref_sf.npu())
    ref_co = oracle.per_token_cast_only(x, ref_sf, CANONICAL_G)
    assert_fp8_near(q.cpu(), ref_co, "cast_only")
    d = (q.cpu().view(torch.uint8).int() - ref_q.view(torch.uint8).int()).abs()
    print(f"[check] cast_only matches the oracle; vs the fused path "
          f"{int((d > 0).sum())}/{d.numel()} codes differ (the 1/sf reciprocal)")

    q2, sf2 = launch(ref_q.npu(), "requant", x_sf=ref_sf.npu())
    rq, rsf = oracle.requant_per_token(ref_q, ref_sf, CANONICAL_G)
    assert_fp32_ulps(sf2.cpu(), rsf, "requant sf", max_ulps=1)
    assert_fp8_near(q2.cpu(), rq, "requant q")
    print("[check] requant matches dequantize-then-quantize in torch")


def main() -> int:
    m, k = sim.sim_shapes()
    if status.unimplemented(VARIANT, lambda: compile_kernel(k, "full")):
        return 0
    sim.maybe_reexec()
    sim.print_banner("asc", "per_token", "06_split_requant")

    def body():
        demo_numbers()
        test_correctness()

    return status.run_variant(VARIANT, body)


if __name__ == "__main__":
    sys.exit(main())
