"""per_token 06 (PTO) -- sf_only, cast_only and requant in VMI.

Read the ASC variant for what the three modes are and why they are compile-time
flags rather than separate kernels.

### PTO vs ASC: the win compounds across configurations

This variant is where the structural advantage from variant 02 pays off visibly.
The kernel has four modes, and each mode needs the broadcast-and-apply step. In
ASC that step is four `BRC_B32` loads plus a `vsel` per 64-lane register, written
out in the dequantize stage *and* the quantize stage:

    ASC, twice over:
        s0 = S.vld(xsf_ub[g],     dist="BRC_B32")
        s1 = S.vld(xsf_ub[g + 1], dist="BRC_B32")
        ... S.vmul(raw, S.vsel(s0, s1, mask_low))

In VMI each is one load, so the duplication costs almost nothing:

    PTO, twice over:
        scale = V.vload(xsf_ub[g], size=128, stride=1, dist_mode="brc", group=4)
        ... V.vmul(raw, scale, mask)

This is the mechanism behind the production port's 83 removed lines. A single
teaching variant shows a small difference; a kernel that branches over round_sf,
packing, FP4, column-major, requant and the split spells the same broadcast
machinery once per path, and VMI shrinks every one of them.

### requant exercises both directions at once

The dequantize stage broadcasts the *input* scales and the quantize stage
broadcasts the *output* inverses, so one mode uses the same VMI idiom twice with
different data. In ASC those are two near-identical six-operation blocks; in VMI
they are two one-line loads.

Run:  python puzzles/pto/quant/answer/per_token/06_split_requant.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[5]))

import torch
import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import vmi as V

from common import oracle, sim, status
from common.check import assert_fp8_near, assert_fp32_ulps
from common.consts import CANONICAL_G, E4M3_CLAMP_MIN, E4M3_MAX
from common.demo import randn_with_zero_row

VARIANT = "pto/per_token/06_split_requant"
LANES = 64
PAIR = 128
SF_PAD = 64


@tilelang.jit(target="pto")
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
                # --- BEGIN SOLUTION hint="gate the passes on the mode. requant first dequantizes x_ub into val_ub using one brc load of the input scales, with a barrier after. sf_only emits passes 1-2 only. cast_only skips pass 1 and sets inv = V.vdiv(1.0, given sf). Everything else is variant 01."
                with T.SimdVF():
                    mask = V.create_mask(PAIR, size=PAIR)
                    m64 = V.create_mask(LANES, size=LANES)
                    qmax = V.vbrc(T.float32(E4M3_MAX), size=LANES)
                    clamp_v = V.vbrc(T.float32(E4M3_CLAMP_MIN), size=LANES)

                    if mode == "requant":
                        # stage 0: dequantize into the scratch buffer. One brc
                        # load per 128 channels, where ASC needs 4 loads + 2 vsel.
                        for pair in T.serial(hidden // PAIR):
                            col = pair * PAIR
                            group = pair * groups_per_pair
                            scale = V.vload(xsf_ub[group], size=PAIR, stride=1,
                                            dist_mode="brc", group=groups_per_pair)
                            raw = V.vcvt(V.vload(x_ub[col], size=PAIR), "float32")
                            V.vstore(V.vmul(raw, scale, mask), val_ub[col])
                        T.simd.mem_bar("VST_VLD")

                    if need_amax:
                        for pair in T.serial(hidden // PAIR):
                            col = pair * PAIR
                            group = pair * groups_per_pair
                            if mode == "requant":
                                xv = V.vload(val_ub[col], size=PAIR)
                            else:
                                xv = V.vcvt(V.vload(x_ub[col], size=PAIR), "float32")
                            V.vstore(V.vcmax(V.vabs(xv, mask), mask,
                                             group=groups_per_pair), amax_ub[group])
                        T.simd.mem_bar("VST_VLD")

                        clamped = V.vmax(V.vload(amax_ub[0], size=LANES), clamp_v, m64)
                        V.vstore(V.vdiv(clamped, qmax, m64), sf_ub[0])
                        V.vstore(V.vdiv(qmax, clamped, m64), inv_ub[0])
                    elif sf_is_input:
                        # cast_only: invert the given scale. This reciprocal is why
                        # cast_only can differ from the fused path by a code.
                        one = V.vbrc(T.float32(1.0), size=LANES)
                        V.vstore(V.vdiv(one, V.vload(xsf_ub[0], size=LANES), m64),
                                 inv_ub[0])
                    T.simd.mem_bar("VST_VLD")

                    if need_quant:
                        for pair in T.serial(hidden // PAIR):
                            col = pair * PAIR
                            group = pair * groups_per_pair
                            inv = V.vload(inv_ub[group], size=PAIR, stride=1,
                                          dist_mode="brc", group=groups_per_pair)
                            if mode == "requant":
                                xv = V.vload(val_ub[col], size=PAIR)
                            else:
                                xv = V.vcvt(V.vload(x_ub[col], size=PAIR), "float32")
                            V.vstore(V.vcvt(V.vmul(xv, inv, mask), "float8_e4m3fn",
                                            rounding="R", saturate="SAT"), q_ub[col])
                # --- END SOLUTION
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
    print("[demo] every mode needs the broadcast-and-apply step; requant needs it")
    print("[demo] twice (input scales, then output inverses). In VMI each is one")
    print("[demo] brc load; in ASC each is 4 loads + a vsel per register. That is")
    print("[demo] how the saving compounds across a multi-config kernel.")
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
    sim.print_banner("pto", "per_token", "06_split_requant")

    def body():
        demo_numbers()
        test_correctness()

    return status.run_variant(VARIANT, body)


if __name__ == "__main__":
    sys.exit(main())
