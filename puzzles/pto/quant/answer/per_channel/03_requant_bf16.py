"""per_channel 03 (PTO) -- requantize per-token input to per-channel scales.

Read the ASC variant, including its section on why the FP8 output is not bit-exact
against torch (requantization creates exact ties) -- the same applies here.

### PTO vs ASC: both broadcast patterns, and VMI improves one of them

The kernel applies an input scale that varies along **K** and an output scale that
varies along **M**, a few instructions apart. VMI helps with the first and not the
second, which is exactly what the earlier variants predict:

    dequantize (scale varies along K, so a broadcast is needed)
        ASC  2 x S.vld(dist="BRC_B32") + S.vsel(lo, hi, mask_low)   3 ops
        PTO  V.vload(..., dist_mode="brc", group=2)                 1 op

    quantize (scale varies along M, one value per lane already)
        ASC  S.vld(inv_ub[col])                                     1 op
        PTO  V.vload(inv_ub[col], size=LANES)                       1 op

So the saving lands entirely on the axis that needed emulating. This is the
cleanest side-by-side in the ladder of *when* `group=` pays: it is not about
reductions or broadcasts as such, it is about whether the scale's axis lines up
with the register's lanes.

Run:  python puzzles/pto/quant/answer/per_channel/03_requant_bf16.py
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
from common.consts import BLOCK_MN, CANONICAL_G, E4M3_CLAMP_MIN, E4M3_MAX
from common.demo import randn_with_zero_row

VARIANT = "pto/per_channel/03_requant_bf16"
LANES = 128


@tilelang.jit(target="pto")
def compile_kernel(hidden: int, group_tokens: int = BLOCK_MN,
                   in_group: int = CANONICAL_G):
    """Per-token-quantized input -> per-channel-quantized output."""
    assert hidden % 128 == 0 and group_tokens == 32 and in_group == 32
    num_tokens = T.dynamic("num_tokens")
    num_groups = T.ceildiv(num_tokens, group_tokens)
    num_in_groups = hidden // in_group
    num_col_tiles = hidden // LANES

    @T.prim_func
    def per_channel_requant(
        Q_in: T.Tensor((num_tokens, hidden), T.float8_e4m3fn),
        Sf_in: T.Tensor((num_tokens, num_in_groups), T.float32),
        Q: T.Tensor((num_tokens, hidden), T.float8_e4m3fn),
        Sf: T.Tensor((num_groups, hidden), T.float32),
    ):
        with T.Kernel(1):
            q_in_ub = T.alloc_shared((group_tokens, hidden), T.float8_e4m3fn)
            sf_in_ub = T.alloc_shared((group_tokens, LANES), T.float32)
            val_ub = T.alloc_shared((group_tokens, hidden), T.float32)
            q_ub = T.alloc_shared((group_tokens, hidden), T.float8_e4m3fn)
            sf_ub = T.alloc_shared((hidden,), T.float32)
            inv_ub = T.alloc_shared((hidden,), T.float32)

            for mg in T.serial(num_groups):
                T.copy(Q_in[mg * group_tokens, 0], q_in_ub)
                for row in T.serial(group_tokens):
                    T.copy(Sf_in[mg * group_tokens + row, 0],
                           sf_in_ub[row, 0:num_in_groups])
                # --- BEGIN SOLUTION hint="stage 1 dequantize: per row, per 128-lane tile, the INPUT scale varies along K so fetch it with one V.vload(..., size=128, stride=1, dist_mode='brc', group=4); multiply the converted FP8 and store float32 into val_ub. Barrier. stage 2 is variant 01 unchanged, reading val_ub with a plain V.vload: the OUTPUT scale varies along M, one value per lane."
                with T.SimdVF():
                    mask = V.create_mask(LANES, size=LANES)
                    qmax = V.vbrc(T.float32(E4M3_MAX), size=LANES)
                    groups_per_tile = LANES // in_group

                    # ---- stage 1: dequantize. Scale varies along K -> one brc
                    # load with group=, where ASC needs two loads and a select.
                    for row in T.serial(group_tokens):
                        for ct in T.serial(num_col_tiles):
                            col = ct * LANES
                            g = ct * groups_per_tile
                            scale = V.vload(sf_in_ub[row, g], size=LANES, stride=1,
                                            dist_mode="brc", group=groups_per_tile)
                            raw = V.vcvt(V.vload(q_in_ub[row, col], size=LANES),
                                         "float32")
                            V.vstore(V.vmul(raw, scale, mask), val_ub[row, col])
                    T.simd.mem_bar("VST_VLD")

                    # ---- stage 2: quantize. Scale varies along M: no broadcast.
                    for ct in T.serial(num_col_tiles):
                        col = ct * LANES
                        acc = V.alloc_local((1,), V.vreg(LANES, T.float32))
                        acc[0] = V.vbrc(T.float32(E4M3_CLAMP_MIN), size=LANES)
                        for row in T.serial(group_tokens):
                            acc[0] = V.vmax(
                                acc[0],
                                V.vabs(V.vload(val_ub[row, col], size=LANES), mask),
                                mask)
                        # precision="exact" is the 0-ULP divide, as production uses.
                        V.vstore(V.vdiv(acc[0], qmax, mask, precision="exact"),
                                 sf_ub[col])
                        V.vstore(V.vdiv(qmax, acc[0], mask, precision="exact"),
                                 inv_ub[col])
                    T.simd.mem_bar("VST_VLD")

                    for ct in T.serial(num_col_tiles):
                        col = ct * LANES
                        inv = V.vload(inv_ub[col], size=LANES)   # one plain load
                        for row in T.serial(group_tokens):
                            V.vstore(V.vcvt(
                                V.vmul(V.vload(val_ub[row, col], size=LANES), inv,
                                       mask), "float8_e4m3fn", rounding="R",
                                saturate="SAT"), q_ub[row, col])
                # --- END SOLUTION
                T.copy(q_ub, Q[mg * group_tokens, 0])
                T.copy(sf_ub, Sf[mg, 0])

    return per_channel_requant


def launch(q_in: torch.Tensor, sf_in: torch.Tensor):
    m, hidden = q_in.shape
    q = torch.empty((m, hidden), dtype=torch.float8_e4m3fn, device=q_in.device)
    sf = torch.empty((m // BLOCK_MN, hidden), dtype=torch.float32, device=q_in.device)
    compile_kernel(hidden)(q_in, sf_in, q, sf)
    status.assert_on_device("per_channel 03", q, sf)
    return q, sf


def demo_numbers() -> None:
    print("[demo] one kernel, both broadcast patterns:")
    print("[demo]   dequantize: input scale varies along K (one per 32 channels)")
    print("[demo]               ASC: 2 x BRC_B32 + vsel    PTO: one brc load")
    print("[demo]   quantize  : output scale varies along M (one per channel)")
    print("[demo]               both: one plain load, no broadcast")
    print("[demo] so group= pays on the axis that needed emulating, and nowhere")
    print("[demo] else. That is the whole rule.")
    print("[demo] the cost of a scale is entirely about which axis it varies")
    print("[demo] along relative to the register's lanes.")
    print("[demo] the dequantized values need a scratch buffer, because the new")
    print("[demo] per-channel amax is not known until all 32 tokens are done.")


def test_correctness() -> None:
    m, k = sim.sim_shapes()
    torch.manual_seed(0)
    x = randn_with_zero_row(m, k, torch.device("cpu")) * 3
    q_in, sf_in = oracle.per_token(x, CANONICAL_G, round_sf=True)

    # reference: dequantize, then an ordinary per_channel pass
    dq = oracle.cast_back(q_in, sf_in, (1, CANONICAL_G), out_dtype=torch.float32)
    ref_q, ref_sf = oracle.per_channel(dq, BLOCK_MN)

    q, sf = launch(q_in.npu(), sf_in.npu())
    assert_fp32_ulps(sf.cpu(), ref_sf, f"sf({m},{k})", max_ulps=0)
    print(f"[check] shape=({m},{k}) scales are bit-exact vs torch")

    # The FP8 values are NOT bit-exact here, and the reason is worth reporting
    # rather than tolerating silently: requantizing input that already sits on
    # the FP8 grid produces exact ties. See the docstring section.
    d = (q.cpu().view(torch.uint8).int() - ref_q.view(torch.uint8).int()).abs()
    n_diff, worst = int((d > 0).sum()), int(d.max())
    print(f"[check] FP8 codes differing from torch: {n_diff}/{d.numel()} "
          f"({n_diff / d.numel():.1%}), worst difference {worst} code")
    assert worst <= 1, (
        f"differences of more than one FP8 code ({worst}) are not tie-breaking "
        f"and indicate a real bug"
    )
    # and every difference must be a genuine tie: the exact product lands
    # exactly halfway between two representable codes.
    lo = torch.minimum(q.cpu().float(), ref_q.float())
    hi = torch.maximum(q.cpu().float(), ref_q.float())
    exact = dq / sf.cpu().repeat_interleave(BLOCK_MN, dim=0)
    mid = (lo + hi) / 2
    ties = ((d > 0) & torch.isclose(exact, mid, rtol=1e-6)).sum()
    print(f"[check] of those, {int(ties)} are exact ties "
          f"(the product is the midpoint of two FP8 codes)")
    assert int(ties) == n_diff, (
        "some differences are not ties, so this is not just tie-breaking"
    )

    b1 = oracle.cast_back(q_in, sf_in, (1, CANONICAL_G), out_dtype=torch.float32)
    b2 = oracle.cast_back(q.cpu(), sf.cpu(), (BLOCK_MN, 1), out_dtype=torch.float32)
    e1 = (b1 - x).abs().max().item() / x.abs().max().item()
    e2 = (b2 - x).abs().max().item() / x.abs().max().item()
    print(f"[check] error vs the original: after 1 quantization {e1:.1%}, "
          f"after requantization {e2:.1%}")
    assert e2 >= e1, "requantizing cannot recover precision"


def main() -> int:
    k = sim.sim_shapes()[1]
    if status.unimplemented(VARIANT, lambda: compile_kernel(k)):
        return 0
    sim.maybe_reexec()
    sim.print_banner("pto", "per_channel", "03_requant_bf16")

    def body():
        demo_numbers()
        test_correctness()

    return status.run_variant(VARIANT, body)


if __name__ == "__main__":
    sys.exit(main())
