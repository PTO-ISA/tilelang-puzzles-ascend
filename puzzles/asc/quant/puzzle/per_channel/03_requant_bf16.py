"""per_channel 03 (ASC) -- requantize per-token input to per-channel scales.

New config: `in_config.with_sf`. The input is already quantized -- FP8 values with
*per-token* scales -- and must come out quantized with *per-channel* scales. See
the torch variant for the contract.

### Why this variant is interesting: both broadcast patterns at once

The input scales are per-token: one per 32 channels, so applying them needs the
broadcast-and-select pattern from per_token/01. The output scales are per-channel:
one per lane, so applying those is a plain contiguous load, as in variant 01.

So a single kernel uses both idioms, on the same data, a few instructions apart:

    dequantize   scale varies along K  -> BRC_B32 x2 + vsel per register
    quantize     scale varies along M  -> one plain vld

Reading them side by side is the clearest way to see that "how expensive is a
scale?" is entirely a question of which axis it varies along relative to the
register's lanes.

### The scratch buffer is unavoidable

The new per-channel amax cannot be known until all 32 tokens have been
dequantized, so the dequantized values must be stored somewhere in between --
`val_ub`, a float32 buffer the size of the token group, with a barrier on each
side. That is `dequant_ub` in `per_channel_cast_asc.py`.

### The FP8 output is not bit-exact against torch, and that is correct

This is the one variant in the ladder whose quantized values do **not** match the
torch reference byte for byte. About 2% of them differ, always by exactly one FP8
code, and the test asserts both of those bounds rather than loosening the
tolerance and moving on.

The cause is specific to requantization. The input values already sit on the FP8
grid, and the new scale is derived from a maximum that is itself one of those grid
values, so the products land on **exact ties**. A measured example:

    dequantized value   4.5
    per-channel scale   6/448
    exact product       4.5 * 448 / 6  =  336.0   exactly

e4m3's spacing near there is 32, so 336 is precisely halfway between the
representable codes 320 and 352. Any tie-breaking rule has to pick one, and the
hardware and torch pick differently -- the hardware's `ROUND_R` and torch both
claim round-to-nearest-ties-to-even, but they are rounding float32 values that
have already been rounded once on the way to 336, so they do not see the same tie.

Nothing is wrong with either answer. What would be wrong is a test that reported
PASS without saying so, or one loose enough to also pass if the kernel had a real
defect -- hence the assertions on the worst difference and on every difference
being a genuine midpoint.

### Requantization only loses precision

The input has already been through one quantization, so its values are already on
the FP8 grid. Requantizing cannot recover anything; the test checks the kernel
against dequantize-then-quantize in torch rather than against the original data.

Run:  python puzzles/asc/quant/answer/per_channel/03_requant_bf16.py
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
from common.consts import BLOCK_MN, CANONICAL_G, E4M3_CLAMP_MIN, E4M3_MAX
from common.demo import randn_with_zero_row

VARIANT = "asc/per_channel/03_requant_bf16"
LANES = 64


@tilelang.jit(target="ascend")
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
                # TODO: stage 1 dequantize: per row, per 64-lane tile, the INPUT
                #       scale varies along K so it needs two S.vld(...,
                #       dist='BRC_B32') plus S.vsel(lo, hi, mask_low) as in
                #       per_token/01; multiply the unpacked FP8 and store float32
                #       into val_ub. Barrier. stage 2 is variant 01 unchanged,
                #       reading val_ub with a plain S.vld: the OUTPUT scale varies
                #       along M, one value per lane.
                raise NotImplementedError("asc/per_channel/03_requant_bf16: implement per_channel_requant")
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
    print("[demo]               -> 2 x BRC_B32 + vsel per 64-lane register")
    print("[demo]   quantize  : output scale varies along M (one per channel)")
    print("[demo]               -> one plain vld, no broadcast")
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
    sim.print_banner("asc", "per_channel", "03_requant_bf16")

    def body():
        demo_numbers()
        test_correctness()

    return status.run_variant(VARIANT, body)


if __name__ == "__main__":
    sys.exit(main())
