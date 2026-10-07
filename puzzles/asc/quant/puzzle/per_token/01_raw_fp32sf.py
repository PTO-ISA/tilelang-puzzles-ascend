"""per_token 01 (ASC) -- quantize to FP8, with a reduction inside the vector unit.

Read puzzles/torch/quant/answer/per_token/01_raw_fp32sf.py for the maths, and the
cast_back ASC variants for the memory model. The new thing here is the
**reduction**, and it restructures the whole kernel.

    amax = max(|x|) over each group of 32 channels
    sf   = max(amax, 1e-4) / 448
    q    = cast_e4m3(x * (448 / amax))

### Three passes, two barriers

cast_back was one pass: load, scale, store. Quantizing cannot be, because no
output value can be written until the group's maximum is known. So:

    pass 1   read x, reduce |x| per group, write amax to UB
    ---- S.mem_bar("VST_VLD") ----
    pass 2   read amax, compute the scale and its inverse, write both to UB
    ---- S.mem_bar("VST_VLD") ----
    pass 3   read x again, multiply by the inverse, convert, store

Those barriers are not optional and not inserted for you. Within a vector scope
the hardware does **not** track whether a vector load aliases an earlier vector
store to the same UB address, so without `mem_bar` pass 2 may read amax values
that pass 1 has not yet written. This is the single most common source of silent
wrong answers in these kernels.

(`"VST_VLD"` means "all prior vector stores are visible to subsequent vector
loads". There are also `"VLD_VST"` and `"VV_ALL"`.)

### Reducing 32 of 64 lanes: masks again

`S.vcmax` reduces across a register and produces one value. But a register holds
64 float32 lanes and a group is 32 channels, so each register contains **two**
groups. The reduction has to be restricted to half the lanes at a time:

    mask_low  = S.pset(32, "PAT_VL32")          # lanes 0-31
    mask_all  = S.pset(32, "PAT_ALL")
    mask_high = S.pnot(mask_low, mask_all)      # lanes 32-63
    S.vsts(amax_ub[group],     S.vcmax(abs_x, mask_low),  dist="ONEPT_B32")
    S.vsts(amax_ub[group + 1], S.vcmax(abs_x, mask_high), dist="ONEPT_B32")

Two masked reductions and two single-element stores (`ONEPT_B32` writes one lane's
worth) per register. The PTO version replaces all four operations with one
`vcmax(..., group=4)`, which is the clearest demonstration of VMI's segmented
operations in the repo.

### The scale pass is 64 scales at a time

Pass 2 is a nice detail: the amax values were written as individual elements, but
they are now *contiguous*, so one 64-lane load picks up 64 of them and the clamp,
divide and reciprocal all happen 64 groups at a time. Writing the reduction's
output back to UB converts a cross-lane problem into an element-wise one.

### GPU vs NPU

On a GPU this kernel is `T.reduce_absmax(fragment, out, dim=1)` plus two
`T.Parallel` loops; the compiler decides whether the reduction is intra-thread,
intra-warp or through shared memory, and inserts whatever synchronisation that
needs. Here you choose the reduction's lane extent with a predicate, place its
result in UB yourself, and write the barriers. Same three lines of maths.

### Simulation

Expect roughly 25-40 s at M=32, K=128 -- more than cast_back, because there are
three passes over the data instead of one.

Run:  python puzzles/asc/quant/answer/per_token/01_raw_fp32sf.py
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

VARIANT = "asc/per_token/01_raw_fp32sf"
LANES = 64
PAIR = 128          # two 64-lane strips: the smallest unit holding 4 groups
SF_PAD = 64


# No out_idx: tilelang's automatic output allocation rejects float8_e4m3fn with
# "MemoryError: Unsupported code 10" on this toolchain, so launch() allocates the
# outputs and passes them in. See common/probe/fp8_out_idx.py, which reproduces
# the failure and will start reporting YES if a later tilelang fixes it. This is
# also how the production kernels are called.
@tilelang.jit(target="ascend")
def compile_kernel(hidden: int, group_size: int = CANONICAL_G):
    """Quantize bfloat16 -> FP8 e4m3 with one FP32 scale per 32 channels."""
    assert hidden % PAIR == 0, "this kernel steps 128 channels (4 groups) at a time"
    assert group_size == 32
    num_groups = hidden // group_size
    assert num_groups <= SF_PAD, "one register must hold all of a token's scales"
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
            amax_ub = T.alloc_shared((SF_PAD,), T.float32)
            sf_ub = T.alloc_shared((SF_PAD,), T.float32)
            inv_ub = T.alloc_shared((SF_PAD,), T.float32)

            for token in T.serial(num_tokens):
                T.copy(X[token, 0], x_ub)
                # TODO: three passes. (1) per 128-channel pair, load two 64-lane
                #       strips with S.vld(x_ub[col], dist='UNPK_B16') + S.vcvt to
                #       float32, S.vabs, then for each strip store S.vcmax(abs,
                #       mask_low) and S.vcmax(abs, mask_high) to amax_ub with
                #       dist='ONEPT_B32'. (2) S.mem_bar('VST_VLD'); load 64 amax
                #       values at once, S.vmaxs by E4M3_CLAMP_MIN, then sf =
                #       S.vdiv(clamped, 448) and inv = S.vdiv(448, clamped), store
                #       both. (3) S.mem_bar('VST_VLD'); reload x, build the
                #       inverse vector from four BRC_B32 loads plus two S.vsel,
                #       multiply, S.vcvt to float8_e4m3fn and store with
                #       dist='PK4_B32'
                raise NotImplementedError("asc/per_token/01_raw_fp32sf: implement per_token_cast")
                T.copy(q_ub, Q[token, 0])
                T.copy(sf_ub[0:num_groups], Sf[token, 0])

    return per_token_cast


def launch(x: torch.Tensor):
    m, hidden = x.shape
    num_groups = hidden // CANONICAL_G
    q = torch.empty((m, hidden), dtype=torch.float8_e4m3fn, device=x.device)
    sf = torch.empty((m, num_groups), dtype=torch.float32, device=x.device)
    compile_kernel(hidden)(x, q, sf)
    status.assert_on_device("per_token 01", q, sf)
    return q, sf


def demo_numbers() -> None:
    print("[demo] why three passes and two barriers:")
    print("[demo]   pass 1  x -> amax per group        (writes amax_ub)")
    print("[demo]   barrier VST_VLD")
    print("[demo]   pass 2  amax -> scale, 1/scale     (reads amax_ub)")
    print("[demo]   barrier VST_VLD")
    print("[demo]   pass 3  x * (1/scale) -> FP8       (reads inv_ub)")
    print("[demo] no output value exists until its group's maximum is known")
    print("[demo] reduction arithmetic for one register (64 float32 lanes):")
    print(f"[demo]   64 lanes / {CANONICAL_G} channels per group = 2 groups")
    print("[demo]   so 2 masked vcmax + 2 ONEPT stores per register")


def test_correctness() -> None:
    m, k = sim.sim_shapes()
    torch.manual_seed(0)
    x = randn_with_zero_row(m, k, torch.device("cpu"))   # row 0 exercises the clamp
    ref_q, ref_sf = oracle.per_token(x, CANONICAL_G)
    q, sf = launch(x.npu())
    assert_fp32_ulps(sf.cpu(), ref_sf, f"sf({m},{k})", max_ulps=1)
    assert_fp8_near(q.cpu(), ref_q, f"q({m},{k})")
    print(f"[check] shape=({m},{k}) sf={tuple(sf.shape)} matches the torch oracle")
    assert not q.cpu().float().isnan().any(), "the zero row must not produce NaN"
    print("[check] the all-zero row clamped correctly (no NaN)")


def main() -> int:
    m, k = sim.sim_shapes()
    if status.unimplemented(VARIANT, lambda: compile_kernel(k)):
        return 0
    sim.maybe_reexec()
    sim.print_banner("asc", "per_token", "01_raw_fp32sf")

    def body():
        demo_numbers()
        test_correctness()

    return status.run_variant(VARIANT, body)


if __name__ == "__main__":
    sys.exit(main())
