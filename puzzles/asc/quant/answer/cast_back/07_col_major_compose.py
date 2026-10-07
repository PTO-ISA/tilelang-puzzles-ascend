"""cast_back 07 (ASC) -- column-major scales, and every config at once.

Final cast_back variant. Composed config:

    packed UE8M0 scales + column-major layout + FP4 values -> bfloat16 output

### Column-major scales turn out to be free here

The scale array is stored transposed: `sf_cm[word, token]` rather than
`sf_packed[token, word]` (see the torch variant for why the consuming GEMM wants
it that way).

For *this* kernel that costs nothing at all, and the reason is worth noticing. The
scale load is a **broadcast** -- it reads one scalar and fans it across the
register -- and a broadcast does not care about the stride between consecutive
elements, because it only ever touches one. So the transpose is absorbed entirely
by swapping two index expressions:

    row-major     S.vld(sf_ub[token, word], dist="BRC_B16")
    column-major  S.vld(sf_ub[word, token], dist="BRC_B16")

The expensive direction is the other one: *producing* column-major scales in a
quantize kernel means transposing values that are spread across lanes, which
needs an in-register gather. That is per_token/05, and it is the hardest variant
in the ladder. Consuming them is trivial; producing them is not.

### Scheduling: the scale tile

The scales for one token are now a *column*, so a whole tile of them is DMA'd in
one go and indexed per token, rather than copied per token. Tokens are processed
in blocks of 32 so the UB scale buffer has a static shape.

### Where this lands relative to production

`cast_back_asc.py` adds, beyond this file: multiple vector cores with a persistent
loop, double-buffered UB, a UB-aliasing trick where the packed and decoded scale
buffers are the same allocation, and two further scale-decode strategies
(`use_e2b_scale` for a one-load-fans-256-lanes fast path, and a separate route for
per-channel scales packed along M). None of them changes the arithmetic this file
performs.

Run:  python puzzles/asc/quant/answer/cast_back/07_col_major_compose.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[5]))

import torch
import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S

from common import oracle, sim, status
from common.check import assert_bf16_near
from common.consts import BLOCK_MN, CANONICAL_G, PACK_FACTOR

VARIANT = "asc/cast_back/07_col_major_compose"
LANES = 64
FP4_STRIP = 128
EXP_MASK = 0x7F800000


@tilelang.jit(target="ascend", out_idx=[2])
def compile_kernel(hidden: int, group_size: int = CANONICAL_G,
                   token_block: int = BLOCK_MN):
    """Packed UE8M0 + column-major scales + FP4 values -> bfloat16."""
    assert hidden % FP4_STRIP == 0 and group_size == 32
    num_groups = hidden // group_size
    num_words = num_groups // PACK_FACTOR
    num_tokens = T.dynamic("num_tokens")
    num_blocks = T.ceildiv(num_tokens, token_block)

    @T.prim_func
    def cast_back(
        Q: T.Tensor((num_tokens, hidden), T.float4_e2m1fn),
        SfCm: T.Tensor((num_words, num_tokens), T.uint16),
        Out: T.Tensor((num_tokens, hidden), T.bfloat16),
    ):
        with T.Kernel(1):
            q_ub = T.alloc_shared((hidden,), T.float4_e2m1fn)
            # The scale tile stays transposed in UB, exactly as it is in GM.
            sf_ub = T.alloc_shared((num_words, token_block), T.uint16)
            out_ub = T.alloc_shared((hidden,), T.bfloat16)
            for blk in T.serial(num_blocks):
                T.copy(SfCm[0, blk * token_block], sf_ub)
                for row in T.serial(token_block):
                    token = blk * token_block + row
                    T.copy(Q[token, 0], q_ub)
                    # --- BEGIN SOLUTION hint="combine variants 03 and 06: per 128-value FP4 strip, vcvt to bfloat16 and S.vintlv(zero, x) to get two float32x64 halves; for each half broadcast its packed word with S.vld(sf_ub[word, row], dist='BRC_B16') -- note the transposed index -- then shift/mask as in variant 03 to build the scale; multiply and store both halves"
                    with T.SimdVF():
                        mask_low = S.pset(32, "PAT_VL32")
                        sf_shift = S.vsel(S.vdup(23, T.int32),
                                          S.vdup(15, T.int32), mask_low)
                        exp_mask = S.vdup(EXP_MASK, T.uint32)
                        zero_bf16 = S.vdup(0.0, T.bfloat16)
                        for strip in T.serial(hidden // FP4_STRIP):
                            col = strip * FP4_STRIP
                            # 128 FP4 -> 128 bfloat16 -> two float32x64 halves.
                            x_bf16 = S.vcvt(S.vld(q_ub[col], dist="UNPK4_B8"),
                                            T.bfloat16)
                            x_lo, x_hi = S.vintlv(zero_bf16, x_bf16)

                            # Each 64-lane half spans two groups == one packed
                            # word. Transposed index: [word, token], not [token, word].
                            # Plain Python loop, not T.unroll: `half` must be a
                            # real int so that picking x_lo vs x_hi happens at
                            # trace time. Inside T.unroll it is a symbolic var,
                            # `half == 0` is a PrimExpr rather than a bool, and
                            # the branch silently always takes the first arm.
                            for half, values_bf16 in enumerate((x_lo, x_hi)):
                                word = strip * 2 + half
                                packed = S.vld(sf_ub[word, row], dist="BRC_B16")
                                bits = S.vand(
                                    S.vshl(T.reinterpret(packed, "uint32x64"),
                                           sf_shift), exp_mask)
                                scale = T.reinterpret(bits, "float32x64")
                                values = T.reinterpret(values_bf16, "float32x64")
                                S.vsts(out_ub[col + half * LANES],
                                       S.vcvt(S.vmul(values, scale), T.bfloat16),
                                       dist="PK_B32")
                    # --- END SOLUTION
                    T.copy(out_ub, Out[token, 0])

    return cast_back


def launch(q_packed: torch.Tensor, sf_cm: torch.Tensor) -> torch.Tensor:
    hidden = q_packed.shape[1] * 2
    q = q_packed.view(torch.uint8).view(torch.float4_e2m1fn_x2)
    out = compile_kernel(hidden)(q, sf_cm.view(torch.uint16))
    status.assert_on_device("cast_back 07", out)
    return out


def demo_numbers() -> None:
    torch.manual_seed(0)
    x = torch.randn(32, 128) * 3
    q, packed = oracle.per_token(x, CANONICAL_G, fmt="e2m1",
                                 round_sf=True, packed=True)
    cm = oracle.to_col_major(packed)
    print(f"[demo] values : {tuple(x.shape)} float32 -> {tuple(q.shape)} packed FP4")
    print(f"[demo] scales : {tuple(packed.shape)} row-major -> {tuple(cm.shape)} column-major")
    print("[demo] consuming a column-major scale is free: the load is a broadcast,")
    print("[demo] which touches one element, so the stride never matters --")
    print("[demo]   row-major    sf_ub[token, word]")
    print("[demo]   column-major sf_ub[word, token]")
    print("[demo] producing one is the expensive direction (see per_token/05).")
    in_bytes = x.numel() * 2          # as bfloat16
    out_bytes = q.numel() + cm.numel() * 2
    print(f"[demo] footprint: {in_bytes} B bf16 -> {out_bytes} B "
          f"({in_bytes / out_bytes:.1f}x smaller, values + scales)")


def test_correctness() -> None:
    m, k = sim.sim_shapes()
    assert m % BLOCK_MN == 0
    torch.manual_seed(0)
    x = torch.randn(m, k) * 3
    q_packed, packed = oracle.per_token(x, CANONICAL_G, fmt="e2m1",
                                        round_sf=True, packed=True)
    sf_cm = oracle.to_col_major(packed)
    ref = oracle.cast_back(q_packed, packed, (1, CANONICAL_G),
                           packed=True, fp4=True, out_dtype=torch.bfloat16)
    got = launch(q_packed.npu(), sf_cm.npu()).cpu()
    assert_bf16_near(got, ref, f"cast_back_compose({m},{k})", atol=0.0)
    print(f"[check] shape=({m},{k}) sf_cm={tuple(sf_cm.shape)} "
          f"matches the torch oracle exactly")


def main() -> int:
    m, k = sim.sim_shapes()
    if status.unimplemented(VARIANT, lambda: compile_kernel(k)):
        return 0
    sim.maybe_reexec()
    sim.print_banner("asc", "cast_back", "07_col_major_compose")

    def body():
        demo_numbers()
        test_correctness()

    return status.run_variant(VARIANT, body)


if __name__ == "__main__":
    sys.exit(main())
