"""cast_back 07 (PTO) -- column-major scales, everything composed, one vector wide.

Final cast_back variant. Composed config:

    packed UE8M0 scales + column-major layout + FP4 values -> bfloat16 output

Read the ASC variant for why consuming column-major scales is free (the scale load
is a broadcast, so the stride never matters) and why *producing* them is the hard
direction instead.

### PTO vs ASC: the strip stops being split

An FP4 unpacking load gives 128 values. In ASC the scale machinery works a
register at a time, so the strip is processed as two 64-lane halves and the whole
scale-decode sequence runs **twice** per strip:

    ASC, per strip:
        x_bf16     = vcvt(vld(q_ub[col], UNPK4_B8), bfloat16)
        x_lo, x_hi = vintlv(zero, x_bf16)
        for half in (lo, hi):                      # <- the sequence runs twice
            packed = vld(sf_ub[word, row], BRC_B16)
            bits   = vand(vshl(reinterpret(packed, "uint32x64"), shift), exp_mask)
            vsts(out_ub[...], vcvt(vmul(reinterpret(half, "float32x64"), bits)))

VMI keeps 128 lanes as one logical vector the whole way, so the sequence runs
once:

    PTO, per strip:
        x_f32 = V.vzip(zero, V.vcvt(V.vload(q_ub[col], size=128), "bfloat16"),
                       "float32")
        packed = V.vload(sf_ub[word, row], size=256, stride=token_block,
                         dist_mode="brc", group=2)
        bits   = V.vand(V.vshl(V.vinterpret_cast(packed, "uint32"), shift), exp_mask)
        V.vstore(V.vcvt(V.vmul(x_f32, V.vinterpret_cast(bits, "float32"), mask),
                        "bfloat16"), out_ub[col])

Two VMI features make that possible, and both are the same idea -- *width as an
argument*:

1. **`dist_mode="brc"` with `group=2` and a stride** fetches *two* packed words,
   broadcasting each across 64 lanes, in one load. ASC's `BRC_B16` broadcasts a
   single scalar, so it needs one load per word.
2. **`V.create_mask(32, size=128, group=2)`** builds the repeating
   "first 32 of every 64 lanes" predicate that selects shift 23 vs shift 15 across
   all 128 lanes at once. ASC's `S.pset(32, "PAT_VL32")` is a 64-lane pattern from
   a fixed catalogue; there is no 128-lane form of it.

### What the measurement says -- including where it says "no difference"

`python tools/vf_lines.py` counts vector operations per variant. Across cast_back
PTO runs at 80% of ASC's operation count (57 vs 71), but the saving is not spread
evenly, and **this variant is a draw**: 14 operations each.

That is worth understanding rather than glossing. The grouped scale load does
replace ASC's two broadcasts -- but ASC hoists its shift-pattern setup out of the
strip loop and then reuses it for *both* halves, so the per-half cost it pays is
small. PTO's single-pass form avoids the duplication and pays for an extra
128-lane setup instead. The two roughly cancel.

Where the width advantage really shows is variant 06, which saves 8 of 16
operations. The difference between the two cases: in 06 it is the *value* path
that ASC has to split (128 FP4 values into two float32 registers, with the whole
multiply-convert-store sequence duplicated), whereas here the thing being
duplicated is just a two-instruction scale decode. VMI's "width is an argument"
property pays in proportion to how much work sits inside the duplicated region.

Source *lines* come out essentially equal across the kernel (85 vs 86), because
VMI's mandatory `size=` and mask arguments make each call wider. Both numbers are
worth knowing: the operation count is what the brevity claim is about, and the
line count is the ergonomic price of VMI's explicitness.

(A measurement note, since it would otherwise bias the comparison: the op count
excludes bit reinterpretation, which emits no instruction. ASC spells it
`T.reinterpret` and VMI spells it `V.vinterpret_cast`, so counting the VMI form
alone would have penalised PTO for a purely notational difference -- it put this
variant at +2 before the tool was corrected.)

Run:  python puzzles/pto/quant/answer/cast_back/07_col_major_compose.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[5]))

import torch
import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import vmi as V

from common import oracle, sim, status
from common.check import assert_bf16_near
from common.consts import BLOCK_MN, CANONICAL_G, PACK_FACTOR

VARIANT = "pto/cast_back/07_col_major_compose"
FP4_STRIP = 128
EXP_MASK = 0x7F800000


@tilelang.jit(target="pto", out_idx=[2])
def compile_kernel(hidden: int, group_size: int = CANONICAL_G,
                   token_block: int = BLOCK_MN):
    """Packed UE8M0 + column-major scales + FP4 values -> bfloat16."""
    assert hidden % FP4_STRIP == 0 and group_size == 32
    num_groups = hidden // group_size
    num_words = num_groups // PACK_FACTOR
    words_per_strip = FP4_STRIP // (group_size * PACK_FACTOR)    # 2
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
            sf_ub = T.alloc_shared((num_words, token_block), T.uint16)
            out_ub = T.alloc_shared((hidden,), T.bfloat16)
            for blk in T.serial(num_blocks):
                T.copy(SfCm[0, blk * token_block], sf_ub)
                for row in T.serial(token_block):
                    token = blk * token_block + row
                    T.copy(Q[token, 0], q_ub)
                    # --- BEGIN SOLUTION hint="stay at 128 lanes: x_f32 = V.vzip(zero_bf16, V.vcvt(V.vload(q_ub[col], size=128), 'bfloat16'), 'float32'); fetch both packed words in one go with V.vload(sf_ub[strip*2, row], size=256, stride=token_block, dist_mode='brc', group=2); build sf_shift from V.create_mask(32, size=128, group=2); then shift, mask, reinterpret to float32, multiply and store once"
                    with T.SimdVF():
                        mask = V.create_mask(FP4_STRIP, size=FP4_STRIP)
                        # "first 32 of every 64 lanes", repeating -- there is no
                        # 128-lane equivalent of ASC's PAT_VL32.
                        mask_low = V.create_mask(32, size=FP4_STRIP, group=2)
                        sf_shift = V.vsel(mask_low,
                                          V.vbrc(T.uint32(23), size=FP4_STRIP),
                                          V.vbrc(T.uint32(15), size=FP4_STRIP))
                        exp_mask = V.vbrc(T.uint32(EXP_MASK), size=FP4_STRIP)
                        zero_bf16 = V.vbrc(T.bfloat16(0.0), size=FP4_STRIP)
                        for strip in T.serial(hidden // FP4_STRIP):
                            col = strip * FP4_STRIP
                            word = strip * words_per_strip

                            # 128 FP4 -> 128 bfloat16 -> 128 float32, one name.
                            x_bf16 = V.vcvt(V.vload(q_ub[col], size=FP4_STRIP),
                                            "bfloat16")
                            x_f32 = V.vzip(zero_bf16, x_bf16, "float32")

                            # Two packed words, each broadcast over 64 lanes, in
                            # one load. The transposed layout is just the index.
                            packed = V.vload(sf_ub[word, row], size=FP4_STRIP * 2,
                                             stride=token_block, dist_mode="brc",
                                             group=words_per_strip)
                            bits = V.vand(
                                V.vshl(V.vinterpret_cast(packed, "uint32"), sf_shift),
                                exp_mask)
                            scale = V.vinterpret_cast(bits, "float32")

                            scaled = V.vmul(x_f32, scale, mask)
                            V.vstore(V.vcvt(scaled, "bfloat16"), out_ub[col])
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
    print("[demo] one 128-value FP4 strip needs two packed scale words.")
    print("[demo]   ASC: one BRC_B16 load each, so the decode sequence runs twice")
    print("[demo]   PTO: one brc load with group=2 and stride, decode runs once")
    in_bytes = x.numel() * 2
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
    sim.print_banner("pto", "cast_back", "07_col_major_compose")

    def body():
        demo_numbers()
        test_correctness()

    return status.run_variant(VARIANT, body)


if __name__ == "__main__":
    sys.exit(main())
