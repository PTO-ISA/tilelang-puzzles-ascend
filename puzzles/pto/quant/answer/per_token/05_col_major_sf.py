"""per_token 05 (PTO) -- the in-register transpose in VMI.

Read the ASC variant first: it explains why a transpose inside a vector register
is not a memory operation, and derives the lane arithmetic
(`token = i & 31`, `group = i >> 5`, `source = token * 64 + group`).

### PTO vs ASC: another honest draw

Both backends do the same two things -- build an index vector from the lane id,
then gather -- and VMI does not make either shorter:

    ASC: lane = S.vci(0, T.int32)
         ...
         S.vsts(idx_ub[0], S.vadd(S.vmuls(token, 64), group), dist="NORM_B32")
         vals = S.vgather2(sf_dense_ub[0, base], idx)

    PTO: lane = V.vci(T.int32(0), size=64)
         ...
         V.vstore(V.vadd(V.vmul(token, V.vbrc(T.uint32(64), size=64)), group),
                  idx_ub[0])
         vals = V.vgather(sf_dense_ub[0, base], idx, mask)

VMI is slightly *more* verbose here: there is no scalar-operand `vmul`, so the
stride constant has to be broadcast into a vector first, and `vgather` requires an
explicit mask where `vgather2` takes one optionally.

That is the honest shape of the comparison. VMI's advantages come from `group=`
turning emulated segment behaviour into one operation, and from width being an
argument. Neither applies to a gather: a gather is already exactly what the
hardware does, ASC already says so directly, and there is nothing to factor out.

Production's PTO port shows the same thing -- its `transpose_vectors` is
essentially a transliteration of the ASC one, which is why the port's line savings
came from the reduce/broadcast/convert paths rather than from here.

### One genuine VMI regression, visible nearby

VMI has no per-lane variable *shift*. ASC can build a vector of shift amounts and
apply `S.vshr(v, shifts)`, which production's ASC `cast_back` uses to extract
either byte of a packed scale word. `V.vshrs` takes a scalar amount only, so
production's PTO path computes both byte positions and selects between them with
`vcmp` + `vsel`. This variant does not hit it (a left shift by a *vector* works in
both), but it is the clearest case in the quant family where VMI is the weaker
surface, and it is worth knowing before porting something that relies on it.

Run:  python puzzles/pto/quant/answer/per_token/05_col_major_sf.py
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

VARIANT = "pto/per_token/05_col_major_sf"
LANES = 64
PAIR = 128
SF_STRIDE = 64          # padded row length of the token-major scale buffer


@tilelang.jit(target="pto")
def compile_kernel(hidden: int, group_size: int = CANONICAL_G,
                   token_block: int = BLOCK_MN):
    """Quantize, writing the scales transposed as (num_groups, num_tokens)."""
    assert hidden % PAIR == 0 and group_size == 32 and token_block == 32
    num_groups = hidden // group_size
    log2_block = token_block.bit_length() - 1          # 5
    num_out_values = num_groups * token_block
    groups_per_gather = LANES // token_block           # 2
    assert num_out_values % LANES == 0, "this teaching kernel wants whole gathers"
    num_gathers = num_out_values // LANES
    num_tokens = T.dynamic("num_tokens")
    num_blocks = T.ceildiv(num_tokens, token_block)
    inv_qmax = 1.0 / E4M3_MAX

    @T.prim_func
    def per_token_cast(
        X: T.Tensor((num_tokens, hidden), T.bfloat16),
        Q: T.Tensor((num_tokens, hidden), T.float8_e4m3fn),
        SfCm: T.Tensor((num_groups, num_tokens), T.float32),
    ):
        with T.Kernel(1):
            x_ub = T.alloc_shared((hidden,), T.bfloat16)
            q_ub = T.alloc_shared((hidden,), T.float8_e4m3fn)
            amax_ub = T.alloc_shared((SF_STRIDE,), T.float32)
            inv_ub = T.alloc_shared((SF_STRIDE,), T.float32)
            # token-major scales for a block of tokens, then their transpose
            sf_dense_ub = T.alloc_shared((token_block, SF_STRIDE), T.float32)
            sf_out_ub = T.alloc_shared((num_groups, token_block), T.float32)
            idx_ub = T.alloc_shared((LANES,), T.uint32)

            # --- BEGIN SOLUTION hint="build the gather index vector once with V.vci(T.int32(0), size=64): token = lane & (token_block-1), group = lane >> log2(token_block), idx = token*SF_STRIDE + group -- note there is no scalar-operand vmul, so broadcast the stride. Then per token compute scales into sf_dense_ub[row, :], and after the block transpose with V.vgather(sf_dense_ub[0, base], idx, mask) -> V.vstore(..., sf_out_ub[base, 0])"
            # One lane-index vector serves every block; compute it once.
            with T.SimdVF():
                lane = V.vci(T.int32(0), size=LANES)
                mask64 = V.create_mask(LANES, size=LANES)
                token_of_lane = V.vand(V.vinterpret_cast(lane, "uint32"),
                                       V.vbrc(T.uint32(token_block - 1), size=LANES))
                group_of_lane = V.vinterpret_cast(
                    V.vshrs(lane, log2_block, mask64), "uint32")
                # No scalar-operand vmul in VMI: the stride must be a vector.
                stride_v = V.vbrc(T.uint32(SF_STRIDE), size=LANES)
                V.vstore(V.vadd(V.vmul(token_of_lane, stride_v, mask64),
                                group_of_lane), idx_ub[0])

            for blk in T.serial(num_blocks):
                for row in T.serial(token_block):
                    token = blk * token_block + row
                    T.copy(X[token, 0], x_ub)
                    with T.SimdVF():
                        mask = V.create_mask(PAIR, size=PAIR)
                        m64 = V.create_mask(LANES, size=LANES)
                        shift = V.vbrc(T.uint32(23), size=LANES)
                        one = V.vbrc(T.uint32(1), size=LANES)
                        for pair in T.serial(hidden // PAIR):
                            col = pair * PAIR
                            group = pair * (PAIR // group_size)
                            xv = V.vcvt(V.vload(x_ub[col], size=PAIR), "float32")
                            V.vstore(V.vcmax(V.vabs(xv, mask), mask,
                                             group=PAIR // group_size), amax_ub[group])
                        T.simd.mem_bar("VST_VLD")

                        clamped = V.vmax(V.vload(amax_ub[0], size=LANES),
                                         V.vbrc(T.float32(E4M3_CLAMP_MIN), size=LANES),
                                         m64)
                        bits = V.vinterpret_cast(
                            V.vmul(clamped, V.vbrc(T.float32(inv_qmax), size=LANES), m64),
                            "uint32")
                        biased = V.vadd(V.vshr(V.vsub(bits, one), shift), one)
                        # Scales land token-major; transposed after the block.
                        V.vstore(V.vinterpret_cast(V.vshl(biased, shift), "float32"),
                                 sf_dense_ub[row, 0])
                        V.vstore(V.vinterpret_cast(
                            V.vshl(V.vsub(V.vbrc(T.uint32(254), size=LANES), biased),
                                   shift), "float32"), inv_ub[0])
                        T.simd.mem_bar("VST_VLD")

                        for pair in T.serial(hidden // PAIR):
                            col = pair * PAIR
                            group = pair * (PAIR // group_size)
                            inv = V.vload(inv_ub[group], size=PAIR, stride=1,
                                          dist_mode="brc", group=PAIR // group_size)
                            xv = V.vcvt(V.vload(x_ub[col], size=PAIR), "float32")
                            V.vstore(V.vcvt(V.vmul(xv, inv, mask), "float8_e4m3fn",
                                            rounding="R", saturate="SAT"), q_ub[col])
                    T.copy(q_ub, Q[token, 0])

                # transpose the block's scales in register, then one DMA out
                with T.SimdVF():
                    T.simd.mem_bar("VST_VLD")
                    gmask = V.create_mask(LANES, size=LANES)
                    idx = V.vload(idx_ub[0], size=LANES)
                    for g in T.serial(num_gathers):
                        base = g * groups_per_gather
                        # vgather requires the mask; vgather2 takes it optionally.
                        V.vstore(V.vgather(sf_dense_ub[0, base], idx, gmask),
                                 sf_out_ub[base, 0])
                T.copy(sf_out_ub, SfCm[0, blk * token_block])
            # --- END SOLUTION

    return per_token_cast


def launch(x: torch.Tensor):
    m, hidden = x.shape
    num_groups = hidden // CANONICAL_G
    q = torch.empty((m, hidden), dtype=torch.float8_e4m3fn, device=x.device)
    sf_cm = torch.empty((num_groups, m), dtype=torch.float32, device=x.device)
    compile_kernel(hidden)(x, q, sf_cm)
    status.assert_on_device("per_token 05", q, sf_cm)
    return q, sf_cm


def demo_numbers() -> None:
    tb, stride = BLOCK_MN, SF_STRIDE
    print(f"[demo] transposing a ({tb}, groups) scale block to (groups, {tb}):")
    print("[demo] for output flat index i, the source element is:")
    print(f"[demo]   token = i & {tb - 1}      group = i >> {tb.bit_length() - 1}"
          f"      source = token * {stride} + group")
    for i in (0, 1, 31, 32, 33, 63):
        token, group = i & (tb - 1), i >> (tb.bit_length() - 1)
        print(f"[demo]   lane {i:3d} -> out[g={group}, t={token:2d}] "
              f"<- src flat {token * stride + group}")
    print("[demo] V.vci gives each lane its index, so that table is one vector")
    print("[demo] computed once; V.vgather then fetches 64 elements in the")
    print("[demo] output's order. A register lane cannot move on its own.")
    print("[demo] this variant is a draw: a gather is already exactly what the")
    print("[demo] hardware does, so there is nothing for VMI to factor out, and")
    print("[demo] VMI is slightly longer (no scalar-operand vmul, mandatory mask).")


def test_correctness() -> None:
    m, k = sim.sim_shapes()
    assert m % BLOCK_MN == 0, f"the transpose works on blocks of {BLOCK_MN} tokens"
    torch.manual_seed(0)
    x = randn_with_zero_row(m, k, torch.device("cpu"))
    ref_q, ref_sf = oracle.per_token(x, CANONICAL_G, round_sf=True)
    q, sf_cm = launch(x.npu())
    assert sf_cm.shape == (k // CANONICAL_G, m), sf_cm.shape
    assert_fp32_ulps(sf_cm.cpu().T.contiguous(), ref_sf, f"sf_cm.T({m},{k})", max_ulps=0)
    assert_fp8_near(q.cpu(), ref_q, f"q({m},{k})")
    print(f"[check] shape=({m},{k}) sf_cm={tuple(sf_cm.shape)} transposes back "
          f"to {tuple(ref_sf.shape)} bit-exactly")


def main() -> int:
    m, k = sim.sim_shapes()
    if status.unimplemented(VARIANT, lambda: compile_kernel(k)):
        return 0
    sim.maybe_reexec()
    sim.print_banner("pto", "per_token", "05_col_major_sf")

    def body():
        demo_numbers()
        test_correctness()

    return status.run_variant(VARIANT, body)


if __name__ == "__main__":
    sys.exit(main())
