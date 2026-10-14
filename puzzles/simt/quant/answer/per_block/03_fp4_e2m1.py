"""per_block 03 (SIMT). See doc/quant/per_block/03_fp4_e2m1.md"""

import torch
import tilelang
import tilelang.ascend.language as T

from harness import status
from harness.consts import BLOCK_K, BLOCK_MN, E2M1_CLAMP_MIN, E2M1_MAX

THREADS = 128


def e2m1_nibble(v):
    """float32 expression -> e2m1 nibble expression, rounded half-to-even.

    The same helper as per_token/04, copied rather than shared: each variant
    file stands alone. It is the torch tier's bit-exact packer on scalars.

    This is the clearest case in the whole ladder where SIMT is *not* simpler.
    ASC and PTO own a float32 -> e2m1 hardware convert, so their stage 3 is a
    pre-round to bfloat16 with a sticky bit (`V.vunzip`, `V.vor`, `V.vmin`) and
    then one `V.vcvt` -- four operations. No FP4 convert is reachable from a
    SimtVF region, so the encode is written out: sign, exponent, subnormal
    renormalization, guard/round/sticky, saturation, and the nibble pack. It is
    correct and bit-exact, and it is eighteen lines instead of four.
    """
    bits = T.reinterpret("int32", v)
    sign_nib = (bits >> 28) & 0x8
    exps = (bits >> 23) & 0xFF
    mant = bits & 0x7FFFFF

    is_sub = exps < 127
    sh = T.min(126 - exps, 31)                # see per_token/04 on the clamp
    mant_pre = 0x400000 | (mant >> 1)
    sticky_sub = T.if_then_else(is_sub, (mant & 1) | (mant_pre & ((1 << sh) - 1)), 0)
    mant_eff = T.if_then_else(is_sub, mant_pre >> sh, mant)
    exp_eff = T.max(exps, 126) - 126

    m2 = (mant_eff >> 21) & 3
    guard = m2 & 1
    lsb = (m2 >> 1) & 1
    sticky = sticky_sub | (mant_eff & 0x1FFFFF)
    inc = guard & (T.if_then_else(sticky != 0, 1, 0) | lsb)
    tmp = (((exp_eff << 2) | m2) + inc) >> 1
    return sign_nib | T.min(tmp, 7)


@tilelang.jit(target="ascend")
def compile_kernel(hidden: int, block: tuple = (BLOCK_MN, BLOCK_K)):
    """Quantize bfloat16 -> packed FP4 e2m1 with one FP32 scale per tile."""
    bm, bk = block
    assert bm == 32 and bk == 32, "Ascend per_block granularity is 32x32"
    assert hidden % bk == 0
    num_k_blocks = hidden // bk
    num_tokens = T.dynamic("num_tokens")
    num_m_blocks = T.ceildiv(num_tokens, bm)

    @T.prim_func
    def per_block_cast(
        X: T.Tensor((num_tokens, hidden), T.bfloat16),
        # uint8, two nibbles per byte: a SimtVF region has no FP4 convert, so
        # the packing is explicit and the declared dtype says so.
        Q: T.Tensor((num_tokens, hidden // 2), T.uint8),
        Sf: T.Tensor((num_m_blocks, num_k_blocks), T.float32),
    ):
        with T.Kernel(1):
            x_ub = T.alloc_shared((bm, bk), T.bfloat16)
            q_ub = T.alloc_shared((bm, bk // 2), T.uint8)
            sf_ub = T.alloc_shared((1,), T.float32)
            inv_ub = T.alloc_shared((1,), T.float32)

            for mb in T.serial(num_m_blocks):
                for kb in T.serial(num_k_blocks):
                    T.copy(X[mb * bm, kb * bk], x_ub)

                    # --- BEGIN SOLUTION hint="the reduction and the scale are variant 01's -- the flat T.reduce_absmax and then the clamp -- with E2M1_MAX / E2M1_CLAMP_MIN in place of the e4m3 constants. The output is the work, and it is where SIMT LOSES: ASC and PTO finish with a hardware float32 -> e2m1 convert and only have to pre-round to bfloat16 with a sticky bit, four operations; no such convert is reachable here, so e2m1_nibble(v) above does the whole encode by hand. Read it. Then give each thread one output BYTE -- T.Parallel(bm, bk // 2), encoding x_ub[i, 2*j] and x_ub[i, 2*j+1] and packing them lo | (hi << 4), low nibble first -- so the pack needs no exchange between threads. Write the byte to a shared uint8 tile; a fragment would reject the second index pattern on x_ub, shared does not."
                    with T.SimtVF(threads=THREADS):
                        flat_l = T.alloc_fragment((1, bm * bk), T.float32)
                        tile_l = T.alloc_fragment((1,), T.float32)
                        for i, j in T.Parallel(bm, bk):
                            flat_l[0, i * bk + j] = T.Cast("float32", x_ub[i, j])
                        T.reduce_absmax(flat_l, tile_l, dim=1)

                        for _z in T.Parallel(1):
                            tile = T.max(tile_l[0], E2M1_CLAMP_MIN)
                            sf_ub[0] = tile / E2M1_MAX
                            inv_ub[0] = E2M1_MAX / tile

                        # one thread per output byte: it owns both nibbles
                        for i, j in T.Parallel(bm, bk // 2):
                            inv = inv_ub[0]
                            lo = e2m1_nibble(T.Cast("float32", x_ub[i, 2 * j]) * inv)
                            hi = e2m1_nibble(T.Cast("float32", x_ub[i, 2 * j + 1]) * inv)
                            q_ub[i, j] = T.Cast("uint8", lo | (hi << 4))
                    # --- END SOLUTION

                    T.copy(q_ub, Q[mb * bm, kb * bk // 2])
                    T.copy(sf_ub[0:1], Sf[mb, kb:kb + 1])

    return per_block_cast


def launch(x: torch.Tensor):
    m, hidden = x.shape
    q = torch.empty((m, hidden // 2), dtype=torch.uint8, device=x.device)
    sf = torch.empty((m // BLOCK_MN, hidden // BLOCK_K), dtype=torch.float32,
                     device=x.device)
    compile_kernel(hidden)(x, q, sf)
    status.assert_on_device("per_block 03", q, sf)
    return q.view(torch.int8), sf

if __name__ == "__main__":        # not a script -- see the module docstring
    raise SystemExit("Run it through the harness:  python -m harness.check simt/per_block/03")
