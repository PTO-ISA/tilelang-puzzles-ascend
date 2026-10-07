"""per_token 04 (ASC) -- float32 input, FP4 (e2m1) output.

Two independent dtype changes. One makes the kernel simpler, the other forces a
two-step conversion with a subtle rounding requirement.

### float32 input is the easy case

bfloat16 input needed an unpacking load and a widening convert:

    S.vcvt(S.vld(x_ub[col], dist="UNPK_B16"), T.float32)

float32 input is already the vector unit's compute type, so it is a plain load:

    S.vld(x_ub[col])

(The trade is bandwidth: float32 input moves twice the bytes from GM.)

### FP4 output needs bfloat16 in the middle

There is **no float32 -> e2m1 conversion instruction**. Asking for one is a hard
error on both backends:

    ASC: ValueError: Unsupported vcvt conversion float32->float4_e2m1fn
    PTO: TypeError: ... supports packed FP4 only for bfloat16 to float4_e2m1fn

So the quantized values must go float32 -> bfloat16 -> e2m1. That is two roundings
in a row, and doing it naively biases the result: a value exactly between two FP4
codes can be rounded *up* into the bfloat16 step and then up again, when rounding
it down the second time would have been correct. Classic double rounding.

The fix is **round to odd** on the intermediate: truncate to bfloat16, but force
the last mantissa bit to 1 whenever any truncated bit was nonzero. An odd
intermediate can never sit exactly on a midpoint of the final format, so the
second rounding always goes the right way.

In ASC the whole thing is bit manipulation, and it fuses the two 64-lane halves
into one 128-lane bfloat16 vector on the way:

    low, high = S.vdintlv(T.reinterpret(q0, "uint16x128"),
                          T.reinterpret(q1, "uint16x128"))
    odd       = S.vor(high, S.vmin(low, one_u16))  # set LSB if anything was dropped
    S.vsts(q_ub[col], S.vcvt(T.reinterpret(odd, "bfloat16x128"), T.float4_e2m1fn),
           dist="PK4_B32")

`S.vdintlv` deinterleaves: given two float32 registers viewed as 16-bit lanes it
returns all the low halves in one vector and all the high halves in another. The
high halves *are* the truncated bfloat16 values. `S.vmin(low, one)` is `min(low, 1)`, i.e. 1 if any low bit was set and 0
otherwise -- the sticky bit.

A toolchain note: the scalar-operand form `S.vmins(low, 1)` would read more
naturally, and production uses it, but on this build it fails to compile --
`asc_min_scalar` has no uint16 overload:

    error: no matching function for call to 'asc_min_scalar'

The vector form `S.vmin` against a splatted `1` does have one, so that is what
this file uses. Same result, one extra `vdup`.

### quant_max changes too

e2m1's largest magnitude is 6.0, not 448.0, and the clamp floor becomes
`6.0 * 2^-126`. See the torch variant.

Run:  python puzzles/asc/quant/answer/per_token/04_fp32_in_fp4_out.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[5]))

import torch
import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S

from common import oracle, sim, status
from common.check import assert_fp32_ulps
from common.consts import CANONICAL_G, E2M1_CLAMP_MIN, E2M1_MAX
from common.demo import randn_with_zero_row
from common.math_ops import unpack_e2m1_bytes

VARIANT = "asc/per_token/04_fp32_in_fp4_out"
LANES = 64
PAIR = 128
SF_PAD = 64


@tilelang.jit(target="ascend")
def compile_kernel(hidden: int, group_size: int = CANONICAL_G):
    """Quantize float32 -> packed FP4 e2m1 with one FP32 scale per 32 channels."""
    assert hidden % PAIR == 0 and group_size == 32
    num_groups = hidden // group_size
    num_tokens = T.dynamic("num_tokens")

    @T.prim_func
    def per_token_cast(
        X: T.Tensor((num_tokens, hidden), T.float32),
        Q: T.Tensor((num_tokens, hidden), T.float4_e2m1fn),
        Sf: T.Tensor((num_tokens, num_groups), T.float32),
    ):
        with T.Kernel(1):
            x_ub = T.alloc_shared((hidden,), T.float32)
            q_ub = T.alloc_shared((hidden,), T.float4_e2m1fn)
            amax_ub = T.alloc_shared((SF_PAD,), T.float32)
            sf_ub = T.alloc_shared((SF_PAD,), T.float32)
            inv_ub = T.alloc_shared((SF_PAD,), T.float32)

            for token in T.serial(num_tokens):
                T.copy(X[token, 0], x_ub)
                # TODO: float32 input needs no unpacking load: just
                #       S.vld(x_ub[col]). Use E2M1_MAX/E2M1_CLAMP_MIN instead of
                #       the e4m3 constants. For the store, there is no
                #       float32->e2m1 convert: deinterleave the two quantized
                #       halves with low, high = S.vdintlv(T.reinterpret(q0,
                #       'uint16x128'), T.reinterpret(q1, 'uint16x128')), round to
                #       odd with S.vor(high, S.vmins(low, 1)), reinterpret to
                #       'bfloat16x128', then S.vcvt to T.float4_e2m1fn and store
                #       with dist='PK4_B32'
                raise NotImplementedError("asc/per_token/04_fp32_in_fp4_out: implement per_token_cast")
                T.copy(q_ub, Q[token, 0])
                T.copy(sf_ub[0:num_groups], Sf[token, 0])

    return per_token_cast


def launch(x: torch.Tensor):
    m, hidden = x.shape
    q = torch.empty((m, hidden // 2), dtype=torch.uint8,
                    device=x.device).view(torch.float4_e2m1fn_x2)
    sf = torch.empty((m, hidden // CANONICAL_G), dtype=torch.float32, device=x.device)
    compile_kernel(hidden)(x, q, sf)
    status.assert_on_device("per_token 04", q, sf)
    return q.view(torch.uint8).view(torch.int8), sf


def demo_numbers() -> None:
    print("[demo] there is no float32 -> e2m1 convert instruction:")
    print("[demo]   ASC: ValueError: Unsupported vcvt conversion float32->float4_e2m1fn")
    print("[demo]   PTO: supports packed FP4 only for bfloat16 to float4_e2m1fn")
    print("[demo] so the path is float32 -> bfloat16 -> e2m1, two roundings.")
    print("[demo] round-to-odd on the intermediate avoids double-rounding bias:")
    # 1.0 + a tiny residue: truncating to bf16 gives exactly 1.0, which is a
    # midpoint between e2m1's 1.0 and 1.5 only if the residue is lost.
    v = 1.25 + 2.0 ** -20
    bits = torch.tensor([v], dtype=torch.float32).view(torch.int32).item()
    high, low = (bits >> 16) & 0xFFFF, bits & 0xFFFF
    trunc = torch.tensor([high << 16], dtype=torch.int32).view(torch.float32).item()
    odd = torch.tensor([(high | min(low, 1)) << 16],
                       dtype=torch.int32).view(torch.float32).item()
    print(f"[demo]   v = {v!r}")
    print(f"[demo]     truncated to bf16 : {trunc!r}  (looks exactly like a midpoint)")
    print(f"[demo]     round-to-odd      : {odd!r}  (LSB set, so not a midpoint)")
    assert low != 0 and odd != trunc, "the sticky bit should change the intermediate"
    print("[demo] an odd intermediate can never tie in the second rounding")


def test_correctness() -> None:
    m, k = sim.sim_shapes()
    torch.manual_seed(0)
    x = randn_with_zero_row(m, k, torch.device("cpu"), dtype=torch.float32) * 3
    ref_q, ref_sf = oracle.per_token(x, CANONICAL_G, fmt="e2m1")
    q, sf = launch(x.npu())
    assert_fp32_ulps(sf.cpu(), ref_sf, f"sf({m},{k})", max_ulps=1)

    got_v = unpack_e2m1_bytes(q.cpu())
    ref_v = unpack_e2m1_bytes(ref_q)
    # e2m1 has 8 magnitudes, so the smallest possible disagreement is one code --
    # which here means a factor of up to 1.5. Count them rather than hide them.
    differing = int((got_v != ref_v).sum())
    total = got_v.numel()
    print(f"[check] shape=({m},{k}) FP4 values differing from the torch packer: "
          f"{differing}/{total} ({differing / total:.2%})")
    assert differing / total < 0.02, (
        f"{differing}/{total} FP4 codes differ; the kernel's two-step rounding "
        f"should track the torch packer closely"
    )
    back = got_v * sf.cpu().repeat_interleave(CANONICAL_G, dim=1)
    rel = (back - x).abs().max().item() / x.abs().max().item()
    print(f"[check] round-trip rel-err {rel:.1%} (FP4 has one mantissa bit)")


def main() -> int:
    m, k = sim.sim_shapes()
    if status.unimplemented(VARIANT, lambda: compile_kernel(k)):
        return 0
    sim.maybe_reexec()
    sim.print_banner("asc", "per_token", "04_fp32_in_fp4_out")

    def body():
        demo_numbers()
        test_correctness()

    return status.run_variant(VARIANT, body)


if __name__ == "__main__":
    sys.exit(main())
