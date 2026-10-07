"""per_channel 02 -- power-of-two scales, packed along the token axis.

New config: ``use_packed_ue8m0`` -- but packed along **M**, which is unique to
this kernel. Production calls the flag ``sf_col_pack``.

Every other kernel packs its UE8M0 bytes along the channel axis, because that is
where the scales are numerous. Here the shape is inverted:

    per_token    sf is (M, K/32)     K/32 scales per row -> pack along K
    per_channel  sf is (M/32, K)     only M/32 rows      -> pack along M

With M = 32 there is exactly *one* row of scales, so packing along the channel
axis would do nothing useful. Packing along M fuses m-group ``2i`` and
``2i + 1``:

    word[i, c] = exponent(2i, c) | exponent(2i + 1, c) << 8

On the NPU that is a single ``vintlv`` -- interleave two loaded scale rows -- and
nothing else. The ASC and PTO files show it as ``pack_sf_rows``, which is the
exact shape of production's helper of the same name.

A consequence worth noting: this requires an **even** number of token groups, so
M must be a multiple of 64, not 32. The test checks that the odd case is
rejected rather than silently producing garbage.

Run:  python puzzles/torch/quant/answer/per_channel/02_round_packed_m.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[5]))

import torch

from common import oracle, status
from common.consts import BLOCK_MN, E4M3_CLAMP_MIN, E4M3_MAX, PACK_FACTOR
from common.demo import print_example, randn_with_zero_row
from common.check import assert_fp8_near, assert_same_bytes
from common.math_ops import (
    ceil_log2_exp,
    decode_packed_ue8m0_along_m,
    inv_pow2_from_exp,
    pack_ue8m0_along_m,
)

VARIANT = "torch/per_channel/02_round_packed_m"


def torch_per_channel_cast_packed(x: torch.Tensor, group_tokens: int = BLOCK_MN):
    """Return ``(q, sf_packed)`` with power-of-two scales packed along M.

    ``sf_packed`` is (M/group_tokens/PACK_FACTOR, K) int16.
    """
    # TODO: reduce along dim=1 as in variant 01; exp =
    #       ceil_log2_exp(amax/E4M3_MAX); multiply by
    #       inv_pow2_from_exp(exp).unsqueeze(1); return
    #       pack_ue8m0_along_m((exp+127).to(uint8))
    raise NotImplementedError("torch/per_channel/02_round_packed_m: implement torch_per_channel_cast_packed")


def demo_numbers() -> None:
    x = torch.zeros(64, 2, dtype=torch.bfloat16)
    x[0, 0] = 448.0     # m-group 0, channel 0 -> 2^0
    x[32, 0] = 3.5      # m-group 1, channel 0 -> 2^-7
    x[0, 1] = 1.75      # m-group 0, channel 1 -> 2^-8
    x[32, 1] = 448.0    # m-group 1, channel 1 -> 2^0
    q, packed = torch_per_channel_cast_packed(x)
    unpacked = decode_packed_ue8m0_along_m(packed)
    print(f"[demo] M=64 -> 2 token groups, K=2 channels")
    print(f"[demo]   unpacked scales would be (2, 2); packed is {tuple(packed.shape)}")
    print(f"[demo]   word[0,0] = {packed[0, 0].item()} packs m-group 0 and 1 "
          f"for channel 0")
    print(f"[demo]   decoded: m-group 0 -> {unpacked[0, 0].item():g}, "
          f"m-group 1 -> {unpacked[1, 0].item():g}")
    assert unpacked[0, 0].item() == 1.0
    assert abs(unpacked[1, 0].item() - 2.0 ** -7) < 1e-9
    print("[demo] low byte is the even m-group, high byte the odd one")


def test_correctness() -> None:
    torch.manual_seed(0)
    for m, k in ((64, 128), (128, 64)):
        x = randn_with_zero_row(m, k, torch.device("cpu"))
        q, packed = torch_per_channel_cast_packed(x)
        ref_q, ref_packed = oracle.per_channel(x, BLOCK_MN, round_sf=True, packed=True)
        assert_same_bytes(packed, ref_packed, f"sf_packed({m},{k})")
        assert_fp8_near(q, ref_q, f"q({m},{k})")
        _, ref_f32 = oracle.per_channel(x, BLOCK_MN, round_sf=True)
        assert torch.equal(decode_packed_ue8m0_along_m(packed), ref_f32)
        print(f"[check] shape=({m},{k}) packed={tuple(packed.shape)} int16 ok")

    # an odd number of token groups cannot be packed along M
    x_odd = torch.randn(32, 64)
    try:
        torch_per_channel_cast_packed(x_odd)
    except AssertionError:
        print("[check] M=32 (one token group) is correctly rejected: "
              "packing along M needs an even count, so M must be a multiple of 64")
    else:
        raise AssertionError("expected an even-group-count assertion for M=32")


def main() -> int:
    def body():
        demo_numbers()
        test_correctness()

    return status.run_variant(VARIANT, body)


if __name__ == "__main__":
    sys.exit(main())
