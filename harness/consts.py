"""Format constants and the small shapes used under the CPU simulator."""

# FP8 e4m3 and FP4 e2m1 dynamic range.
E4M3_MAX = 448.0
E2M1_MAX = 6.0

# An all-zero group would give sf_inv = max/0 = Inf and then 0*Inf = NaN, and a
# packed exponent byte of 0x00 -- a NaN scale for the downstream GEMM. Clamping
# amax from below is what keeps a zero row harmless.
E4M3_CLAMP_MIN = 1e-4
E2M1_CLAMP_MIN = 6.0 * (2**-126)

# The production Ascend kernels (github.com/deepseek-ai/TileKernels) fix the
# quant group / block size to 32.
CANONICAL_G = 32
BLOCK_MN = 32
BLOCK_K = 32

# Ascend UE8M0 packing: two exponent bytes -> one int16. (CUDA packs 4 -> int32.)
PACK_FACTOR = 2

# Default simulator shape. Small on purpose: the camodel is cycle-accurate, so
# wall time scales with real instruction count. M must be a multiple of 32 so
# per_block / per_channel have a whole group.
SIM_M = 32
SIM_K = 128

# Time budget per kernel variant under the CPU simulator.
SIM_TARGET_SECONDS = 60
SIM_CEILING_SECONDS = 180
