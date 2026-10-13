# puzzles/asc/quant/answer/per_channel/01_raw_32tokens.py:44-49
# extracted by tools/make_report_snippets.py -- do not edit
acc = S.alloc_local((1,), T.float32)
# seeding with the clamp floor makes the clamp free
acc[0] = S.vdup(E4M3_CLAMP_MIN, T.float32)
for row in T.serial(group_tokens):
    v = S.vabs(S.vcvt(S.vld(x_ub[row, col], dist="UNPK_B16"),
                      T.float32))
