# puzzles/simt/quant/answer/per_channel/01_raw_32tokens.py:50-56
# extracted by tools/make_report_snippets.py -- do not edit
for c in T.Parallel(hidden):
    amax = T.alloc_var(T.float32)
    amax = E4M3_CLAMP_MIN
    for row in T.serial(group_tokens):
        amax = T.max(amax,
                     T.abs(T.Cast("float32", x_ub[row, c])))
    sf_ub[c] = amax / E4M3_MAX
