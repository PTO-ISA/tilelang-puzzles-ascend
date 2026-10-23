# puzzles/simt/quant/answer/per_token/05_col_major_sf.py:56-65
# extracted by tools/make_report_snippets.py -- do not edit
for g in T.Parallel(num_groups):
    amax = T.max(amax_l[g], E4M3_CLAMP_MIN)

    # ceil(log2(amax/448)) from the exponent field, then
    # both scales rebuilt by writing an exponent back.
    bits = T.reinterpret("int32", amax / E4M3_MAX)
    biased = ((bits - 1) >> 23) + 1
    inv_l[g] = T.reinterpret("float32", (254 - biased) << 23)
    # The whole column-major config: a different index.
    sf_out_ub[g, sub] = T.reinterpret("float32", biased << 23)
