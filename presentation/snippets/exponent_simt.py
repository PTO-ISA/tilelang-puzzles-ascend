# puzzles/simt/quant/answer/per_token/02_round_sf.py:46-57
# extracted by tools/make_report_snippets.py -- do not edit
for g in T.Parallel(num_groups):
    amax = T.alloc_var(T.float32)
    amax = E4M3_CLAMP_MIN
    for j in T.serial(group_size):
        v = T.Cast("float32", x_ub[g * group_size + j])
        amax = T.max(amax, T.abs(v))

    bits = T.reinterpret("int32", amax / E4M3_MAX)
    biased = ((bits - 1) >> 23) + 1
    sf_l[g] = T.reinterpret("float32", biased << 23)
    inv_l[g] = T.reinterpret("float32", (254 - biased) << 23)
