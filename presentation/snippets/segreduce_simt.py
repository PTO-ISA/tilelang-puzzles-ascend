# puzzles/simt/quant/answer/per_token/01_raw_fp32sf.py:56-58
# extracted by tools/make_report_snippets.py -- do not edit
for g, j in T.Parallel(num_groups, group_size):
    x_l[g, j] = T.Cast("float32", x_ub[g * group_size + j])
T.reduce_absmax(x_l, amax_l, dim=1)
