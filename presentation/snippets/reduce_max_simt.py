# harness/probe/simt_reduce_max.py:114-117
# extracted by tools/make_report_snippets.py -- do not edit
for g, j in T.Parallel(num_groups, group_size):
    a_l[g, j] = T.abs(
        T.Cast("float32", x_ub[g * group_size + j]))
T.reduce_max(a_l, amax_l, dim=1)
