# puzzles/simt/quant/answer/cast_back/01_e4m3_fp32sf.py:58-59
# extracted by tools/make_report_snippets.py -- do not edit
for i in T.Parallel(hidden):
    o_l[i] = q_l[i] * sf_ub[i // group_size]
