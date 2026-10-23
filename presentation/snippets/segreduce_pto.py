# puzzles/pto/quant/answer/per_token/01_raw_fp32sf.py:54-56
# extracted by tools/make_report_snippets.py -- do not edit
x = V.vcvt(V.vload(x_ub[col], size=PAIR), "float32")
V.vstore(V.vcmax(V.vabs(x, mask), mask, group=groups_per_pair),
         amax_ub[group])
