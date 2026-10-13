# puzzles/pto/quant/answer/cast_back/01_e4m3_fp32sf.py:70-73
# extracted by tools/make_report_snippets.py -- do not edit
if out_dtype == "bfloat16":
    V.vstore(V.vcvt(scaled, "bfloat16"), out_ub[col])
else:
    V.vstore(scaled, out_ub[col])
