# puzzles/asc/quant/answer/cast_back/01_e4m3_fp32sf.py:77-82
# extracted by tools/make_report_snippets.py -- do not edit
if out_dtype == "bfloat16":
    S.vsts(out_ub[col], S.vcvt(scaled, T.bfloat16),
           dist="PK_B32")      # 64 lanes -> 64 x 2 bytes
else:
    S.vsts(out_ub[col], scaled,
           dist="NORM_B32")    # 64 lanes -> 64 x 4 bytes
