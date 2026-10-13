# puzzles/asc/quant/answer/per_token/04_fp32_in_fp4_out.py:73-77
# extracted by tools/make_report_snippets.py -- do not edit
low, high = S.vdintlv(T.reinterpret(q0, "uint16x128"),
                      T.reinterpret(q1, "uint16x128"))
odd = S.vor(high, S.vmin(low, one_u16))
S.vsts(q_ub[col],
       S.vcvt(T.reinterpret(odd, "bfloat16x128"),
