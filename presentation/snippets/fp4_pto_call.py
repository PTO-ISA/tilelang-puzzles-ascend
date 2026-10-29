# puzzles/pto/quant/answer/per_token/04_fp32_in_fp4_out.py:80-81
# extracted by tools/make_report_snippets.py -- do not edit
odd = to_bf16_round_odd(q, PAIR)
V.vstore(V.vcvt(odd, "float4_e2m1fn", rounding="R"), q_ub[col])
