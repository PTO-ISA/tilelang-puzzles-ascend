# puzzles/simt/quant/answer/per_token/04_fp32_in_fp4_out.py:251-256
# extracted by tools/make_report_snippets.py -- do not edit
with T.SimdVF():
    for pair in T.serial(hidden // PAIR):
        S.vsts(q_ub[pair * PAIR],
               S.vcvt(S.vld(bf_ub[pair * PAIR]),
                      T.float4_e2m1fn),
               dist="PK4_B32")
