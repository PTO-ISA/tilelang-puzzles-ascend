# puzzles/simt/quant/answer/per_token/04_fp32_in_fp4_out.py:62-67
# extracted by tools/make_report_snippets.py -- do not edit
sh = T.min(126 - exps, 31)
mant_pre = 0x400000 | (mant >> 1)         # the implicit 1, then make room
low = (1 << sh) - 1                       # the bits about to be shifted out

if branch_free:
    # -(cond) is 0 or all-ones. Both arms are always evaluated, which is
