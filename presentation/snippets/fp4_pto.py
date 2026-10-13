# puzzles/pto/quant/answer/per_token/04_fp32_in_fp4_out.py:31-33
# extracted by tools/make_report_snippets.py -- do not edit
low, high = V.vunzip(x, "uint16")
one = V.vbrc(T.uint16(1), size=lanes)
return V.vinterpret_cast(V.vor(high, V.vmin(low, one)), "bfloat16")
