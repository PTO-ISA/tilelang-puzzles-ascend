# puzzles/simt/quant/answer/per_token/04_fp32_in_fp4_out.py:61-66
# extracted by tools/make_report_snippets.py -- do not edit
bits = T.reinterpret("int32", v)
# bit 3 of (bits >> 28) is bit 31 of bits: the sign nibble, without
# depending on whether >> sign-extends.
sign_nib = (bits >> 28) & 0x8
exps = (bits >> 23) & 0xFF
mant = bits & 0x7FFFFF
