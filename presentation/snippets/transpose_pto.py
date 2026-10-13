# puzzles/pto/quant/answer/per_token/05_col_major_sf.py:57-59
# extracted by tools/make_report_snippets.py -- do not edit
stride_v = V.vbrc(T.uint32(SF_STRIDE), size=LANES)
V.vstore(V.vadd(V.vmul(token_of_lane, stride_v, mask64),
                group_of_lane), idx_ub[0])
