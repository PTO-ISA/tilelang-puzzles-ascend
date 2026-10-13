# puzzles/asc/quant/answer/per_channel/02_round_packed_m.py:88-91
# extracted by tools/make_report_snippets.py -- do not edit
a = S.vld(sf_rows_ub[0, col], dist="NORM_B8")
b = S.vld(sf_rows_ub[1, col], dist="NORM_B8")
# A uint8 register is 256 lanes, so `lo` already holds the
# interleave of the first 128 elements of each input --
