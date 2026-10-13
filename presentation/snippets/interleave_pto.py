# puzzles/pto/quant/answer/per_channel/02_round_packed_m.py:94-97
# extracted by tools/make_report_snippets.py -- do not edit
a = V.vload(sf_rows_ub[0, col], size=BYTE_LANES * 2)
b = V.vload(sf_rows_ub[1, col], size=BYTE_LANES * 2)
# A uint8 register is 256 lanes, so `lo` already holds the
# interleave of the first 128 elements of each input --
