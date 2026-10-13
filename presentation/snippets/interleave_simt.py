# puzzles/simt/quant/answer/per_channel/02_round_packed_m.py:74-76
# extracted by tools/make_report_snippets.py -- do not edit
for c in T.Parallel(hidden):
    packed_ub[PACK_FACTOR * c] = sf_rows_ub[0, c]
    packed_ub[PACK_FACTOR * c + 1] = sf_rows_ub[1, c]
