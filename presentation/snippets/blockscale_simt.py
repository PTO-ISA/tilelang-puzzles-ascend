# puzzles/simt/quant/answer/per_block/01_raw_32x32.py:69-72
# extracted by tools/make_report_snippets.py -- do not edit
for _z in T.Parallel(1):
    tile = T.max(tile_l[0], E4M3_CLAMP_MIN)
    sf_ub[0] = tile / E4M3_MAX
    inv_ub[0] = E4M3_MAX / tile
