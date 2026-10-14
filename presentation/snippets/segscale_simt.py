# puzzles/simt/quant/answer/per_token/01_raw_fp32sf.py:62-65
# extracted by tools/make_report_snippets.py -- do not edit
for g in T.Parallel(num_groups):
    amax_l[g] = T.max(amax_l[g], E4M3_CLAMP_MIN)
    inv_l[g] = E4M3_MAX / amax_l[g]
    sf_l[g] = amax_l[g] / E4M3_MAX
