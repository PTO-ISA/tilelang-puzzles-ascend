# puzzles/asc/quant/answer/per_block/01_raw_32x32.py:52-56
# extracted by tools/make_report_snippets.py -- do not edit
for chunk in T.serial(num_chunks):
    v = S.vabs(S.vcvt(S.vld(flat_x[chunk * LANES],
                            dist="UNPK_B16"), T.float32))
    S.vsts(run_ub[chunk], S.vcmax(v, mask_all),
           dist="ONEPT_B32")
