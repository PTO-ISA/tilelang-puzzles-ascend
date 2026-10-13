# puzzles/torch/quant/answer/per_token/01_raw_fp32sf.py:23-26
# extracted by tools/make_report_snippets.py -- do not edit
grouped = x.float().view(m, k // group_size, group_size)
amax = grouped.abs().amax(dim=-1).clamp(min=E4M3_CLAMP_MIN)
sf = amax / E4M3_MAX
q = (grouped * (E4M3_MAX / amax).unsqueeze(-1)).view(m, k).to(torch.float8_e4m3fn)
