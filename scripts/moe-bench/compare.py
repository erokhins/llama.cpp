"""Print a comparison table from bench.py result files.

usage: python -I compare.py LABEL=path.json [LABEL=path.json ...]
Columns: decode tok/s (median, short prompt), decode after 30k prompt, server-side prompt tok/s
at 4k/15k/30k, client TTFT at 30k, 4-request aggregate tok/s, MTP acceptance.
"""
import json, sys

rows = []
for arg in sys.argv[1:]:
    label, path = arg.split("=", 1)
    d = json.load(open(path))
    s = d["summary"]
    pf = {}
    for p in d["prefill"]:
        n = p["prompt_tokens"]
        st = p.get("server_timings") or {}
        bucket = 4 if n < 8000 else 15 if n < 20000 else 30
        pf[bucket] = (st.get("prompt_per_second"), p["prefill_tps_client"], p["ttft_s"])
    acc = []
    for r in d["decode"] + [d["long_ctx_decode"]] if isinstance(d["long_ctx_decode"], dict) else d["decode"]:
        st = r.get("server_timings") or {}
        if st.get("draft_n"):
            acc.append(st["draft_n_accepted"] / st["draft_n"])
    rows.append((label, s["decode_tps_median"], s["long_ctx_decode_tps"],
                 pf.get(4, (None,)*3)[0], pf.get(15, (None,)*3)[0], pf.get(30, (None,)*3)[0],
                 pf.get(30, (None,)*3)[2], s["concurrency4_aggregate_tps"],
                 (sum(acc)/len(acc)) if acc else None))

def f(v, w=7, p=1):
    return (" " * w) if v is None else f"{v:{w}.{p}f}"

print(f"{'config':34s} {'gen':>7s} {'gen@30k':>7s} {'pp4k':>7s} {'pp15k':>7s} {'pp30k':>7s} {'ttft30k':>7s} {'4req':>7s} {'accept':>7s}")
for r in rows:
    print(f"{r[0]:34s} {f(r[1])} {f(r[2])} {f(r[3],p=0)} {f(r[4],p=0)} {f(r[5],p=0)} {f(r[6],p=2)} {f(r[7])} {f(r[8],p=2)}")
