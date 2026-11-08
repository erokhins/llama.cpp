"""Per-step GPU breakdown from an nsys cuda_gpu_trace CSV (kernels inside CUDA graphs traced per node).

usage: python -I nsys_steps.py TRACE.csv [--seg-gap-us 20] [--tail-s 8] [--show 40] [--top 30] [--anchor REGEX]

Loads the GPU events (kernels, memcpy, memset), keeps the last --tail-s seconds (steady state), and
  1. splits the timeline into *segments*: runs of events whose inter-event idle is below --seg-gap-us (a CUDA graph
     replay is one segment; CPU work between replays shows up as the gap before a segment),
  2. prints the last --show segments in order (gap before, wall, GPU busy, #events, first/last kernel) so the
     structure of a decode step or an MTP round can be read off,
  3. groups segments by their event count (a signature of the graph they replay) and prints, per group, the mean
     gap before, wall, busy and the busy time by kernel family, plus the top kernels,
  4. with --anchor, also splits the tail into steps at every event whose name matches REGEX and prints the mean
     wall / busy / idle per step.
Times in the CSV are ns.
"""
import csv, re, sys
from collections import defaultdict

def arg(name, default, cast=float):
    return cast(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default

def family(name):
    n = name.lower()
    if "moe_cache" in n:                                   return "moe cache lookup (assign+fetch)"
    if "mul_mat_vec_q" in n or "mmvq" in n or "mul_mat_q" in n: return "quantized matmul (mmvq/mmq)"
    if "mul_mat_vec_f" in n or "mul_mat_f" in n:           return "bf16 matmul (mmvf/mmf)"
    if "gemm" in n or "cutlass" in n or "sm90" in n or "sm100" in n or "sm120" in n: return "cublas gemm"
    if "flash_attn" in n or "fattn" in n:                  return "flash attention"
    if "gated_delta" in n or "ssm" in n:                   return "gated delta net / ssm conv"
    if "topk_moe" in n or "moe_weighted" in n:             return "moe router (topk+reduce)"
    if "topk" in n or "top_k" in n or "argsort" in n or "indexer" in n: return "sparse-attn indexer / top-k"
    if "quantize_q8_1" in n:                               return "quantize activations (q8_1)"
    if "norm" in n:                                        return "norm"
    if "dsv4_hc" in n:                                     return "hyper-connection"
    if "fwht" in n or "rope" in n:                         return "rope / fwht"
    if "memcpy" in n or "memset" in n or "cpy_" in n or "set_rows" in n or "get_rows" in n or "concat" in n or "fill_kernel" in n: return "copies / kv store / fill"
    return "elementwise / other"

def main():
    path = sys.argv[1]
    seg_gap = arg("--seg-gap-us", 20.0) * 1000
    tail_s = arg("--tail-s", 8.0)
    show = arg("--show", 40, int)
    top = arg("--top", 30, int)
    anchor = re.compile(arg("--anchor", "", str)) if "--anchor" in sys.argv else None
    rows = []
    with open(path, newline="", encoding="utf-8", errors="ignore") as f:
        r = csv.DictReader(f)
        cols = r.fieldnames
        c_start = next(c for c in cols if c.lower().startswith("start"))
        c_dur = next(c for c in cols if c.lower().startswith("duration"))
        for row in r:
            try:
                rows.append((int(row[c_start]), int(row[c_dur]), row["Name"], row.get("GrdX", "")))
            except ValueError:
                continue
    rows.sort()
    t_end = rows[-1][0] + rows[-1][1]
    rows = [e for e in rows if e[0] >= t_end - tail_s * 1e9]
    print(f"{len(rows)} GPU events in the last {tail_s:.1f} s")

    # segments
    segs = []; cur = [rows[0]]; gaps = [0.0]
    for prev, ev in zip(rows, rows[1:]):
        gap = ev[0] - (prev[0] + prev[1])
        if gap > seg_gap:
            segs.append(cur); gaps.append(gap); cur = [ev]
        else:
            cur.append(ev)
    segs.append(cur)
    def seg_stats(st):
        wall = (st[-1][0] + st[-1][1]) - st[0][0]
        busy = sum(d for _, d, _, _ in st)
        return wall, busy
    print(f"\n{len(segs)} segments (idle > {seg_gap/1000:.0f} us splits); last {show}:")
    print(f"{'gap before':>11} {'wall':>8} {'busy':>8} {'#ev':>5}  first kernel -> last kernel")
    for g, st in list(zip(gaps, segs))[-show:]:
        wall, busy = seg_stats(st)
        first = re.sub(r"[<(].*", "", st[0][2])[:28]; last = re.sub(r"[<(].*", "", st[-1][2])[:28]
        print(f"{g/1e3:9.1f}us {wall/1e6:7.3f}ms {busy/1e6:7.3f}ms {len(st):5d}  {first} -> {last}")

    # group segments by event count
    groups = defaultdict(list)
    for g, st in zip(gaps, segs):
        groups[len(st)].append((g, st))
    print("\nsegment groups by event count (replayed graph signature):")
    for n_ev, lst in sorted(groups.items(), key=lambda kv: -sum(seg_stats(st)[0] for _, st in kv[1])):
        tot_wall = sum(seg_stats(st)[0] for _, st in lst); tot_busy = sum(seg_stats(st)[1] for _, st in lst)
        tot_gap = sum(g for g, _ in lst)
        if tot_wall < 0.01 * sum(seg_stats(st)[0] for _, st in zip(gaps, segs)) and len(lst) < 5:
            continue
        print(f"\n== {n_ev} events x{len(lst)}: mean gap before {tot_gap/len(lst)/1e3:.1f} us, wall {tot_wall/len(lst)/1e6:.3f} ms, "
              f"GPU busy {tot_busy/len(lst)/1e6:.3f} ms, idle inside {(tot_wall-tot_busy)/len(lst)/1e6:.3f} ms; "
              f"total wall share {100*tot_wall/(t_end - rows[0][0]):.1f}%")
        fam = defaultdict(float); kern = defaultdict(lambda: [0.0, 0])
        for _, st in lst:
            for _, d, name, gx in st:
                fam[family(name)] += d
                key = re.sub(r"\(.*", "", name)[:80] + f" grid.x={gx}"
                kern[key][0] += d; kern[key][1] += 1
        m = len(lst)
        for f, d in sorted(fam.items(), key=lambda kv: -kv[1]):
            print(f"   {d/m/1e6:7.3f} ms  {f}")
        print(f"   top {top} kernels (ms per segment, count per segment):")
        for name, (d, c) in sorted(kern.items(), key=lambda kv: -kv[1][0])[:top]:
            print(f"      {d/m/1e6:7.3f} ms  x{c/m:6.1f}  {name}")

    if anchor:
        idx = [i for i, e in enumerate(rows) if anchor.search(e[2])]
        steps = [rows[a:b] for a, b in zip(idx, idx[1:])]
        if steps:
            wall = sum(st[-1][0] + st[-1][1] - st[0][0] for st in steps)  # note: excludes the gap to the next anchor
            period = sum(rows[b][0] - rows[a][0] for a, b in zip(idx, idx[1:]))
            busy = sum(d for st in steps for _, d, _, _ in st)
            n = len(steps)
            print(f"\nanchored steps ({anchor.pattern}): {n} steps, mean period {period/n/1e6:.3f} ms, GPU busy {busy/n/1e6:.3f} ms, "
                  f"idle {(period-busy)/n/1e6:.3f} ms ({100*(period-busy)/period:.1f}%)")

if __name__ == "__main__":
    main()
