"""Copy-vs-compute breakdown of a prompt-processing run from an nsys cuda_gpu_trace CSV.

usage: python -I nsys_prefill.py TRACE.csv [--gap-ms 200] [--top 25]

Finds the longest run of GPU activity whose inter-event gaps stay below --gap-ms (the prompt), then reports, inside
that window: wall time, union GPU-busy time, H2D / D2D / D2H memcpy count, bytes, busy time and effective bandwidth,
kernel busy time by family, the time kernels and H2D copies overlap, and the top kernels. Times in the CSV are ns;
numbers may use a decimal comma.
"""
import csv, re, sys
from collections import defaultdict

sys.path.insert(0, __file__.rsplit("\\", 1)[0] if "\\" in __file__ else ".")
from nsys_steps import family  # noqa: E402


def arg(name, default, cast=float):
    return cast(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


def num(s):
    return float(s.replace(",", ".")) if s else 0.0


def union(intervals):
    total = 0; cur_s = cur_e = None
    for s, e in sorted(intervals):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        total += cur_e - cur_s
    return total


def intersect(a, b):
    a = merged(a); b = merged(b); i = j = 0; tot = 0
    while i < len(a) and j < len(b):
        s = max(a[i][0], b[j][0]); e = min(a[i][1], b[j][1])
        if e > s:
            tot += e - s
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return tot


def merged(intervals):
    out = []
    for s, e in sorted(intervals):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def main():
    path = sys.argv[1]; gap = arg("--gap-ms", 200) * 1e6; top = arg("--top", 25, int)
    ev = []
    with open(path, newline="", encoding="utf-8", errors="ignore") as f:
        for r in csv.DictReader(f):
            s = int(r["Start (ns)"]); d = int(r["Duration (ns)"])
            ev.append((s, s + d, r["Name"], num(r["Bytes (MB)"]) * 1e6, r["SrcMemKd"], r["DstMemKd"]))
    ev.sort()
    # longest dense run
    best = (0, 0, 0); run_start = 0; last_end = ev[0][1]
    for i in range(1, len(ev)):
        if ev[i][0] - last_end > gap:
            if last_end - ev[run_start][0] > best[0]:
                best = (last_end - ev[run_start][0], run_start, i)
            run_start = i
        last_end = max(last_end, ev[i][1])
    if last_end - ev[run_start][0] > best[0]:
        best = (last_end - ev[run_start][0], run_start, len(ev))
    w = ev[best[1]:best[2]]
    wall = (max(e for _, e, *_ in w) - w[0][0]) / 1e6
    print(f"window: {len(w)} events, wall {wall:.1f} ms (events {best[1]}..{best[2]} of {len(ev)})")

    kinds = defaultdict(list); kbytes = defaultdict(float); kcount = defaultdict(int)
    fam = defaultdict(float); kern = defaultdict(lambda: [0, 0.0]); kern_iv = []
    for s, e, name, b, src, dst in w:
        if "memcpy" in name:
            k = "H2D" if "Host-to-Device" in name else "D2H" if "Device-to-Host" in name else "D2D" if "Device-to-Device" in name else name
            kinds[k].append((s, e)); kbytes[k] += b; kcount[k] += 1
        elif "memset" in name:
            kinds["memset"].append((s, e)); kcount["memset"] += 1
        else:
            kern_iv.append((s, e)); fam[family(name)] += (e - s) / 1e6
            kern[name][0] += 1; kern[name][1] += (e - s) / 1e6
    busy_all = union([(s, e) for s, e, *_ in w]) / 1e6
    kbusy = union(kern_iv) / 1e6
    print(f"GPU busy (union of everything) {busy_all:.1f} ms = {100*busy_all/wall:.0f}% of wall; kernels busy {kbusy:.1f} ms")
    for k in ("H2D", "D2D", "D2H", "memset"):
        if kcount[k]:
            bz = union(kinds[k]) / 1e6
            gb = kbytes[k] / 1e9
            bw = f", {gb/(bz/1e3):.1f} GB/s effective" if gb and bz else ""
            print(f"{k:6}: {kcount[k]:6} copies, {gb:7.2f} GB, busy {bz:8.1f} ms ({100*bz/wall:.0f}% of wall){bw}")
    if kinds["H2D"]:
        ov = intersect(kinds["H2D"], kern_iv) / 1e6
        print(f"H2D overlapped with kernels: {ov:.1f} ms; kernels alone {kbusy-ov:.1f} ms; H2D alone {union(kinds['H2D'])/1e6-ov:.1f} ms; idle {wall-busy_all:.1f} ms")
    if kinds["H2D"]:
        # gaps on the copy engine: how long the H2D stream sits idle between copies, and how many pauses there are
        h = sorted(kinds["H2D"]); gaps = [h[i][0] - h[i-1][1] for i in range(1, len(h))]
        big = [g for g in gaps if g > 1e6]
        print(f"H2D pauses > 1 ms: {len(big)} totalling {sum(big)/1e6:.0f} ms; > 20 ms: {sum(1 for g in gaps if g > 20e6)} totalling {sum(g for g in gaps if g > 20e6)/1e6:.0f} ms")
    bin_ms = arg("--timeline", 0)
    if bin_ms:
        t0 = w[0][0]; nb = int(wall / bin_ms) + 1
        hb = [0.0]*nb; kb = [0.0]*nb; fb = [defaultdict(float) for _ in range(nb)]
        def spread(acc, s, e, f_=None):
            for b in range(int((s-t0)/(bin_ms*1e6)), int((e-t0)/(bin_ms*1e6)) + 1):
                if b >= nb: break
                lo = t0 + b*bin_ms*1e6; hi = lo + bin_ms*1e6
                part = max(0, min(e, hi) - max(s, lo)) / 1e6
                if f_ is None: acc[b] += part
                else: acc[b][f_] += part
        for s, e in merged(kinds["H2D"]): spread(hb, s, e)
        for s, e in merged(kern_iv): spread(kb, s, e)
        for s, e, name, b, src, dst in w:
            if "memcpy" not in name and "memset" not in name: spread(fb, s, e, family(name))
        print(f"\ntimeline ({bin_ms:.0f} ms bins): t, H2D busy %, kernel busy %, top kernel families in the bin")
        for b in range(nb):
            tops = ", ".join(f"{f_[:14]} {t:.0f}" for f_, t in sorted(fb[b].items(), key=lambda x: -x[1])[:3])
            print(f"  {b*bin_ms:7.0f}  {100*hb[b]/bin_ms:5.0f}%  {100*kb[b]/bin_ms:5.0f}%   {tops}")
    print("\nkernel time by family (ms, sum of durations):")
    for f_, t in sorted(fam.items(), key=lambda x: -x[1]):
        print(f"  {t:9.1f}  {f_}")
    print(f"\ntop {top} kernels (count, total ms):")
    for name, (n, t) in sorted(kern.items(), key=lambda x: -x[1][1])[:top]:
        print(f"  {n:7} {t:9.1f}  {name[:110]}")


main()
