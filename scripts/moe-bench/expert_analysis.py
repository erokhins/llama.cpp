"""Compare expert-cache residency against recorded usage.

usage: python -I expert_analysis.py experts.bin resident.bin SLOT_BUDGETS [strata_profile.bin ...] [--oracle-out FILE]

experts.bin   records of int32 {layer, n_used, n_tokens} + ids           (LLAMA_RECORD_EXPERTS)
resident.bin  snapshots of int32 {n} + n x {layer, expert}                 (LLAMA_MOE_CACHE_DUMP)
SLOT_BUDGETS  "layers:slots,..." per cache group, e.g. "0-13:2500,14-18:900" (from the init_device_lookup / cache size log),
              or a single number for one shared budget over all cached layers
strata_profile.bin  Strata STRP v1 files: ranked (layer, expert) pairs; the first SLOTS pairs are its resident set
--oracle-out  write the oracle set (top pairs per group budget, least important first) as int32 pairs for LLAMA_MOE_CACHE_PRELOAD
"""
import struct, sys
import numpy as np

def read_usage(path):
    data = np.fromfile(path, dtype=np.int32)
    i = 0; counts = {}; n_lookups = 0
    while i + 3 <= len(data):
        il, n_used, n_tok = data[i], data[i+1], data[i+2]; i += 3
        ids = data[i:i + n_used*n_tok]; i += n_used*n_tok
        c = counts.setdefault(int(il), np.zeros(512, dtype=np.int64))
        np.add.at(c, ids, 1); n_lookups += len(ids)
    return counts, n_lookups

def read_resident(path):
    data = np.fromfile(path, dtype=np.int32)
    snaps = []; i = 0
    while i < len(data):
        n = data[i]; i += 1
        snaps.append(data[i:i + 2*n].reshape(-1, 2)); i += 2*n
    return snaps

def read_strp(path):
    # STRP v1: magic, 5 x uint32 {version, n_layers, n_expert, slots, n_ranked}, n_ranked x (uint16 layer, uint16 expert), then an int32 rank table
    with open(path, "rb") as f:
        magic = f.read(4); hdr = struct.unpack("<5I", f.read(20))
        assert magic == b"STRP", path
        version, nl, ne, slots, n_ranked = hdr
        pairs = np.frombuffer(f.read(4*n_ranked), dtype=np.uint16).reshape(-1, 2).astype(np.int32)
    return pairs, slots

def coverage(counts, resident_pairs):
    res = {}
    for il, e in resident_pairs: res.setdefault(int(il), set()).add(int(e))
    hit = tot = 0
    for il, c in counts.items():
        tot += c.sum()
        s = res.get(il)
        if s: hit += c[list(s)].sum()
    return hit / tot if tot else 0.0

def parse_budgets(spec, layers):
    if ":" not in spec:
        return [(sorted(layers), int(spec))]
    out = []
    for part in spec.split(","):
        rng, slots = part.split(":")
        if "-" in rng:
            a, b = rng.split("-"); ls = list(range(int(a), int(b)+1))
        else:
            ls = [int(x) for x in rng.split(".")]
        out.append(([l for l in ls if l in layers], int(slots)))
    return out

def oracle(counts, budgets):
    pairs = []
    for ls, slots in budgets:
        cand = [(int(counts[l][e]), l, e) for l in ls for e in range(512) if counts.get(l) is not None and counts[l][e] > 0]
        cand.sort(reverse=True)
        pairs += [(l, e) for _, l, e in cand[:slots]]
    return pairs

def main():
    argv = sys.argv[1:]
    skip = set()
    for i, a in enumerate(argv):
        if a in ("--oracle-out", "--strata-slots"): skip.add(i); skip.add(i + 1)
    args = [a for i, a in enumerate(argv) if i not in skip and not a.startswith("--")]
    oracle_out = sys.argv[sys.argv.index("--oracle-out") + 1] if "--oracle-out" in sys.argv else None
    counts, n_lookups = read_usage(args[0])
    snaps = read_resident(args[1]) if args[1] != "-" else []
    layers = set(counts)
    budgets = parse_budgets(args[2], layers)
    total_slots = sum(s for _, s in budgets)
    print(f"usage: {n_lookups} lookups over {len(layers)} layers; distinct (layer,expert) used: {sum(int((c > 0).sum()) for c in counts.values())}")
    # concentration of the usage
    allc = np.sort(np.concatenate([c for c in counts.values()]))[::-1]
    cum = np.cumsum(allc) / allc.sum()
    for k in (2000, 4000, 6000, 8000, 10000, total_slots):
        if k <= len(cum): print(f"  top {k:5d} (layer,expert) pairs cover {100*cum[k-1]:.1f}% of lookups")
    orc = oracle(counts, budgets)
    print(f"oracle (top pairs per group, {total_slots} slots): coverage {100*coverage(counts, orc):.1f}%")
    for i, s in enumerate(snaps):
        if i in (0, len(snaps)//2, len(snaps)-1):
            print(f"llama.cpp resident snapshot {i} ({len(s)} experts): coverage of the whole run {100*coverage(counts, s):.1f}%")
    strata_slots = int(sys.argv[sys.argv.index("--strata-slots") + 1]) if "--strata-slots" in sys.argv else total_slots
    for p in args[3:]:
        pairs, slots = read_strp(p)
        for k in sorted({strata_slots, total_slots}):
            print(f"strata profile {p.split(chr(92))[-1].split('/')[-1]} first {k} pairs: coverage {100*coverage(counts, pairs[:k]):.1f}%")
        strata_res = set(map(tuple, pairs[:strata_slots].tolist()))
        if snaps:
            llama_res = set(map(tuple, snaps[-1].tolist()))
            print(f"  overlap of Strata's resident set with llama.cpp's final one: {len(strata_res & llama_res)} pairs")
        orc_set = set(orc)
        print(f"  overlap of Strata's resident set with the oracle: {len(strata_res & orc_set)} of {len(orc_set)}")
    if oracle_out:
        arr = np.array(orc[::-1], dtype=np.int32)  # least important first
        arr.tofile(oracle_out); print("oracle written:", oracle_out, len(orc), "pairs")

if __name__ == "__main__":
    main()
