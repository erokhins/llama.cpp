# Byte-compare every tensor Strata copied into dense.bin against the same tensor in the GGUF.
import struct, sys, hashlib
gg, idx, pool = sys.argv[1:4]
f = open(gg, 'rb')
def r(fmt): return struct.unpack('<'+fmt, f.read(struct.calcsize('<'+fmt)))[0]
def s(): return f.read(r('Q')).decode()
def skip(t):
    if t == 8: s(); return
    if t == 9:
        at = r('I'); n = r('Q')
        for _ in range(n): skip(at)
        return
    f.read({0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}[t])
assert f.read(4) == b'GGUF'; r('I'); nt = r('Q'); nkv = r('Q')
align = 32
for _ in range(nkv):
    k = s(); t = r('I')
    if k == 'general.alignment': align = r('I')
    else: skip(t)
info = {}
for _ in range(nt):
    n = s(); nd = r('I'); [r('Q') for _ in range(nd)]; r('I'); info[n] = r('Q')
base = (f.tell() + align - 1) // align * align
p = open(pool, 'rb'); ok = bad = 0
for line in open(idx):
    if line.startswith('#'): continue
    c = line.split(); name, kind, off, size = c[0], int(c[2]), int(c[3]), int(c[4])
    if kind == 0 or size == 0: continue   # kind 0 = read natively from the GGUF
    f.seek(base + info[name]); a = f.read(size)
    p.seek(off); b = p.read(size)
    if a == b: ok += 1
    else: bad += 1; print('MISMATCH', name)
print(f'copied tensors identical: {ok}, mismatched: {bad}')
