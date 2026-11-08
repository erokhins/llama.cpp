"""Prefill benchmark of the Strata engine driven over its stdin protocol (no HTTP server, no launch profile).

usage: python -I strata_pp_engine.py NAME [sizes_chars,...] [-- extra engine args, e.g. --prefill 8192]

Builds the same prompts as pp_bench.py (bench corpus, unique nonce), tokenizes them with llama-tokenize (same GGUF
vocabulary), starts strata.exe --serve with the launch profile's args plus the extra args, with STRATA_PREFILL_TIMING=1
so the engine prints its per-phase GPU timeline per prompt on stderr, then sends each prompt as "GEN 1 <ids>" and
reads the PP progress lines. Output: builds-bench/strata-pp-NAME.json and builds-bench/strata-pp-NAME.engine.log
(stderr: "strata prefill timing: ..." lines). Prints one line per prompt: tokens, wall ms, tok/s (engine's PP mean), and
the engine's timing line.
"""
import json, os, pathlib, subprocess, sys, time, uuid

T = pathlib.Path(os.environ.get("BENCH_DATA", r"C:\Users\erokh\qwen3.8-flash-next"))
S = pathlib.Path(r"C:\Users\erokh\AppData\Local\Strata")
EXE = S / r"components\engine\0.1.55-windows-x86_64-cuda-cuda13\strata.exe"
MODEL = S / r"Strata-data\models\IQ3_S\Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf"
TOKENIZE = pathlib.Path(__file__).resolve().parents[2] / "build" / "bin" / "llama-tokenize.exe"
ARGS = ["--serve", "--pack", str(S / r"Strata-data\packs\iq3_s"), "--native", str(MODEL),
        "--ple-gguf", str(S / r"Strata-data\models\IQ3_S\Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf"),
        "--expert-profile", str(S / r"terminal\0.1.55\data\expert-profile.bin"),
        "--expert-cache", "auto", "--spec", "4", "--spec-min-p", "0.5", "--mtp", str(S / r"Strata-data\mtp\rt"),
        "--max-context", "131072", "--kv", "int8", "--kv-resident", "32768", "--prefill", "auto:32768"]
CORPUS_DIR = S / r"terminal\0.1.55"

argv = sys.argv[1:]
extra = []
if "--" in argv:
    i = argv.index("--"); extra = argv[i + 1:]; argv = argv[:i]
NAME = argv[0]
SIZES = [int(x) for x in argv[1].split(",")] if len(argv) > 1 else [56_000, 112_000, 56_000, 112_000]
OUT = T / "builds-bench" / f"strata-pp-{NAME}.json"
LOG = T / "builds-bench" / f"strata-pp-{NAME}.engine.log"

corpus = ""
for p in sorted(CORPUS_DIR.rglob("*.py")):
    if "site-packages" in p.parts or "python" == p.parts[len(CORPUS_DIR.parts)]:
        continue
    try:
        corpus += f"\n# ==== {p.name} ====\n" + p.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        pass
    if len(corpus) > 400_000:
        break


def tokenize(text):
    tmp = T / "builds-bench" / f"strata-pp-{NAME}.prompt.txt"
    tmp.write_text(text, encoding="utf-8")
    out = subprocess.run([str(TOKENIZE), "-m", str(MODEL), "-f", str(tmp), "--ids", "--log-disable"],
                         capture_output=True, text=True, timeout=300).stdout
    line = [l for l in out.splitlines() if l.startswith("[")][-1]
    return json.loads(line)


nonce = lambda: f"[session {uuid.uuid4()}]\n"
prompts = [tokenize(nonce() + "Here is a codebase:\n" + corpus[:c] + "\n\nSummarize what this code does in one sentence.")
           for c in SIZES]
warm = tokenize(nonce() + "Say hello.")
print(f"prompts tokenized: {[len(p) for p in prompts]}", flush=True)

env = dict(os.environ); env["STRATA_PREFILL_TIMING"] = "1"
log = open(LOG, "w", encoding="utf-8")
proc = subprocess.Popen([str(EXE)] + ARGS + extra, cwd=str(CORPUS_DIR), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                        stderr=log, text=True, encoding="utf-8", bufsize=1, env=env)
t0 = time.time()
while True:
    line = proc.stdout.readline()
    if not line:
        sys.exit(f"engine exited before READY (see {LOG})")
    if line.startswith("READY"):
        print(f"READY in {time.time() - t0:.0f} s; extra args: {' '.join(extra)}", flush=True); break


def gen(ids):
    proc.stdin.write(f"GEN 1 {','.join(map(str, ids))}\n"); proc.stdin.flush()
    t0 = time.time(); pp = []
    while True:
        line = proc.stdout.readline()
        if not line:
            sys.exit("engine died mid-run")
        if line.startswith("PP "):
            pp.append(line.split())
        elif line.startswith("ERR"):
            sys.exit(f"engine error: {line.strip()}")
        elif line.startswith("DONE"):
            break
    wall = time.time() - t0
    last = pp[-1] if pp else []
    return {"prompt_tokens": len(ids), "wall_s": round(wall, 3), "client_pp_tps": round(len(ids) / wall, 1),
            "pp_read": int(last[1]) if len(last) > 1 else None,
            "pp_ms": float(last[3]) if len(last) > 3 else None,
            "pp_tps_mean": float(last[4]) if len(last) > 4 else None, "pp_lines": len(pp)}


gen(warm)
log.flush(); off = LOG.stat().st_size
res = []
for ids in prompts:
    r = gen(ids)
    log.flush(); time.sleep(0.3)
    txt = LOG.read_text(encoding="utf-8", errors="ignore")[off:]; off = LOG.stat().st_size
    tl = [l for l in txt.splitlines() if l.startswith("strata prefill timing")]
    r["timing"] = tl
    print(f"pp n={r['prompt_tokens']:>6}  wall {r['wall_s']*1000:.0f} ms = {r['client_pp_tps']} tok/s;"
          f" engine PP mean {r['pp_tps_mean']} tok/s ({r['pp_lines']} chunks)", flush=True)
    for l in tl:
        print("   " + l[:400], flush=True)
    res.append(r)
try:
    proc.stdin.write("QUIT\n"); proc.stdin.flush(); proc.stdin.close(); proc.wait(timeout=60)
except Exception:
    proc.kill()
OUT.write_text(json.dumps({"extra": extra, "started": time.strftime("%Y-%m-%d %H:%M:%S"), "prefill": res}, indent=1))
