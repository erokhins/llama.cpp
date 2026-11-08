#!/usr/bin/env bash
# usage: paw_test.sh BIN_DIR NAME [extra llama-server args...]
# Phase-aware workspace check: load, long prompt, 600-token generation, long prompt again, short generation.
# Prints VRAM in use at each phase, the server's workspace log lines, and prompt/generation speeds.
set -u
BINDIR=$1; NAME=$2; shift 2
T=${BENCH_DATA:-C:/Users/erokh/qwen3.8-flash-next}   # data dir: outputs go to $T/builds-bench, draft head in $T/models
D=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); R=$(cd "$D/../.." && pwd)   # script dir, repo root
BIN=$BINDIR/llama-server.exe
M=C:/Users/erokh/AppData/Local/Strata/Strata-data/models/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf
KEY=${BENCH_API_KEY:-sk-local}
OUT=$T/builds-bench; mkdir -p "$OUT"; LOG=$OUT/paw-$NAME.log
vram(){ nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' '; }
taskkill //IM llama-server.exe //F >/dev/null 2>&1; sleep 2
PID=$(python -I $D/srv.py start "$LOG" -- "$BIN" -m "$M" --host 127.0.0.1 --port 8080 --api-key $KEY --alias q \
  --kv-unified -ctk q8_0 -ctv q8_0 -fa on --jinja -t 16 --load-mode none "$@")
for i in $(seq 1 300); do grep -q "listening on" "$LOG" && break; tasklist //FI "PID eq $PID" | grep -q llama-server || { echo DIED; tail -20 "$LOG"; exit 1; }; sleep 2; done
echo "loaded:            VRAM $(vram) MiB"
python -I - "$KEY" <<'EOF'
import json, pathlib, sys, urllib.request, time, subprocess, threading
KEY = sys.argv[1]
CORPUS_DIR = pathlib.Path(r"C:\Users\erokh\AppData\Local\Strata\terminal\0.1.55")
corpus = ""
for p in sorted(CORPUS_DIR.rglob("*.py")):
    if "site-packages" in p.parts or "python" == p.parts[len(CORPUS_DIR.parts)]:
        continue
    try: corpus += f"\n# ==== {p.name} ====\n" + p.read_text(encoding="utf-8", errors="ignore")
    except OSError: pass
    if len(corpus) > 400_000: break
def vram():
    return subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip()
def chat(prompt, max_tokens):
    body = json.dumps({"model": "q", "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "temperature": 0}).encode()
    req = urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions", data=body, headers={"Authorization": "Bearer " + KEY, "Content-Type": "application/json"})
    t = time.time(); d = json.load(urllib.request.urlopen(req, timeout=900)); dt = time.time() - t
    tm = d.get("timings", {})
    return tm, dt
samples = []
def sampler(stop):
    while not stop.is_set():
        samples.append(int(vram())); time.sleep(1.0)
for rnd in (1, 2):
    tm, dt = chat(corpus[:118_000] + f"\n\nSummarize the code above in one sentence. ({rnd})", 16)
    print(f"prompt {rnd}:          {tm.get('prompt_n')} tokens, server pp {tm.get('prompt_per_second'):.0f} tok/s, VRAM {vram()} MiB")
    stop = threading.Event(); samples.clear(); th = threading.Thread(target=sampler, args=(stop,)); th.start()
    tm, dt = chat(f"Write a detailed essay about the history of the printing press. ({rnd})", 600)
    stop.set(); th.join()
    print(f"generation {rnd}:      {tm.get('predicted_n')} tokens at {tm.get('predicted_per_second'):.1f} tok/s; VRAM during generation min {min(samples)} max {max(samples)} MiB, after {vram()} MiB")
EOF
python -I $D/srv.py stop $PID 5 2>/dev/null
echo "== workspace log lines"; grep -E "phase-aware workspace|sched_reserve_decode|compute buffer size|reserve took" "$LOG" | sed -E 's/^[0-9.]+ I +//' | tail -12 | cut -c1-150
