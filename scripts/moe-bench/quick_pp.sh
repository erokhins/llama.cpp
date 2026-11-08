#!/usr/bin/env bash
# usage: quick_pp.sh BIN_DIR NAME [extra llama-server args...]
# Prompt-processing check: one ~30k-token prompt (same corpus as bench.py), prints the server's
# prompt eval line and the VRAM in use. Leaves nothing running.
set -u
BINDIR=$1; NAME=$2; shift 2
T=${BENCH_DATA:-C:/Users/erokh/qwen3.8-flash-next}   # data dir: outputs go to $T/builds-bench, draft head in $T/models
D=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); R=$(cd "$D/../.." && pwd)   # script dir, repo root
BIN=$BINDIR/llama-server.exe
M=C:/Users/erokh/AppData/Local/Strata/Strata-data/models/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf
KEY=${BENCH_API_KEY:-sk-local}
OUT=$T/builds-bench; mkdir -p "$OUT"; LOG=$OUT/quickpp-$NAME.log
taskkill //IM llama-server.exe //F >/dev/null 2>&1; sleep 2
PID=$(python -I $D/srv.py start "$LOG" -- "$BIN" -m "$M" --host 127.0.0.1 --port 8080 --api-key $KEY --alias q \
  -c 131072 -np 4 --kv-unified -ctk q8_0 -ctv q8_0 -fa on --jinja -t 16 --load-mode none "$@")
for i in $(seq 1 300); do grep -q "listening on" "$LOG" && break; tasklist //FI "PID eq $PID" | grep -q llama-server || { echo DIED; tail -20 "$LOG"; exit 1; }; sleep 2; done
echo "ready; VRAM $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
python -I - "$KEY" <<'EOF'
import json, pathlib, sys, urllib.request, time
KEY = sys.argv[1]
CORPUS_DIR = pathlib.Path(r"C:\Users\erokh\AppData\Local\Strata\terminal\0.1.55")
corpus = ""
for p in sorted(CORPUS_DIR.rglob("*.py")):
    if "site-packages" in p.parts or "python" == p.parts[len(CORPUS_DIR.parts)]:
        continue
    try: corpus += f"\n# ==== {p.name} ====\n" + p.read_text(encoding="utf-8", errors="ignore")
    except OSError: pass
    if len(corpus) > 400_000: break
prompt = corpus[:118_000] + "\n\nSummarize the code above in one sentence."
body = json.dumps({"model": "q", "messages": [{"role": "user", "content": prompt}], "max_tokens": 16, "temperature": 0}).encode()
req = urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions", data=body, headers={"Authorization": "Bearer " + KEY, "Content-Type": "application/json"})
t = time.time(); d = json.load(urllib.request.urlopen(req, timeout=600)); dt = time.time() - t
tm = d.get("timings", {})
print(f"prompt_n={tm.get('prompt_n')} prompt_ms={tm.get('prompt_ms'):.0f} server_pp={tm.get('prompt_per_second'):.0f} tok/s  client_total={dt:.1f}s")
EOF
python -I $D/srv.py stop $PID 5 2>/dev/null
