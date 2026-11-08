#!/usr/bin/env bash
# usage: mtp_trace.sh BIN_DIR NAME [extra llama-server args...]  (same three prompts as expert_trace.sh, no recorder; prints speed, acceptance, cache statistics)
# Runs the three bench.py decode prompts (1024 tokens each) with expert recording on:
#   LLAMA_RECORD_EXPERTS -> builds-bench/experts-NAME.bin   (routed expert ids per layer per token)
#   LLAMA_MOE_CACHE_DUMP -> builds-bench/resident-NAME.bin  (resident set, appended at every hit-rate log)
set -u
BINDIR=$1; NAME=$2; shift 2
T=${BENCH_DATA:-C:/Users/erokh/testing}   # data dir: outputs go to $T/builds-bench, draft heads live in $T/mtp
D=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); R=$(cd "$D/../.." && pwd)   # script dir, repo root
BIN=$BINDIR/llama-server.exe
M=C:/Users/erokh/AppData/Local/Strata/Strata-data/models/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf
KEY=${BENCH_API_KEY:-sk-local}
OUT=$T/builds-bench; mkdir -p "$OUT"; LOG=$OUT/mtptrace-$NAME.log

taskkill //IM llama-server.exe //F >/dev/null 2>&1; sleep 2
PID=$(python -I $D/srv.py start "$LOG" -- "$BIN" -m "$M" --host 127.0.0.1 --port 8080 --api-key $KEY --alias q \
  -c 32768 -np 2 --kv-unified -ctk q8_0 -ctv q8_0 -fa on --jinja -t 16 -b 2048 -ub 2048 --load-mode none -lv 4 "$@")
for i in $(seq 1 300); do grep -q "listening on" "$LOG" && break; tasklist //FI "PID eq $PID" | grep -q llama-server || { echo DIED; tail -20 "$LOG"; exit 1; }; sleep 2; done
echo "ready; VRAM $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
python -I - "$KEY" <<'EOF'
import json, sys, time, urllib.request
KEY = sys.argv[1]
PROMPTS = [
    "Explain in detail how a B-tree index works in a relational database, including insertion, deletion and node splitting. Use examples.",
    "Write a complete Python implementation of an LRU cache with O(1) get and put, with type hints, docstrings and unit tests.",
    "Describe the history of the Roman Empire from Augustus to the fall of the Western Empire, covering the main emperors and crises.",
]
for p in PROMPTS:
    body = json.dumps({"model": "q", "messages": [{"role": "user", "content": p}], "max_tokens": 1024, "temperature": 0}).encode()
    req = urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions", data=body, headers={"Authorization": "Bearer " + KEY, "Content-Type": "application/json"})
    t = time.time(); d = json.load(urllib.request.urlopen(req, timeout=900)); dt = time.time() - t
    tm = d.get("timings", {})
    print(f"  {tm.get('predicted_n')} tokens at {tm.get('predicted_per_second'):.1f} tok/s")
EOF
python -I $D/srv.py stop $PID 5 2>/dev/null
grep -E "eval time =|draft acceptance" "$LOG" | grep -v "prompt eval" | sed -E 's/^[0-9.]+ I slot print_timing: //' | cut -c1-150
grep -E "llama_moe_cache: device|MoE cache size" "$LOG" | tail -2 | sed -E 's/^[0-9.]+ I //' | cut -c1-160
