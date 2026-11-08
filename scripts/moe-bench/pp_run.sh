#!/usr/bin/env bash
# usage: pp_run.sh BIN_DIR NAME [extra llama-server args...]
# Prefill benchmark of a llama-server build: starts the server with the base flags (context/batch flags must be
# given in the extra args), runs scripts/pp_bench.py (1k/4k/15k/30k/30k prompts, unique prefixes), prints the
# server's prompt timing lines, stops the server gracefully. Output: builds-bench/pp-NAME.json, builds-bench/pp-NAME.log
set -u
BINDIR=$1; NAME=$2; shift 2
T=${BENCH_DATA:-C:/Users/erokh/testing}   # data dir: outputs go to $T/builds-bench, draft heads live in $T/mtp
D=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); R=$(cd "$D/../.." && pwd)   # script dir, repo root
BIN=$BINDIR/llama-server.exe
M=C:/Users/erokh/AppData/Local/Strata/Strata-data/models/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf
KEY=${BENCH_API_KEY:-sk-local}
OUT=$T/builds-bench; mkdir -p "$OUT"; LOG=$OUT/pp-$NAME.log
taskkill //IM llama-server.exe //F >/dev/null 2>&1; sleep 2
PID=$(python -I $D/srv.py start "$LOG" -- "$BIN" -m "$M" --host 127.0.0.1 --port 8080 --api-key $KEY --alias q \
  --kv-unified -ctk q8_0 -ctv q8_0 -fa on --jinja -t 16 --load-mode none "$@")
for i in $(seq 1 300); do grep -q "listening on" "$LOG" && break; tasklist //FI "PID eq $PID" | grep -q llama-server || { echo DIED; tail -20 "$LOG"; exit 1; }; sleep 2; done
echo "ready after ${i}x2s; VRAM $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
python -I $D/pp_bench.py http://127.0.0.1:8080 $KEY q $OUT/pp-$NAME.json ${PP_SIZES:-} 2>&1 | grep -vE "^$"
python -I $D/srv.py stop $PID 30 2>/dev/null
grep -E "prompt eval time" "$LOG" | sed -E 's/^[0-9.]+ I //' | cut -c1-140
