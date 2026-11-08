#!/usr/bin/env bash
# usage: run_bin.sh BIN_DIR NAME [extra llama-server args...]
# Takes the directory holding llama-server.exe, benchmarks over
# 127.0.0.1 (not localhost), writes results-NAME.json / server-NAME.log into ./builds-bench/,
# and stops the server with a graceful shutdown so the MoE cache prints its hit statistics.
set -u
BINDIR=$1; NAME=$2; shift 2
T=${BENCH_DATA:-C:/Users/erokh/qwen3.8-flash-next}   # data dir: outputs go to $T/builds-bench, draft head in $T/models
D=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); R=$(cd "$D/../.." && pwd)   # script dir, repo root
BIN=$BINDIR/llama-server.exe
M=C:/Users/erokh/AppData/Local/Strata/Strata-data/models/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf
KEY=${BENCH_API_KEY:-sk-local}
OUT=$T/builds-bench; mkdir -p "$OUT"
LOG=$OUT/server-$NAME.log

taskkill //IM llama-server.exe //F >/dev/null 2>&1; sleep 3

PID=$(python -I $D/srv.py start "$LOG" -- "$BIN" -m "$M" --host 127.0.0.1 --port 8080 --api-key $KEY --alias qwen3.8-flash-next-iq3_s \
  -c 131072 -np 4 --kv-unified -ctk q8_0 -ctv q8_0 -fa on --jinja -t 16 \
  -b 4096 -ub 4096 --load-mode none "$@")

for i in $(seq 1 600); do
  if grep -q "listening on" "$LOG"; then break; fi
  if ! tasklist //FI "PID eq $PID" | grep -q llama-server; then echo "SERVER DIED"; tail -25 "$LOG"; exit 1; fi
  sleep 2
done
grep -q "listening on" "$LOG" || { echo "TIMEOUT"; tail -25 "$LOG"; exit 1; }
echo "ready; VRAM $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
grep -i -E "moe.cache|prefetch|mtp|draft|spec" "$LOG" | head -8

python -I $D/bench.py http://127.0.0.1:8080 $KEY qwen3.8-flash-next-iq3_s $OUT/results-$NAME.json 2>&1 | grep -E "SUMMARY|Error|Traceback"
grep -E "draft acceptance" "$LOG" | tail -2

# graceful stop: Ctrl-C equivalent via taskkill without /F, then wait for the stats to be logged
python -I $D/srv.py stop $PID 60
grep -E "llama_moe_cache|MoE cache" "$LOG" | tail -6
