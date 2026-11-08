#!/usr/bin/env bash
# usage: quick_tg.sh BIN_DIR NAME [extra llama-server args...]
# Fast sanity check: start the server, run two 256-token generations, print the server's
# timing lines and the MoE cache statistics (graceful stop), leave nothing running.
set -u
BINDIR=$1; NAME=$2; shift 2
T=${BENCH_DATA:-C:/Users/erokh/qwen3.8-flash-next}   # data dir: outputs go to $T/builds-bench, draft head in $T/models
D=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); R=$(cd "$D/../.." && pwd)   # script dir, repo root
BIN=$BINDIR/llama-server.exe
M=C:/Users/erokh/AppData/Local/Strata/Strata-data/models/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf
KEY=${BENCH_API_KEY:-sk-local}
OUT=$T/builds-bench; mkdir -p "$OUT"; LOG=$OUT/quickhit-$NAME.log
taskkill //IM llama-server.exe //F >/dev/null 2>&1; sleep 2
PID=$(python -I $D/srv.py start "$LOG" -- "$BIN" -m "$M" --host 127.0.0.1 --port 8080 --api-key $KEY --alias q \
  -c 131072 -np 4 --kv-unified -ctk q8_0 -ctv q8_0 -fa on --jinja -t 16 -b 4096 -ub 4096 --load-mode none "$@")
for i in $(seq 1 300); do grep -q "listening on" "$LOG" && break; tasklist //FI "PID eq $PID" | grep -q llama-server || { echo DIED; tail -20 "$LOG"; exit 1; }; sleep 2; done
echo "ready after ${i}x2s; VRAM $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
for n in 1 2 3 4; do
  curl -s --max-time 180 http://127.0.0.1:8080/v1/chat/completions -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
    -d '{"model":"q","messages":[{"role":"user","content":"Write a detailed essay about the history of the printing press. Request '"$n"'"}],"max_tokens":512,"temperature":0}' >/dev/null
done
tasklist //FI "PID eq $PID" | grep -q llama-server || echo "DIED during requests (no assert message: likely a crash)"
grep -E "prompt eval time|eval time =|draft acceptance" "$LOG" | cut -c1-150
python -I $D/srv.py stop $PID 60
grep -E "llama_moe_cache|MoE cache|moe_cache" "$LOG" | tail -8 | cut -c1-200
