#!/usr/bin/env bash
# usage: nsys_decode.sh NAME [extra llama-server args...]
# Profiles llama-server under Nsight Systems (CUDA kernel trace, kernels inside CUDA graphs resolved per node) for a
# short decode workload: 2 requests x 256 tokens in the 20 GB layout. The session ends after DURATION seconds and nsys
# writes builds-bench/nsys-NAME.nsys-rep; then the kernel summary and a steady-state per-step breakdown are printed.
set -u
NAME=$1; shift
T=${BENCH_DATA:-C:/Users/erokh/qwen3.8-flash-next}   # data dir: outputs go to $T/builds-bench, draft head in $T/models
D=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); R=$(cd "$D/../.." && pwd)   # script dir, repo root
BIN=$R/build/bin/llama-server.exe
NSYS=$(find /c/Users/erokh/tools/nsys -path "*target-windows-x64*" -iname nsys.exe | head -1)
M=C:/Users/erokh/AppData/Local/Strata/Strata-data/models/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf
KEY=${BENCH_API_KEY:-sk-local}
OUT=$T/builds-bench; LOG=$OUT/nsys-$NAME.log; SLOG=$OUT/nsys-$NAME.server.log; REP=$OUT/nsys-$NAME
DURATION=${DURATION:-110}
taskkill //IM llama-server.exe //F >/dev/null 2>&1; sleep 2
rm -f "$REP.nsys-rep" "$REP.sqlite" "$SLOG"
PID=$(python -I $D/srv.py start "$LOG" -- "$NSYS" profile --trace=cuda --cuda-graph-trace=node --sample=none --cpuctxsw=none -d $DURATION --kill=true --force-overwrite=true -o "$REP" \
  "$BIN" -m "$M" --host 127.0.0.1 --port 8080 --api-key $KEY --alias q -c 32768 -np 2 --kv-unified -ctk q8_0 -ctv q8_0 -fa on --jinja -t 16 -b 2048 -ub 2048 --load-mode none --log-file "$SLOG" "$@")
for i in $(seq 1 150); do grep -q "listening on" "$SLOG" 2>/dev/null && break; [ $i -gt 5 ] && ! tasklist | grep -qi "llama-server.exe" && break; sleep 2; done
grep -q "listening on" "$SLOG" 2>/dev/null || { echo "server did not start"; tail -n 20 "$SLOG"; exit 1; }
for n in 1 2; do
  curl -s --max-time 180 http://127.0.0.1:8080/v1/chat/completions -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
    -d '{"model":"q","messages":[{"role":"user","content":"Write a detailed essay about the history of the printing press. Request '"$n"'"}],"max_tokens":256,"temperature":0}' >/dev/null
done
grep -E "eval time =" "$SLOG" | tail -1 | cut -c1-150
# wait for nsys to end the session and write the report
for i in $(seq 1 200); do [ -f "$REP.nsys-rep" ] && ! tasklist | grep -qi "nsys.exe" && break; sleep 2; done
ls -l "$REP.nsys-rep" 2>/dev/null | awk '{print $5/1048576 " MB report"}'
"$NSYS" stats --report cuda_gpu_kern_sum --format csv --force-export=true -o "$REP" "$REP.nsys-rep" >/dev/null 2>&1
"$NSYS" stats --report cuda_gpu_trace --format csv --force-export=true -o "$REP" "$REP.nsys-rep" >/dev/null 2>&1
ls "$OUT" | grep "nsys-$NAME"
