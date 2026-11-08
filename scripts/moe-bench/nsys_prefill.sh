#!/usr/bin/env bash
# usage: nsys_prefill.sh NAME [extra llama-server args...]
# Profiles one prompt (PP_CHARS chars of the bench corpus, default 56000 = ~15k tokens) under Nsight Systems and writes
# builds-bench/nsys-NAME.{nsys-rep,sqlite} plus the cuda_gpu_trace CSV; then scripts/nsys_prefill.py prints the
# copy-vs-compute breakdown. Context/batch flags go in the extra args.
set -u
NAME=$1; shift
T=${BENCH_DATA:-C:/Users/erokh/testing}   # data dir: outputs go to $T/builds-bench, draft heads live in $T/mtp
D=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); R=$(cd "$D/../.." && pwd)   # script dir, repo root
BIN=$R/build/bin/llama-server.exe
NSYS=$(find /c/Users/erokh/tools/nsys -path "*target-windows-x64*" -iname nsys.exe | head -1)
M=C:/Users/erokh/AppData/Local/Strata/Strata-data/models/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf
KEY=${BENCH_API_KEY:-sk-local}
OUT=$T/builds-bench; LOG=$OUT/nsys-$NAME.log; SLOG=$OUT/nsys-$NAME.server.log; REP=$OUT/nsys-$NAME
DURATION=${DURATION:-120}
taskkill //IM llama-server.exe //F >/dev/null 2>&1; sleep 2
rm -f "$REP.nsys-rep" "$REP.sqlite" "$SLOG"
PID=$(python -I $D/srv.py start "$LOG" -- "$NSYS" profile --trace=cuda --cuda-graph-trace=node --sample=none --cpuctxsw=none -d $DURATION --kill=true --force-overwrite=true -o "$REP" \
  "$BIN" -m "$M" --host 127.0.0.1 --port 8080 --api-key $KEY --alias q --kv-unified -ctk q8_0 -ctv q8_0 -fa on --jinja -t 16 --load-mode none --log-file "$SLOG" "$@")
for i in $(seq 1 150); do grep -q "listening on" "$SLOG" 2>/dev/null && break; [ $i -gt 5 ] && ! tasklist | grep -qi "llama-server.exe" && break; sleep 2; done
grep -q "listening on" "$SLOG" 2>/dev/null || { echo "server did not start"; tail -n 20 "$SLOG"; exit 1; }
python -I $D/pp_bench.py http://127.0.0.1:8080 $KEY q $OUT/nsys-$NAME.pp.json ${PP_CHARS:-56000}
grep -E "prompt eval time" "$SLOG" | tail -1 | sed -E 's/^[0-9.]+ I //' | cut -c1-150
for i in $(seq 1 200); do [ -f "$REP.nsys-rep" ] && ! tasklist | grep -qi "nsys.exe" && break; sleep 2; done
ls -l "$REP.nsys-rep" 2>/dev/null | awk '{print $5/1048576 " MB report"}'
"$NSYS" stats --report cuda_gpu_trace --format csv --force-export=true -o "$REP" "$REP.nsys-rep" >/dev/null 2>&1
python -I $D/nsys_prefill.py "${REP}_cuda_gpu_trace.csv"
