#!/usr/bin/env bash
# usage: nsys_strata_pp.sh NAME
# Runs the Strata engine server (same args as the launch profile) under Nsight Systems, sends one prompt of PP_CHARS
# chars (default 56000, ~15k tokens) with scripts/pp_bench.py, then prints the copy-vs-compute breakdown with
# scripts/nsys_prefill.py. Output: builds-bench/nsys-NAME.*
set -u
NAME=$1
T=${BENCH_DATA:-C:/Users/erokh/testing}   # data dir: outputs go to $T/builds-bench, draft heads live in $T/mtp
D=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); R=$(cd "$D/../.." && pwd)   # script dir, repo root
S=C:/Users/erokh/AppData/Local/Strata
EXE=$S/components/engine/0.1.55-windows-x86_64-cuda-cuda13/strata.exe
NSYS=$(find /c/Users/erokh/tools/nsys -path "*target-windows-x64*" -iname nsys.exe | head -1)
KEY=${BENCH_API_KEY:-sk-local}
OUT=$T/builds-bench; LOG=$OUT/nsys-$NAME.log; REP=$OUT/nsys-$NAME
DURATION=${DURATION:-150}
taskkill //IM llama-server.exe //F >/dev/null 2>&1; taskkill //IM strata.exe //F >/dev/null 2>&1; sleep 2
rm -f "$REP.nsys-rep" "$REP.sqlite"
cd "$S/terminal/0.1.55" || exit 1
PID=$(python -I $D/srv.py start "$LOG" -- "$NSYS" profile --trace=cuda --cuda-graph-trace=node --sample=none --cpuctxsw=none -d $DURATION --kill=true --force-overwrite=true -o "$REP" \
  "$EXE" --serve --pack "$S/Strata-data/packs/iq3_s" --native "$S/Strata-data/models/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf" \
  --ple-gguf "$S/Strata-data/models/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf" --expert-profile "$S/terminal/0.1.55/data/expert-profile.bin" \
  --expert-cache auto --spec 4 --spec-min-p 0.5 --mtp "$S/Strata-data/mtp/rt" --max-context 131072 --kv int8 --kv-resident 32768 --prefill auto:32768 --batch 4)
for i in $(seq 1 120); do curl -s -o /dev/null -m 2 http://127.0.0.1:8080/v1/models && break; sleep 2; done
echo "server up after ${i}x2s"; curl -s -m 5 http://127.0.0.1:8080/v1/models | head -c 300; echo
python -I $D/pp_bench.py http://127.0.0.1:8080 $KEY ${MODEL:-qwen3.8-flash-next-iq3_s} $OUT/nsys-$NAME.pp.json ${PP_CHARS:-56000}
for i in $(seq 1 200); do [ -f "$REP.nsys-rep" ] && ! tasklist | grep -qi "nsys.exe" && break; sleep 2; done
taskkill //IM strata.exe //F >/dev/null 2>&1
ls -l "$REP.nsys-rep" 2>/dev/null | awk '{print $5/1048576 " MB report"}'
"$NSYS" stats --report cuda_gpu_trace --format csv --force-export=true -o "$REP" "$REP.nsys-rep" >/dev/null 2>&1
python -I $D/nsys_prefill.py "${REP}_cuda_gpu_trace.csv" --top 30
