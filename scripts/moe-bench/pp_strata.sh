#!/usr/bin/env bash
# usage: [PP_SIZES=56000,112000,...] pp_strata.sh NAME [extra strata args...]
# Prefill benchmark of the Strata engine server with the launch-profile args (same as nsys_strata_pp.sh, no profiler),
# measured by scripts/pp_bench.py on the same prompts as pp_run.sh. Extra args go last (e.g. --prefill 8192 overrides
# the default --prefill auto:32768). Output: builds-bench/pp-strata-NAME.{json,log}
set -u
NAME=$1; shift
T=${BENCH_DATA:-C:/Users/erokh/qwen3.8-flash-next}   # data dir: outputs go to $T/builds-bench, draft head in $T/models
D=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); R=$(cd "$D/../.." && pwd)   # script dir, repo root
S=C:/Users/erokh/AppData/Local/Strata
EXE=$S/components/engine/0.1.55-windows-x86_64-cuda-cuda13/strata.exe
KEY=${BENCH_API_KEY:-sk-local}
OUT=$T/builds-bench; LOG=$OUT/pp-strata-$NAME.log
taskkill //IM llama-server.exe //F >/dev/null 2>&1; taskkill //IM strata.exe //F >/dev/null 2>&1; sleep 2
cd "$S/terminal/0.1.55" || exit 1
PID=$(python -I $D/srv.py start "$LOG" -- "$EXE" --serve --pack "$S/Strata-data/packs/iq3_s" \
  --native "$S/Strata-data/models/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf" \
  --ple-gguf "$S/Strata-data/models/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf" \
  --expert-profile "$S/terminal/0.1.55/data/expert-profile.bin" \
  --expert-cache auto --spec 4 --spec-min-p 0.5 --mtp "$S/Strata-data/mtp/rt" --max-context 131072 --kv int8 --kv-resident 32768 \
  --prefill auto:32768 --batch 4 "$@")
# the launcher process exits once the engine is up (so no PID check): wait for the HTTP endpoint, give up after 10 min
for i in $(seq 1 300); do curl -s -o /dev/null -m 2 http://127.0.0.1:8080/v1/models && break; sleep 2; done
curl -s -o /dev/null -m 2 http://127.0.0.1:8080/v1/models || { echo "DIED (no HTTP after ${i}x2s)"; tail -20 "$LOG"; taskkill //IM strata.exe //F >/dev/null 2>&1; exit 1; }
echo "ready after ${i}x2s; VRAM $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
python -I $D/pp_bench.py http://127.0.0.1:8080 $KEY ${MODEL:-qwen3.8-flash-next-iq3_s} $OUT/pp-strata-$NAME.json ${PP_SIZES:-56000,112000,56000,112000} 2>&1 | grep -vE "^$"
taskkill //IM strata.exe //F >/dev/null 2>&1; sleep 2
grep -i -E "prefill|chunk|prompt" "$LOG" | grep -v -i "warning" | head -8 | cut -c1-200
