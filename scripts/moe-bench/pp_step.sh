#!/usr/bin/env bash
# usage: [CONFIGS="bestgen c128k"] [PP_SIZES=56000,112000,112000] pp_step.sh STEP [extra llama-server args...]
# Prefill measurement of the current build of this repo after one porting step, on the two reference configs:
#   bestgen: -c 32768 -np 2 -b 2048 -ub 2048, 24 GB expert cache, MTP-3 draft (the generation-first layout)
#   c128k:   -c 131072 -np 4 -b 4096 -ub 4096, 12 GB expert cache
# Output: builds-bench/pp-STEP-CONFIG.{json,log}; prints one line per prompt (server tok/s).
set -u
STEP=$1; shift
T=${BENCH_DATA:-C:/Users/erokh/testing}   # data dir: outputs go to $T/builds-bench, draft heads live in $T/mtp
D=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); R=$(cd "$D/../.." && pwd)   # script dir, repo root
B=$R/build/bin
export PP_SIZES=${PP_SIZES:-56000,112000,112000}
MTP="-md $T/mtp/mtp-Qwen3.8-Flash-Next-Q8_0.gguf --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-n-ubatch 64 --backend-sampling --spec-draft-backend-sampling"
for cfg in ${CONFIGS:-bestgen c128k}; do
  echo "=== $STEP $cfg"
  case $cfg in
    bestgen) bash $D/pp_run.sh $B $STEP-$cfg -c 32768 -np 2 -b 2048 -ub 2048 --moe-cache-mib 24576 $MTP "$@" 2>&1 | grep -E "^pp |ready|DIED" ;;
    bestgen-nomtp) bash $D/pp_run.sh $B $STEP-$cfg -c 32768 -np 2 -b 2048 -ub 2048 --moe-cache-mib 24576 "$@" 2>&1 | grep -E "^pp |ready|DIED" ;;
    c128k)   bash $D/pp_run.sh $B $STEP-$cfg -c 131072 -np 4 -b 4096 -ub 4096 --moe-cache-mib 12288 "$@" 2>&1 | grep -E "^pp |ready|DIED" ;;
  esac
done
taskkill //IM llama-server.exe //F >/dev/null 2>&1
echo "=== done $STEP"
