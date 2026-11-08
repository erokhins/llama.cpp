#!/usr/bin/env bash
# usage: [ENV=..] strata_bench.sh NAME [extra strata args...]
# Strata bench mode (no --serve) on the 89-token teacher-forcing prompt, 600 new tokens, verify profiler on, no width cycling.
set -u
NAME=$1; shift
S=C:/Users/erokh/AppData/Local/Strata
EXE=$S/components/engine/0.1.55-windows-x86_64-cuda-cuda13/strata.exe
D=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); T=${BENCH_DATA:-C:/Users/erokh/qwen3.8-flash-next}
IDS=$(python -I -c "import json,sys;print(','.join(map(str,json.load(open(sys.argv[1]))['prompt_ids'])))" "$D/tf-ref.json")
OUT=$T/builds-bench/strata-bench-$NAME.log
cd "$S/terminal/0.1.55" || exit 1
STRATA_VERIFY_PROFILE=1 "$EXE" --pack "$S/Strata-data/packs/iq3_s" \
  --native "$S/Strata-data/models/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf" \
  --ple-gguf "$S/Strata-data/models/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf" \
  --expert-profile "$S/terminal/0.1.55/data/expert-profile.bin" --expert-cache auto --prefill auto \
  --spec 4 --spec-min-p 0.5 --mtp "$S/Strata-data/mtp/rt" --max-context 131072 --kv int8 --kv-resident 32768 \
  --tokens "$IDS" --max-new ${MAX_NEW:-600} "$@" > "$OUT" 2>&1
echo "exit=$?"
grep -E "^decode |window sizes|window cost|TAIL_SKIP|route tail|CPU experts|hit rate" "$OUT" | cut -c1-260
