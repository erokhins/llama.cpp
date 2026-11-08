#!/usr/bin/env bash
# usage: [ENV=..] llama_bench_tf.sh NAME [extra llama-server args...]
# llama.cpp counterpart of strata_bench.sh: the 89-token teacher-forcing prompt as raw token ids, 600 new tokens, greedy,
# 20 GB layout with MTP-4 fixed width. Prints the server's timing lines and the cache statistics.
set -u
NAME=$1; shift
T=${BENCH_DATA:-C:/Users/erokh/qwen3.8-flash-next}   # data dir: outputs go to $T/builds-bench, draft head in $T/models
D=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); R=$(cd "$D/../.." && pwd)   # script dir, repo root
BIN=${BIN:-$R/build/bin/llama-server.exe}   # BIN=... runs another build (A/B)
M=C:/Users/erokh/AppData/Local/Strata/Strata-data/models/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf
KEY=${BENCH_API_KEY:-sk-local}
LOG=$T/builds-bench/llama-bench-$NAME.log
taskkill //IM llama-server.exe //F >/dev/null 2>&1; sleep 1
PID=$(python -I $D/srv.py start "$LOG" -- "$BIN" -m "$M" --host 127.0.0.1 --port 8080 --api-key $KEY --alias q \
  -c 32768 -np 2 --kv-unified -ctk q8_0 -ctv q8_0 -fa on --jinja -t 16 -b 2048 -ub 2048 --load-mode none -lv 4 --moe-cache-mib 24576 \
  -md $T/models/mtp-Qwen3.8-Flash-Next-Q8_0.gguf --spec-type draft-mtp --spec-draft-n-max 4 --spec-draft-n-ubatch 64 "$@")
for i in $(seq 1 150); do grep -q "listening on" "$LOG" && break; sleep 2; done
python -I - "$KEY" "${MAX_NEW:-600}" "$D/tf-ref.json" <<'PY'
import json, sys, urllib.request
key, n = sys.argv[1], int(sys.argv[2])
ids = json.load(open(sys.argv[3]))["prompt_ids"]
body = json.dumps({"prompt": ids, "n_predict": n, "temperature": 0, "top_k": 1, "cache_prompt": False}).encode()
req = urllib.request.Request("http://127.0.0.1:8080/completion", data=body, headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
d = json.load(urllib.request.urlopen(req, timeout=900)); t = d.get("timings", {})
print(f"  {t.get('predicted_n')} tokens at {t.get('predicted_per_second'):.1f} tok/s (prompt {t.get('prompt_n')} tokens)")
PY
python -I $D/srv.py stop $PID 10 >/dev/null 2>&1
grep -E "eval time =|draft acceptance|device lookups" "$LOG" | grep -v "prompt eval" | tail -4 | sed -E 's/^[0-9.]+ I //' | cut -c1-170
