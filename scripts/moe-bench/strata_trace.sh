#!/usr/bin/env bash
# Runs the three bench.py decode prompts (1024 tokens) against Strata started with --expert-profile-save, then stops it
# gracefully so the learned profile (resident set first, ranked by heat) is written to builds-bench/strata-profile-after.bin.
set -u
PY=C:/Users/erokh/AppData/Local/Strata/terminal/current/python/python.exe
S=C:/Users/erokh/AppData/Local/Strata/terminal/current/strata.py
KEY=${BENCH_API_KEY:-sk-local}   # Strata server: set BENCH_API_KEY to its key
T=${BENCH_DATA:-C:/Users/erokh/testing}
rm -f $T/builds-bench/strata-profile-after.bin
taskkill //IM llama-server.exe //F >/dev/null 2>&1
"$PY" "$S" runtime start --profile launch-iq3_s-trace --wait 600 2>&1 | tail -1
python -I - "$KEY" <<'PYEOF'
import json, sys, time, urllib.request
KEY = sys.argv[1]
PROMPTS = [
    "Explain in detail how a B-tree index works in a relational database, including insertion, deletion and node splitting. Use examples.",
    "Write a complete Python implementation of an LRU cache with O(1) get and put, with type hints, docstrings and unit tests.",
    "Describe the history of the Roman Empire from Augustus to the fall of the Western Empire, covering the main emperors and crises.",
]
for p in PROMPTS:
    body = json.dumps({"model": "qwen3.8-flash-next-iq3_s", "messages": [{"role": "user", "content": p}], "max_tokens": 1024, "temperature": 0}).encode()
    req = urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions", data=body, headers={"Authorization": "Bearer " + KEY, "Content-Type": "application/json"})
    t = time.time(); d = json.load(urllib.request.urlopen(req, timeout=900)); dt = time.time() - t
    n = d.get("usage", {}).get("completion_tokens"); print(f"  {n} tokens in {dt:.1f}s ({(n or 0)/dt:.1f} tok/s)")
    time.sleep(2)
PYEOF
sleep 3
"$PY" "$S" runtime stop --profile launch-iq3_s-trace 2>&1 | tail -1
grep -E "expert profile saved|hit rate" C:/Users/erokh/AppData/Local/Strata/launch/iq3_s-trace.engine.log | tail -3 | sed 's/^[^ ]* [^ ]* //' | cut -c1-150
ls -l $T/builds-bench/strata-profile-after.bin 2>/dev/null | awk '{print $5 " bytes"}'
