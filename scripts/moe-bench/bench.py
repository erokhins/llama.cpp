"""OpenAI-compatible server benchmark: identical workload for any backend.

usage: python -I bench.py BASE_URL API_KEY MODEL OUT.json
"""
import json, sys, time, uuid, pathlib, threading, statistics, urllib.request

BASE, KEY, MODEL, OUT = sys.argv[1:5]
CORPUS_DIR = pathlib.Path(r"C:\Users\erokh\AppData\Local\Strata\terminal\0.1.55")

# Long-context filler: real source code (realistic for coding workloads), deterministic order.
corpus = ""
for p in sorted(CORPUS_DIR.rglob("*.py")):
    if "site-packages" in p.parts or "python" == p.parts[len(CORPUS_DIR.parts)]:
        continue
    try:
        corpus += f"\n# ==== {p.name} ====\n" + p.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        pass
    if len(corpus) > 400_000:
        break

DECODE_PROMPTS = [
    "Explain in detail how a B-tree index works in a relational database, including insertion, deletion and node splitting. Use examples.",
    "Write a complete Python implementation of an LRU cache with O(1) get and put, with type hints, docstrings and unit tests.",
    "Describe the history of the Roman Empire from Augustus to the fall of the Western Empire, covering the main emperors and crises.",
]


def chat(messages, max_tokens, temperature=0.0):
    body = json.dumps({
        "model": MODEL, "messages": messages, "max_tokens": max_tokens,
        "temperature": temperature, "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=body, headers={
        "Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
    t0 = time.perf_counter(); t_first = None; usage = None; n_chunks = 0; timings = None
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            ev = json.loads(data)
            if ev.get("usage"):
                usage = ev["usage"]
            if ev.get("timings"):
                timings = ev["timings"]
            for ch in ev.get("choices", []):
                d = ch.get("delta", {})
                if d.get("content") or d.get("reasoning_content") or d.get("reasoning"):
                    n_chunks += 1
                    if t_first is None:
                        t_first = time.perf_counter()
    t_end = time.perf_counter()
    pt = usage.get("prompt_tokens") if usage else None
    ct = usage.get("completion_tokens") if usage else n_chunks
    ttft = (t_first or t_end) - t0
    dec = (ct - 1) / (t_end - t_first) if t_first and ct and ct > 1 and t_end > t_first else None
    return {"prompt_tokens": pt, "completion_tokens": ct, "ttft_s": round(ttft, 3),
            "total_s": round(t_end - t0, 3), "decode_tps": round(dec, 1) if dec else None,
            "prefill_tps_client": round(pt / ttft, 1) if pt else None, "server_timings": timings}


def nonce():
    return f"[session {uuid.uuid4()}]\n"


results = {"base": BASE, "model": MODEL, "started": time.strftime("%Y-%m-%d %H:%M:%S")}
log = lambda *a: print(*a, flush=True)

log("warmup..."); chat([{"role": "user", "content": nonce() + "Say hello."}], 32)

# 1) single-stream decode
results["decode"] = []
for p in DECODE_PROMPTS:
    r = chat([{"role": "user", "content": nonce() + p}], 1024)
    log("decode", r); results["decode"].append(r)

# 2) prefill at increasing context (unique prefix => no cache reuse)
results["prefill"] = []
for chars in (14_000, 56_000, 112_000):
    msg = nonce() + "Here is a codebase:\n" + corpus[:chars] + "\n\nSummarize what this code does in 3 sentences."
    r = chat([{"role": "user", "content": msg}], 128)
    log("prefill", r); results["prefill"].append(r)

# 3) decode at long context (~32k): reuse nothing, measure decode after big prompt
msg = nonce() + "Here is a codebase:\n" + corpus[:112_000] + "\n\nWrite a detailed review of this code, file by file."
r = chat([{"role": "user", "content": msg}], 1024)
log("long-ctx decode", r); results["long_ctx_decode"] = r

# 4) concurrency 4
out = [None] * 4
def worker(i):
    out[i] = chat([{"role": "user", "content": nonce() + DECODE_PROMPTS[i % 3]}], 512)
t0 = time.perf_counter()
ths = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
[t.start() for t in ths]; [t.join() for t in ths]
wall = time.perf_counter() - t0
agg = sum(o["completion_tokens"] for o in out) / wall
results["concurrency4"] = {"per_request": out, "wall_s": round(wall, 2), "aggregate_tps": round(agg, 1)}
log("concurrency4 aggregate tok/s", round(agg, 1), "wall", round(wall, 2))

results["summary"] = {
    "decode_tps_median": statistics.median(d["decode_tps"] for d in results["decode"] if d["decode_tps"]),
    "prefill": [(p["prompt_tokens"], p["prefill_tps_client"], p["ttft_s"]) for p in results["prefill"]],
    "long_ctx_decode_tps": results["long_ctx_decode"]["decode_tps"],
    "concurrency4_aggregate_tps": round(agg, 1),
}
log("SUMMARY", json.dumps(results["summary"]))
pathlib.Path(OUT).write_text(json.dumps(results, indent=1))
