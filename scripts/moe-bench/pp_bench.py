"""Prefill-only benchmark for any OpenAI-compatible server (same corpus as bench.py).

usage: python -I pp_bench.py BASE_URL API_KEY MODEL OUT.json [sizes_chars,...]

Each prompt gets a unique nonce (no prefix-cache reuse); the first request is a warmup.
Reports the server's own timings (prompt_n, prompt_ms, prompt tok/s) and client TTFT.
"""
import json, sys, time, uuid, pathlib, urllib.request

BASE, KEY, MODEL, OUT = sys.argv[1:5]
SIZES = [int(x) for x in sys.argv[5].split(",")] if len(sys.argv) > 5 else [3_500, 14_000, 56_000, 112_000, 112_000]
CORPUS_DIR = pathlib.Path(r"C:\Users\erokh\AppData\Local\Strata\terminal\0.1.55")

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


def chat(content, max_tokens):
    body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": content}],
                       "max_tokens": max_tokens, "temperature": 0, "stream": True,
                       "stream_options": {"include_usage": True}}).encode()
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=body, headers={
        "Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
    t0 = time.perf_counter(); t_first = None; usage = None; timings = None
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
                if (d.get("content") or d.get("reasoning_content") or d.get("reasoning")) and t_first is None:
                    t_first = time.perf_counter()
    t_end = time.perf_counter()
    pt = usage.get("prompt_tokens") if usage else None
    ttft = (t_first or t_end) - t0
    tm = timings or {}
    return {"prompt_tokens": pt, "ttft_s": round(ttft, 3), "total_s": round(t_end - t0, 3),
            "client_pp_tps": round(pt / ttft, 1) if pt else None,
            "server_prompt_n": tm.get("prompt_n"), "server_prompt_ms": tm.get("prompt_ms"),
            "server_pp_tps": round(tm["prompt_per_second"], 1) if tm.get("prompt_per_second") else None}


nonce = lambda: f"[session {uuid.uuid4()}]\n"
chat(nonce() + "Say hello.", 8)  # warmup
res = []
for chars in SIZES:
    r = chat(nonce() + "Here is a codebase:\n" + corpus[:chars] + "\n\nSummarize what this code does in one sentence.", 8)
    print(f"pp n={r['prompt_tokens']:>6}  server {r['server_pp_tps']} tok/s ({r['server_prompt_ms'] and round(r['server_prompt_ms'])} ms)"
          f"  client ttft {r['ttft_s']} s = {r['client_pp_tps']} tok/s", flush=True)
    res.append(r)
pathlib.Path(OUT).write_text(json.dumps({"base": BASE, "started": time.strftime("%Y-%m-%d %H:%M:%S"), "prefill": res}, indent=1))
