"""Greedy-generation equivalence check between OpenAI-compatible servers.

collect: python -I gen_compare.py collect BASE_URL API_KEY MODEL OUT.json
compare: python -I gen_compare.py compare A.json B.json [C.json ...]
"""
import json, sys, urllib.request

PROMPTS = [
    "What is the capital of Australia? Answer in one sentence.",
    "Write a Python function that returns the n-th Fibonacci number iteratively.",
    "List the first ten prime numbers, separated by commas.",
    "Explain what a hash table is to a 10-year-old.",
    "Translate to French: 'The weather is nice today, let's go for a walk in the park.'",
    "What is 17 * 23? Show your work briefly.",
    "Write a haiku about autumn leaves.",
    "Summarize the plot of Romeo and Juliet in three sentences.",
]
MAX_TOKENS = 256


def collect(base, key, model, out, nolp=""):
    res = []
    for p in PROMPTS:
        body = {"model": model, "messages": [{"role": "user", "content": p}],
                "max_tokens": MAX_TOKENS, "temperature": 0.0, "top_k": 1, "seed": 0,
                "logprobs": True, "top_logprobs": 3,
                "chat_template_kwargs": {"reasoning_effort": "low"}}
        if nolp:  # e.g. Strata with --parallel rejects logprobs for max_tokens > 1
            del body["logprobs"], body["top_logprobs"]
        req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
        d = json.load(urllib.request.urlopen(req, timeout=600))
        ch = d["choices"][0]; msg = ch["message"]
        lp = (ch.get("logprobs") or {}).get("content") or []
        res.append({"prompt": p,
                    "reasoning": msg.get("reasoning_content") or msg.get("reasoning") or "",
                    "content": msg.get("content") or "",
                    "usage": d.get("usage"),
                    "tokens": [{"t": x["token"], "lp": x["logprob"],
                                "top": [(y["token"], y["logprob"]) for y in x.get("top_logprobs", [])]} for x in lp]})
        print(f"{p[:40]!r}: {d.get('usage')} logprobs={len(lp)}", flush=True)
    json.dump(res, open(out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)


def full_text(r):
    return r["reasoning"] + "\x00" + r["content"]


def compare(paths):
    runs = [json.load(open(p, encoding="utf-8")) for p in paths]
    ref, ref_name = runs[0], paths[0]
    for other, name in zip(runs[1:], paths[1:]):
        print(f"\n=== {ref_name}  vs  {name}")
        same = 0
        for a, b in zip(ref, other):
            ta, tb = full_text(a), full_text(b)
            if ta == tb:
                same += 1; print(f"  IDENTICAL  ({len(ta)} chars)  {a['prompt'][:45]!r}"); continue
            i = next((k for k in range(min(len(ta), len(tb))) if ta[k] != tb[k]), min(len(ta), len(tb)))
            note = ""
            # token-level view of the divergence point, when both sides returned logprobs
            if a["tokens"] and not b["tokens"]:
                # only the reference has logprobs: find the ref token covering the divergence char
                pos = 0; seq = [x["t"] for x in a["tokens"]]
                off = len(a["reasoning"]) + 1 if i > len(a["reasoning"]) else 0
                tgt = i - off; seg = a["tokens"]
                for k, x in enumerate(seg):
                    if pos + len(x["t"]) > tgt:
                        top = x["top"]; m = (top[0][1] - top[1][1]) if len(top) > 1 else None
                        note = f" | ref token #{k} {x['t']!r}, ref top-2 margin {m:.3f}" + (f" (2nd: {top[1][0]!r})" if m is not None else "")
                        break
                    pos += len(x["t"])
            if a["tokens"] and b["tokens"]:
                j = next((k for k in range(min(len(a["tokens"]), len(b["tokens"])))
                          if a["tokens"][k]["t"] != b["tokens"][k]["t"]), None)
                if j is not None:
                    ta_, tb_ = a["tokens"][j], b["tokens"][j]
                    top = ta_["top"]
                    margin = (top[0][1] - top[1][1]) if len(top) > 1 else None
                    note = (f" | token #{j}: {ta_['t']!r} vs {tb_['t']!r}; ref top-2 logprob margin "
                            f"{margin:.3f}" if margin is not None else f" | token #{j}")
            print(f"  DIVERGE at char {i}/{len(ta)}  {a['prompt'][:45]!r}{note}")
            print(f"      ref: ...{ta[max(0, i-40):i+40]!r}")
            print(f"      oth: ...{tb[max(0, i-40):i+40]!r}")
        print(f"  -> {same}/{len(ref)} identical")


if __name__ == "__main__":
    if sys.argv[1] == "collect":
        collect(*sys.argv[2:7])
    else:
        compare(sys.argv[2:])
