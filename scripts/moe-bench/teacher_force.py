"""Teacher-forced next-token comparison: feed identical token-id prefixes to each engine, read top-K logprobs.

  ref     python -I teacher_force.py ref     LLAMA_URL KEY OUT_ref.json [N_GEN]
  llama   python -I teacher_force.py llama   LLAMA_URL KEY REF.json OUT.json
  strata  python -I teacher_force.py strata  REF.json OUT.json          (spawns strata.exe --serve itself)
  compare python -I teacher_force.py compare REF.json A.json B.json
"""
import json, math, subprocess, sys, time, urllib.request

TOPK = 20
PROMPT = ("Write a detailed, well-structured technical essay (about 2000 words) on how modern CPUs execute "
          "instructions: pipelining, out-of-order execution, branch prediction, caches and memory hierarchy, "
          "and SIMD. Include concrete examples.")


def post(url, key, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    return json.load(urllib.request.urlopen(req, timeout=3600))


def cmd_ref(url, key, out, n_gen="3000"):
    text = post(url + "/apply-template", key, {"messages": [{"role": "user", "content": PROMPT}],
                                               "chat_template_kwargs": {"reasoning_effort": "low"}})["prompt"]
    prompt_ids = post(url + "/tokenize", key, {"content": text, "add_special": False, "parse_special": True})["tokens"]
    r = post(url + "/completion", key, {"prompt": prompt_ids, "n_predict": int(n_gen), "temperature": 0, "top_k": 1,
                                        "return_tokens": True, "cache_prompt": False})
    gen = r["tokens"]
    print(f"prompt {len(prompt_ids)} tokens, generated {len(gen)} tokens, stop={r.get('stop_type')}")
    json.dump({"prompt_text": text, "prompt_ids": prompt_ids, "gen_ids": gen, "gen_text": r["content"]},
              open(out, "w", encoding="utf-8"), ensure_ascii=False)


def positions(ref):
    """(prefix ids, forced next id) for every generated position."""
    full = ref["prompt_ids"] + ref["gen_ids"]
    p = len(ref["prompt_ids"])
    return [(full[:p + k], full[p + k]) for k in range(len(ref["gen_ids"]))]


def cmd_llama(url, key, ref_path, out):
    ref = json.load(open(ref_path, encoding="utf-8")); res = []; t0 = time.time()
    for k, (prefix, nxt) in enumerate(positions(ref)):
        r = post(url + "/completion", key, {"prompt": prefix, "n_predict": 1, "n_probs": TOPK, "temperature": 0,
                                            "top_k": 1, "cache_prompt": True, "id_slot": 0})
        cp = r["completion_probabilities"][0]
        res.append({"argmax": cp["id"], "top": [[t["id"], t["logprob"]] for t in cp["top_logprobs"]]})
        if k % 250 == 0:
            print(f"  llama {k}/{len(ref['gen_ids'])}  {time.time() - t0:.0f}s", flush=True)
    json.dump(res, open(out, "w"))


STRATA_DIR = r"C:\Users\erokh\AppData\Local\Strata"
STRATA_EXE = STRATA_DIR + r"\components\engine\0.1.55-windows-x86_64-cuda-cuda13\strata.exe"
STRATA_ARGS = ["--serve", "--pack", STRATA_DIR + r"\Strata-data\packs\iq3_s",
               "--native", STRATA_DIR + r"\Strata-data\models\IQ3_S\Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf",
               "--ple-gguf", STRATA_DIR + r"\Strata-data\models\IQ3_S\Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf",
               "--expert-profile", STRATA_DIR + r"\terminal\0.1.55\data\expert-profile.bin",
               "--expert-cache", "auto", "--spec", "4", "--spec-min-p", "0.5",
               "--mtp", STRATA_DIR + r"\Strata-data\mtp\rt", "--max-context", "131072",
               "--kv", "int8", "--kv-resident", "32768", "--prefill", "auto:32768"]   # the server's args minus --batch 4


def cmd_strata(ref_path, out):
    ref = json.load(open(ref_path, encoding="utf-8"))
    log = open(out + ".engine.log", "w")
    proc = subprocess.Popen([STRATA_EXE] + STRATA_ARGS, cwd=STRATA_DIR + r"\terminal\0.1.55",
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True, bufsize=1)
    t0 = time.time()
    while True:
        line = proc.stdout.readline()
        if not line:
            sys.exit(f"engine exited before READY (see {out}.engine.log)")
        if line.startswith("READY"):
            print(f"  strata READY in {time.time() - t0:.0f}s: {line.strip()}", flush=True); break
    res = []; t0 = time.time()
    for k, (prefix, nxt) in enumerate(positions(ref)):
        proc.stdin.write(f"GEN 1 logprobs={TOPK} {','.join(map(str, prefix))}\n"); proc.stdin.flush()
        tok = lp = None
        while True:
            line = proc.stdout.readline()
            if not line:
                sys.exit("engine died mid-run")
            if line.startswith("T "):
                tok = int(line.split()[1])
            elif line.startswith("LP "):
                lp = line.split()[1:]
            elif line.startswith("ERR"):
                sys.exit(f"engine error: {line.strip()}")
            elif line.startswith("DONE"):
                break
        # "LP <logprob of T> id:logprob id:logprob ..."
        top = [[int(x.split(":")[0]), float(x.split(":")[1])] for x in lp[1:]]
        res.append({"argmax": tok, "top": top})
        if k % 250 == 0:
            print(f"  strata {k}/{len(ref['gen_ids'])}  {time.time() - t0:.0f}s", flush=True)
    proc.stdin.close(); proc.wait(timeout=60)
    json.dump(res, open(out, "w"))


def cmd_compare(ref_path, a_path, b_path):
    ref = json.load(open(ref_path, encoding="utf-8"))
    A, B = json.load(open(a_path)), json.load(open(b_path))
    forced = ref["gen_ids"]; n = min(len(A), len(B), len(forced))
    agree = 0; kls = []; dlp = []; disagree = []; tie_thresh = []
    for k in range(n):
        a, b = A[k], B[k]
        if a["argmax"] == b["argmax"]:
            agree += 1
        else:
            ta = sorted(a["top"], key=lambda x: -x[1]); margin = ta[0][1] - ta[1][1] if len(ta) > 1 else float("inf")
            disagree.append((k, a["argmax"], b["argmax"], margin))
        da, db = dict(map(tuple, a["top"])), dict(map(tuple, b["top"]))
        # KL(A||B) over A's top-K, B's missing mass floored at B's smallest reported logprob
        floor = min(db.values())
        kls.append(sum(math.exp(la) * (la - db.get(t, floor)) for t, la in da.items()))
        f = forced[k]
        if f in da and f in db:
            dlp.append(abs(da[f] - db[f]))
    kls.sort(); dlp.sort()
    q = lambda xs, p: xs[min(len(xs) - 1, int(p * len(xs)))]
    print(f"positions compared: {n}")
    print(f"argmax agreement  : {agree}/{n} = {agree / n:.2%}")
    print(f"KL(A||B) top-{TOPK}  : mean {sum(kls) / n:.5f}  median {q(kls, .5):.5f}  p99 {q(kls, .99):.5f}  max {kls[-1]:.4f}")
    print(f"|dlogprob| forced : median {q(dlp, .5):.4f}  p90 {q(dlp, .9):.4f}  p99 {q(dlp, .99):.4f}  max {dlp[-1]:.4f}")
    if disagree:
        ms = sorted(d[3] for d in disagree)
        print(f"argmax disagreements: {len(disagree)}; A's top-2 margin at those: median {q(ms, .5):.3f}, max {ms[-1]:.3f}")
        print(f"  share with margin < 0.1: {sum(m < 0.1 for m in ms) / len(ms):.0%};  < 0.5: {sum(m < 0.5 for m in ms) / len(ms):.0%}")


if __name__ == "__main__":
    {"ref": cmd_ref, "llama": cmd_llama, "strata": cmd_strata, "compare": cmd_compare}[sys.argv[1]](*sys.argv[2:])
