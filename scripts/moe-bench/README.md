# moe-bench

Benchmark and profiling scripts for the MoE expert-cache work on Qwen3.8-Flash-Next IQ3_S (llama.cpp vs Strata),
RTX 5090 / Windows / Git Bash. What each script measures, how to read its output, and all results are in
[`docs/strata-moe-cache-handoff.md`](../../docs/strata-moe-cache-handoff.md) (sections "Files" and "How to start things").

Paths to the model and to Strata are hard-coded for the original machine (`C:/Users/erokh/...`); edit `M=` / `S=` lines
to run elsewhere. Environment:

- `BENCH_DATA` — data dir (default `C:/Users/erokh/testing`): outputs go to `$BENCH_DATA/builds-bench`, the MTP draft head is read from `$BENCH_DATA/mtp`.
- `BENCH_API_KEY` — API key passed to llama-server (default `sk-local`); set it to Strata's key for scripts that talk to the Strata HTTP server.
- `BIN` — another `llama-server.exe` for same-session A/B (`llama_bench_tf.sh`).

Servers are always addressed as `127.0.0.1` (on Windows `localhost` adds ~2 s per request).
