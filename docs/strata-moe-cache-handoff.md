# Handoff: Strata vs llama.cpp on Qwen3.8-Flash-Next IQ3_S

Session date: 2026-11-02. Everything lives under `C:\Users\erokh\testing\` unless a path says otherwise.

## TL;DR

- Both engines serve **the same weights at the same quantization**. On a teacher-forced next-token test they compute
  practically the same distributions: KL ≈ 0.003, 97.6% top-1 agreement, and every disagreement was a near-tie.
- **Strata is ~2x faster at generation and ~2.5–5x faster at prompt reading** than the best llama.cpp config found.
- The best llama.cpp settings come from flags that are already in mainline but off by default:
  - `--moe-cache-mib 8192` (GPU expert cache): biggest win.
  - MTP draft head, `-md mtp-*.gguf --spec-type draft-mtp --spec-draft-n-max 2`: small extra gain.
  - No source build or PR merge was needed.
- **State at end of session: nothing is serving on :8080.** See "How to start things" below.
- **Evening update:** Unsloth PR #268 merged onto b11960 and built locally closes much of the gap: decode 92,
  prompt 3,350 tok/s at 30k, 142 tok/s with 4 requests (12 GB cache, no MTP). Details in the "Update" section below.
- **2026-11-03 update:** with the 20 GB expert-cache layout (`-c 32768 -np 2 -b 2048 -ub 2048 --moe-cache-mib 24576`)
  and three CUDA fixes found by Nsight Systems (commit 7ad8dae47), llama.cpp reaches **134-139 tok/s with MTP-4**
  (`--spec-draft-n-ubatch 64`) and 107-110 without MTP; Strata 183-211. The remaining gap is attributed in the
  "Nsight Systems attribution" section: inline PCIe fetch of expert misses, the 4 separate draft decodes per round,
  and launch gaps between small kernels.

## Update, 2026-11-02 evening: Unsloth PR #268 applied to llama.cpp

- **Correction to "Why Strata is faster", point 1:** mainline's `--moe-cache-mib` already streams expert misses
  over PCIe and computes them on the GPU (`src/llama-moe-cache.cpp`: the per-layer copy callback reads the routed
  ids back to the host, uploads the missing experts from pinned host memory into a VRAM bank, then `mul_mat_id`
  runs on the bank). Misses are *not* computed on the CPU. What differs from Strata is the pipeline: a host
  round trip and stream sync per MoE layer per token, a cold cache, no seeding from the prompt.
- **Unsloth PR #268** (`unslothai/llama.cpp#268`, branch `moe-offload-prefetch-cache`, open, base b11577) fixes
  exactly that: on-device slot lookup (no per-layer readback), misses read from pinned mapped host memory, next-split
  host-weight prefetch during prompt processing, cache seeding at prompt end, device-to-device copies of cached
  experts in prefill. All on by default; kill switches `GGML_SCHED_PREFETCH=0`, `LLAMA_MOE_CACHE_DEVICE=0`,
  `LLAMA_MOE_CACHE_SEED=0`.
- **Builds** (all with the per-user toolchain in `C:\Users\erokh\tools`, see `tools\buildenv.sh`; no admin was
  available, nothing system-wide was installed, the driver is untouched):
  - `src\llama-base` = cbe28bb3c (b11577, the PR's base), `src\llama-pr268` = PR as is,
  - `src\llama.cpp\build` = b11960 built locally (matches the prebuilt within noise, so toolchain is not a factor),
  - **`src\llama-merge` = PR #268 merged onto b11960** (3 merge commits: 7 conflict hunks resolved by taking the
    PR's `llama-moe-cache.cpp` and re-applying upstream's `no_alloc` move, op-count assert 103, `model.no_alloc()`).
    Binary: `src\llama-merge\build\bin\llama-server.exe`.
  - MSVC pitfalls: `GGML_NATIVE=ON` adds no `/arch` flags (decode 4x slower); pass AVX-512 flags explicitly;
    `GGML_AVX512_BF16` does not compile; copy the three CUDA DLLs next to the exe.
- **Results** (`scripts/run_bin.sh`, bench over 127.0.0.1, server-side prompt tok/s; `builds-bench/`):

  | Config | Gen | Gen after 30k | Prompt 30k | TTFT 30k | 4 requests |
  |---|---|---|---|---|---|
  | prebuilt b11960, 8 GB cache + MTP2 (old run, localhost) | 86 | 73 | 1,571 | 21.3 s | 74 |
  | prebuilt b11960, 8 GB cache (old run, localhost) | 76 | 68 | 1,779 | 19.1 s | 104 |
  | b11577 (PR base), 8 GB cache + MTP2 | 44 | 35 | 1,409 | 21.5 s | 49 |
  | PR #268 on b11577, 8 GB cache + MTP2 | 84 | 68 | 2,526 | 12.0 s | 69 |
  | **PR #268 on b11960, 8 GB cache + MTP2** | **91** | **77** | 2,892 | 10.5 s | 82 |
  | PR #268 on b11960, 8 GB cache, no MTP | 90 | 77 | **3,365** | **9.1 s** | 131 |
  | **PR #268 on b11960, 12 GB cache, no MTP** | **92** | **80** | 3,348 | 9.1 s | **142** |
  | Strata (old run) | 183 | 172 | 7,260 | 6.3 s | 161 |

  - On its own base the PR doubles decode and prompt speed. On b11960 it adds ~6% decode with MTP, ~18% without,
    and ~1.8–1.9x prompt speed; TTFT at 30k halves.
  - With the PR, MTP no longer pays: +1 tok/s decode, costs 14% prompt speed and 37% concurrent throughput.
    **New best single-user and multi-user config: `--moe-cache-mib 12288` without MTP on the merged build.**
    A bigger cache no longer costs prompt speed (the prefetch hides the expert stream), so 12 GB beats 8 GB everywhere.
  - Gap to Strata is now 2.0x decode, 2.2x prompt, 1.1x concurrent (was 2.1x, 4.6x, 1.5x).
  - The cache hit statistics are still not captured: llama-server only logs them on Ctrl-C and `scripts/srv.py`'s
    console Ctrl-C did not reach it (falls back to force kill).

## Update, 2026-11-02 night: where the decode time goes, and the one kernel fix that paid

Goal was "cut kernel count per layer for 30-40% decode". Result: **not available, +5% taken**, reasons measured.

- **Kernel count was a false lead.** The scheduler dump shows 4,265 graph nodes per token, but the CUDA backend already
  fuses the router (softmax+argsort+get_rows+norm), the 9-way expert sum, the hyper-connection gate chain, rms_norm+mul,
  the gated-delta-net scan and gate/up+SwiGLU. Dispatched launches per token: about 1,700 for the main split, 2,354 for
  the whole graph. Strata's 1-token window graph has 1,909 nodes (`STRATA_VERIFY_NODES=1`). Same ballpark.
- **Steady-state GPU profile** (events inside the replayed CUDA graph, `GGML_CUDA_OP_TIMING=1 -lv 4`, slowed by the
  event nodes themselves so read the shares, not the ms): dense GEMVs 700 launches / 36%, expert matmuls + cache lookup
  10%, about 1,400 small elementwise/copy kernels at their 2-3 us floor for the rest. Flat; no single hot spot.
- **Other things ruled out by measurement:** CPU hops (whole-layer placement, 2 splits/token: no faster), hit rate
  (89% vs 93%: no change), CUDA graph chunking (none), fit granularity (moves the split layer, nothing else).
- **The one inefficiency found:** the BF16 hyper-connection up-projection (10240 rows x 320) ran at 630 GB/s because
  `mmvf` uses one block per row. Routing single-token F16/BF16 GEMVs with <= 1024 columns and >= 2048 rows to the MMF
  tensor-core path makes that shape 3.2x faster (`ggml/src/ggml-cuda/mmvf.cu`, `ggml_cuda_should_use_mmvf`). Shapes with
  few rows stay on `mmvf` (MMF is 3-4x slower there). 2490/2490 `test-backend-ops` MUL_MAT cases pass.

  | Merged build, 12 GB cache, no MTP | Gen | Gen after 30k | Prompt 30k | 4 requests |
  |---|---|---|---|---|
  | before | 92 | 80 | 3,348 | 142 |
  | **after the GEMV rule** | **97** | **84** | 3,388 | **151** |

- **What is left and why it is not 30%:** dense weights need 2.3 ms/token at full bandwidth and take about 3.5 ms; the
  ~1,400 small kernels cost 3-4 ms at their launch floor; experts ~1 ms; the rest is gaps. Each remaining item is worth
  3-8%: merge GEMVs that share an input at load (hc down+inject, alpha+beta, qkv+gate: needs loader support, the GGUF has
  no merged tensors), skip the recurrent-state shuffle for a single sequence, fuse scale+silu and the indexer fill/concat.
  The 2x to Strata is tokens per pass (its 6-token window costs 7% more nodes than 1 token), not kernels per token.
- **Diagnostics added to the merged tree (env-gated, off by default):** `GGML_CUDA_KERNEL_STATS=1` (dispatched op
  histogram + CUDA graph node counts), `GGML_CUDA_OP_TIMING=1` (per-op GPU time on replays 50, 150, ...), a periodic
  MoE-cache hit-rate log every 1000 decodes, and `TEST_QWEN4EXP_SHAPES=1` perf cases in `test-backend-ops`. All need `-lv 4`.
- Tools: `scripts/quick_hit.sh` (4x512 tokens, hit-rate log), `scripts/quick_pp.sh` (one 30k prompt).

## Update, 2026-11-02 late: per-stage comparison of one decode step, llama.cpp vs Strata

Method: llama.cpp `GGML_CUDA_OP_TIMING=2 -lv 4` (CUDA event nodes at stage boundaries inside the replayed graph, replay 250,
12 GB cache, no MTP), corrected by subtracting the measured ~8.5 us event overhead per segment. Strata in bench mode
(`scripts/strata_stage_bench.sh`, `STRATA_VERIFY_PROFILE=1 STRATA_SPEC_T_CYCLE=1,5`), its own GPU stamps, 89-token prompt,
600 tokens. Strata's T1 = a verify window of one token, i.e. a plain non-speculative step.

| Stage (ms per 1-token step) | llama.cpp | Strata T1 |
|---|---|---|
| hyper-connections (norm, down, up, inject, combine) | 2.07 | 1.88 |
| token mixer (SSM qkv/conv/recurrence/out, attention, indexer) | 2.68 | 2.23 |
| router + expert lookup/plan | 1.29 | 0.44 (+0.27 host wait) |
| routed experts (gate/up, down, weighted sum) | 0.82 | 1.06 resident (+0.59 misses: waits, PCIe, CPU) |
| shared expert | 0.51 | 0.67 |
| output head | 0.34 | 0.33 |
| unclassified / gaps | 0.81 + ~2.4 | 0.02 + ~1.3 |
| **whole step** | **10.9** | **8.8** |
| 5-token window | n/a (MTP verify of 3 tokens gave +1 tok/s) | **14.3** |

- Per non-speculative step llama.cpp is 24% slower, not 2x. Stage by stage the two engines are within tens of
  percent of each other; the largest single difference is router + lookup (0.85 ms, the `MOE_CACHE_LOOKUP` op at
  ~17-26 us per layer plus a 6-op router chain), then unattributed gaps.
- Strata's 2x comes from the window: 5 tokens cost 14.3 ms, 1.6x a single token, so ~2.9 accepted tokens per 14 ms.
  llama.cpp's MTP path runs the draft as a separate context and its 3-token verify costs ~2.4x a single token.
- Strata's T1 expert cost is higher than llama.cpp's because its misses go to the CPU pool and the GPU waits for them;
  that is the price of its 20 GB expert cache layout, which pays off in the wide windows.

## Update, 2026-11-03 early: why MTP is expensive in llama.cpp, what was fixed, what was tried

A round with 4 drafts = 4 draft decodes (3.4 ms together, `dur(b,g,a)` in the `draft-mtp: statistics` log line) plus one
verify of up to 5 tokens (about 22 ms, 2x a single token). Strata's 5-token window is 14.3 ms (1.6x its single token).

- **Fix 1 (flags, +33%): the draft model's context stole the expert cache.** The draft inherits the main `-ub 4096` and
  reserves a 3.8 GB compute buffer plus a 128k context; the fit then shrank the main MoE cache from 12.3 to 8.5 GB and
  kept no expert layer resident, so the 50-expert verify missed far more. `--spec-draft-n-ubatch 64` cuts the draft
  buffer to 0.35 GB; with `-b 2048 -ub 2048` the main compute buffer halves too and the fit keeps a 14.4 GB cache.
  MTP4 decode: 79 tok/s (defaults) -> 105 tok/s (`-b 2048 -ub 2048 --spec-draft-n-ubatch 64 --moe-cache-mib 16384`).
  Above 14 GB of cache the gain flattens (+1% at 16 GB). `--spec-draft-backend-sampling`: no effect.
- **The verify misses 12% of its expert reads** (device lookups at 5 tokens: 88% hits, ~120 missed experts per round,
  ~230 MB over mapped host memory). Strata runs the same window at 97-99% hits with a 20 GB cache and computes the few
  misses on its CPU pool while the GPU continues. Residency, not the lookup mechanism, is the lever: the DMA path
  (`LLAMA_MOE_CACHE_DEVICE=0`) is 9% slower.
- **Draft quality: the same.** (Corrects an earlier note here that compared llama.cpp's runtime acceptance with Strata's
  offline probe figure of 0.89.) Strata's serve log over 86,665 generated tokens: 0.710 accepted per draft with its
  0.5 gate, 2.26 drafts per round, 2.61 tokens per round, 14.1 ms per round. llama.cpp with the same 0.5 gate: 0.70 per
  draft (per position 0.86 / 0.52 / 0.32 / 0.22); ungated: 0.45 per draft, 4 drafts per round, 2.8 tokens per round,
  about 26 ms per round. Same head, same mathematics; Strata gets fewer tokens per round and wins only on the cost of
  the round.
- **Gating the drafts (`--spec-draft-p-min 0.5 / 0.7`) made it slower (75 / 66 tok/s) despite better acceptance**, and
  the reason is structural: the verify width then varies from 1 to 5 tokens, and every width change rebuilds the ggml
  graph and re-captures the CUDA graph (`graphs reused` 727 -> 273, `graph reallocs` 14 -> 49-89, about 13 ms per
  round). Strata keeps one captured graph per window width and replays it.
- **Tried: a per-shape graph cache in `llama_context`** (`gf_res_prev` keyed by n_tokens/n_seqs/outputs; a graph built
  for a width is re-allocated in the scheduler instead of rebuilt). Blocker found and left documented in the code:
  `ggml_backend_sched_split_graph` rewrites node sources in place to its transfer copies and forgets the builder's
  explicit backend pins at reset; restoring sources (`snapshot_srcs`/`restore_srcs`) and replaying pins
  (`llm_graph_result::pins`) was not enough, the re-split still crashes (no assert) in the scheduler, first on the draft
  model's 187-node graph, then on the target's. Even disabled, the keyed result cache changed the draft's behaviour
  (acceptance 0.45 -> 0.41, 104 -> 96 tok/s), so **the attempt was reverted**; `src/llama-merge` is back at its last
  commit (MMF GEMV rule + diagnostics). The design and the blocker are recorded here only. A padded fixed-width verify
  would avoid the engine change but cannot beat the ungated fixed width, so it was not pursued.
- **Where the VRAM goes (MiB, MTP run, fit log).** llama.cpp: dense weights 3,982, MoE cache 14,374, main KV 1,632 +
  recurrent 408 (128k, 4 slots), main compute buffer 3,360 (`-ub 2048`; 5,795 at 4096), prefetch staging 794, draft
  model 3,291 (Unsloth's Q8_0 head: full 248k-vocab embedding + output ~1.3 GB, Q8_0 experts), draft KV 320, draft
  compute 349, fit headroom 1,024. Strata: the same dense weights, KV resident only for 32k cells per attention layer
  (~0.4 GB, the rest streams from pinned RAM), draft head 835 (experts requantized to ~1.3 MB each, vocab subset),
  expert cache 20,900 (10,756 slots; with `--batch 4` the slot sessions take 2.9 GB of it). That is the 6 GB.
  On top of size, Strata's cache is seeded from `expert-profile.bin` and adapts; its serve-mode hit rate starts at
  80-86% and settles at 97-99%, llama.cpp's CLOCK cache sits at 88% for 5-token verifies at 14.4 GB.
- **Smaller draft heads (requantized from Unsloth's Q8_0 with `llama-quantize --allow-requantize`, files in `mtp\`):**

  | Head | File | Draft VRAM | Expert cache (fit) | MTP4 decode | Accepted per draft |
  |---|---|---|---|---|---|
  | Q8_0 (Unsloth) | 3.9 GB | 3.3 GB | 14.4 GB | 102 | 0.452 |
  | IQ4_XS | 2.2 GB | 1.85 GB | 15.8 GB | 95 | 0.415 |
  | **Q3_K_M** | 2.1 GB | 1.78 GB | 15.9 GB | **105** | 0.433 |
  | Q2_K | 1.7 GB | 1.51 GB | 16.2 GB | 98 | 0.370 |

  Freeing 1.5 GB for experts buys 4% at best, and 2-bit heads lose more in acceptance than they give back. Together
  with the 14 -> 16 GB cache runs (+1%), this shows the CLOCK cache is flat above ~14 GB: the lever is hit rate per GB
  (profile seeding, adaptation), not more GB. The 640-column expert down projections cannot take K-quants (fall back).
- **The compute buffer cannot be handed back after prefill**, and what fills it: ggml's allocator only grows and the
  MoE cache is sized once at creation. The buffer scales with `-ub` x `-c`: the sparse-attention indexer scores every
  query against all 131,072 cells (~1.2 MB per token) plus the top-k workspace, and gathers the full indexer key cache
  (256 MiB per attention layer). Strata scores blockwise with a running top-k. In llama.cpp the levers are a smaller
  `-c` (scores shrink with it) or a smaller `-ub`; the blockwise indexer kernel is the real fix.
- **Reaching Strata's expert cache size (measured 2026-11-03).** The fit gives the cache whatever the context,
  compute buffer and draft head leave. MTP4, Q3_K_M head unless noted:

  | Layout | KV | Compute buffer | Expert cache | MTP decode | Accepted per draft |
  |---|---|---|---|---|---|
  | 128k, 4 slots, ub 2048 | 2.0 GB | 3.4 GB | 15.9 GB | 105 | 0.433 |
  | 128k, 4 slots, compute buffer released (simulated with ub 64) | 2.0 GB | ~0.05 GB | 18.3 GB | 108 | 0.426 |
  | 32k, 2 slots, ub 2048 | 0.5 GB | 1.7 GB | 20.5 GB | 116 | 0.430 |
  | 32k, 2 slots, ub 2048, Q8_0 head | 0.5 GB | 1.7 GB | 19.3 GB | 115 | 0.452 |
  | 32k, 2 slots, compute buffer released | 0.5 GB | ~0.05 GB | 21.5 GB | 119 | 0.435 |
  | Strata (serve, `--batch 4`) | 0.4 GB resident | small | 17.5-20.4 GB | 183 (5-token windows) | 0.71 gated |

  So Strata's cache size is reached today with `-c 32768 -np 2` (20.5 GB, same acceptance as Strata's with the Q8_0
  head). Releasing the compute buffer after prefill is worth +1 GB at 32k and +2.4 GB at 128k, about +3% decode either
  way, because the CLOCK cache gains little above ~16 GB. Prefill cost of the micro-batch: 2,442 tok/s at ub 2048 vs
  1,616 at 1024 on a 31k prompt.
- **Implemented: `--phase-aware-workspace`** (commit `3d8925f7c` in `src\llama-merge`; env `LLAMA_ARG_PHASE_AWARE_WORKSPACE`).
  After a streak of decode-sized batches (<= 32 tokens, hysteresis 8, `LLAMA_PAW_HYSTERESIS`) `llama_context` rebuilds
  its scheduler and reserves only the decode graph (`sched_reserve_decode`), which frees the prompt-sized compute
  buffers; a batch larger than the current reservation runs the normal full reservation again (prompt buffers and the
  prefetch staging come back). The idea and the flag name come from the GenerelSchwerz fork's "phase-aware workspace"
  (never upstreamed, 50-129 files with unrelated features); this is a 120-line reimplementation on our tree.
  Measured at 32k / 2 slots / `-ub 2048`: VRAM in use during generation 31,428 -> 30,214 MiB (-1.2 GB), prompt
  processing 2,566 vs 2,568 tok/s, generation 104.3 vs 104.4 tok/s. At 128k / 4 slots with MTP (both contexts shrink,
  target to 470 MiB and draft to 339 MiB): 31,862 -> 28,900 MiB (-3.0 GB) during generation, prompt 1,990 tok/s,
  generation 103-109 tok/s; a second prompt regrows correctly (the full reservation takes 7-90 ms). The freed VRAM is
  just free: nothing uses it yet. `scripts/paw_test.sh` is the check, log lines need `-lv 4`.
- **To let the expert cache use it (not implemented, est. 1-2 days):** the cache bank is a fixed `cudaMalloc`, so it
  needs (1) the bank allocated from a CUDA virtual-memory reservation (`cuMemAddressReserve` + `cuMemMap`, primitives in
  `ggml_cuda_pool_vmm`) so it grows and shrinks in place and stays contiguous for `mul_mat_id`; (2)
  `llama_moe_cache::grow/shrink` (extend the SLRU/CLOCK structures and the device `state` tensor, evict tail slots before
  a shrink); (3) call grow after `sched_reserve_decode` and shrink before the full reservation. Measured upper bound of
  the payoff with the current cache policy: +3% decode.
- **What would make MTP as cheap as Strata's window**, in order: (1) expert residency for the 50-expert verify: free
  VRAM (smaller draft head, KV streaming or a smaller `-c`, smaller `-ub`) and a profile-seeded or adaptive cache
  (the 12% misses are ~7 ms of the 22); (2) graph reuse per verify width so the draft can be gated (needs scheduler
  support for re-splitting, or one scheduler allocation per width); (3) the draft steps without host round trips (the
  residual goes GPU -> host -> GPU and logits come back for sampling four times a round).

## Update, 2026-11-03: which experts each engine keeps, versus the experts actually used

Setup: the 20 GB layout (`-c 32768 -np 2 -b 2048 -ub 2048`, no MTP), three bench.py decode prompts x 1024 tokens.
llama.cpp recorded every routed expert id (`LLAMA_RECORD_EXPERTS`, a `cb_eval` on `ffn_moe_topk-*`) and its
resident set at every hit-rate log (`LLAMA_MOE_CACHE_DUMP`); Strata ran the same prompts with
`--expert-profile-save`, whose output lists its resident experts first. Caches were the same size: llama.cpp 12,079
slots (23.6 GB), Strata 12,289 slots (23.3 GiB). Tools: `scripts/expert_trace.sh`, `scripts/strata_trace.sh`,
`scripts/expert_analysis.py`, data in `builds-bench/experts-cold.bin`, `resident-cold.bin`, `strata-profile-after.bin`.

| | llama.cpp (CLOCK, cold start) | Strata (profile-seeded, adaptive) | static oracle, same slot count |
|---|---|---|---|
| Dynamic hit rate over the run | 96.6% | 97.6% | n/a |
| Static coverage of the run's lookups by the final resident set | 68.6% (72.8% at the first snapshot) | 73.3% | 93.7% |
| Overlap with Strata's resident set | 9,759 of ~12,100 pairs (81%) | | 8,591 of 12,079 (71%) |

- The workload touches 21,994 of the 24,576 (layer, expert) pairs; the best fixed set of 12,079 covers 93.8% of
  lookups, and both engines reach a higher *dynamic* hit rate than that by adapting, because the hot set drifts
  through a response. Their resident sets converge to the same experts (81% overlap).
- **At equal size the expert choice is not the difference: 96.6% vs 97.6%.** Earlier runs at 14 GB (88% for 5-token
  verifies) were a size problem, not a policy problem, and the 32k layout removes it.
- `LLAMA_MOE_CACHE_PRELOAD=file` (int32 layer, expert pairs, least important first) now warm-starts the cache; the
  oracle file for this workload is `builds-bench/oracle-preload.bin`. **Measured:** preloading the oracle set lifts
  the hit rate from 96.6% to 97.0% on the traced prompts (misses -12%) and from 95.6% to 96.2% on the quick test,
  i.e. to Strata's level, and decode speed does not move (106.5 vs 105.6 tok/s, noise). Verdict of the experiment:
  **"non-optimal expert choice" is ruled out; the remaining gap is the implementation** (per-step overhead in router +
  lookup and gaps, and the speculative round structure).

## Update, 2026-11-03: Nsight Systems attribution of the 1-token step and the 5-token verify

Setup: the 20 GB layout (`-c 32768 -np 2 -b 2048 -ub 2048 --moe-cache-mib 24576`), 2 x 256 tokens, Nsight Systems
2026.3.2 (per-user install under `tools\nsys`, `--cuda-graph-trace=node` so kernels inside CUDA graphs are traced
individually). Tools: `scripts/nsys_decode.sh NAME [server args]` (profile + CSV export; the server writes its own
`--log-file`, nsys swallows its stdout), `scripts/nsys_steps.py TRACE.csv` (segments = graph replays, per-step
busy/idle, kernel families). Data: `builds-bench/nsys-plain*.{nsys-rep,csv}`, `nsys-mtp4*`. Under the profiler the
server ran at its normal speed (100 tok/s plain, 105 tok/s MTP4), so the traces are representative. Note: nsys
reports kernel durations that overlap slightly inside a graph, so "sum of kernel time" exceeds wall time by ~10%;
wall times below are what matter.

**Where a 1-token step goes (period 9.8 ms, median 9.55):** main-model graph replay 8.97 ms wall, other GPU work
(logit copy, input uploads) 0.09 ms, GPU idle 0.73 ms (sampling, batch build, graph reuse check, chunk launches).
Strata's 1-token window: 8.81 ms, of which 7.50 ms GPU stages and 1.3 ms outside. So **llama.cpp's CPU side is
already leaner than Strata's; the 1-token gap is 1.5 ms of GPU time inside the graph**, mainly the expert fetch
(0.75-1.75 ms per step: misses are copied from mapped host memory *inside the graph*, serialising the GPU on PCIe)
and ~1,700 small kernels with ~0.6 us launch gaps each (~1 ms).

**Where an MTP-4 round goes (period 25.1 ms, median 24.3; ~3.8 tokens per round):**

| part | llama.cpp | Strata T5 round (14.33 ms) |
|---|---|---|
| verify graph, 5 tokens through 48 layers | **19.1 ms** wall | 13.04 ms GPU stages (incl. its MTP head 0.38) |
| drafting: 4 x (MTP layer + full 248k-vocab head), eager-ish small graphs | 2.6 ms (lm_head 0.40 ms x 4) | inside the 13.04 (vocab-subset head) |
| draft update over the accepted tokens | 0.4 ms | - |
| GPU idle (CPU: sampling, 5 graph builds, uploads) | 2.8 ms | ~1.3 ms |

The 5-token verify vs Strata's 5-token stages, by family (ms):

| family | llama.cpp verify (19.1) | Strata T5 (13.04) | note |
|---|---|---|---|
| expert misses: fetch over PCIe / Strata CPU pool wait | **4.56** (`moe_cache_fetch`, 48 x 95 us) | 2.06 (waitA+waitB+PCIe+waitCPU) | ~1.4-2 misses/layer x ~55 us (2.1 MB at ~38 GB/s) + 17 us empty-launch floor x 48 |
| expert matmuls (hits) | 2.82 (`mul_mat_vec_q_moe`) + 0.36 quantize | 2.59 (VRAM hits) | equal |
| router + slot assignment | 0.21 + 0.20 (`moe_cache_assign`) | 0.62 (router+plan) | equal |
| shared experts | ~0.6 | 0.95 | |
| dense BF16 (hc up/down, out-proj, qkv) | **4.92**, of which `mul_mat_f` with 10 blocks 2.39 | 4.7 (hc 2.73 + out-proj 1.14 + qkv 0.84) | MMF tiles 32 rows: 320-row matrices run on 10 SMs (24.5 us vs 7.5 us at 1 token) |
| copies / state writeback | **1.61** (448 D2D memcpys: 5 rollback snapshots x 36 GDN layers, each a strided memcpy + copy) | 0.35 (copy+combine) | |
| norms + elementwise + rope | 1.67 | fused into Strata's stages | |
| attention + indexer + GDN | 0.71 | 1.31 | |

Conclusions: (1) **the expert cache's miss handling is the largest single difference** (+2.5 ms per verify, +0.6 ms per
plain step): llama.cpp moves ~2 MB per miss through the SMs inline, Strata computes misses on the CPU concurrently and
only waits. At this layout llama.cpp also sees more misses per verify (~1.4-2 per layer) than Strata reports (0.65
distinct / 0.78 routed per layer). (2) **Two dispatch problems are specific to the 5-token width**: MMF for few-row
BF16 matrices (2.4 ms, fixable by routing <4096-row, >=2048-col matrices to MMVF for up to 8 columns) and the
rollback snapshot copies (1.6 ms, `ggml_cont` + `ggml_cpy` per slot; a direct strided `ggml_cpy` is one kernel).
(3) Drafting costs 3 ms of GPU plus ~2 ms of CPU per round because each of the 4 draft steps is a separate decode
with a full-vocabulary head; Strata's vocab-subset head and single captured round avoid both.

**Fixes applied (commit 7ad8dae47 in `src\llama-merge`), measured in the 20 GB layout with `scripts/quick_hit.sh`
(4 x 512 tokens, MTP flags `-md mtp/...Q8_0.gguf --spec-type draft-mtp --spec-draft-n-max 4 --spec-draft-n-ubatch 64`):**

| build | MTP-4 tok/s | plain decode tok/s |
|---|---|---|
| before (nsys runs, same layout) | 105-116 | 100-106 |
| + MMVF for <=512-row BF16 at 2..8 columns, + direct strided `ggml_cpy` for the conv-state snapshots | 132-137 | 107-110 |
| + fixed-grid `moe_cache_fetch` (512 blocks looping over work items; empty launch ~3 us instead of 17) | 134-139 | 107-110 |

Outputs are unchanged (identical accepted/generated counts across runs; `test-backend-ops` MUL_MAT and CPY pass).
`GGML_CUDA_NO_MMVF_SMALL_ROWS=1` restores the MMF path for A/B tests. Expert cache statistics are now logged every
250 decode calls (`-lv 4`), so MTP runs report a hit rate too. Plain decode hit rate in this layout: 95.7%, with
80 GB uploaded over 2,048 tokens (39 MB per step = the ~1 ms per step of inline PCIe fetch).

What remains, in order of size: (a) miss handling: fetch misses outside the critical path (hits first, misses on a
second graph branch or on the CPU as Strata does), worth up to ~3 ms per verify and ~0.7 ms per plain step; (b) the
draft side: 4 full-vocabulary heads (1.6 ms) and 4 separate decodes with their CPU overhead (~2 ms) per round; (c)
~1 ms per step of launch gaps between ~1,700 small kernels (fusion of norm/elementwise chains).

## Update, 2026-11-03: MTP head-to-head on the same prompts, and why Strata misses fewer experts

Same three 1024-token prompts (`scripts/mtp_trace.sh` for llama.cpp, `scripts/strata_trace.sh` for Strata), 20 GB-class
expert cache on both, commit 7ad8dae47. Logs: `builds-bench/mtptrace-pmin0.log`, `mtptrace-pmin05.log`,
`compare-mtp-run.txt`, Strata's `launch\iq3_s-trace.engine.log`.

| | llama.cpp MTP-4, fixed width (`--spec-draft-p-min 0`) | llama.cpp MTP-4 with Strata's gate (`--spec-draft-p-min 0.5`) | Strata (`--spec 4 --spec-min-p 0.5`) |
|---|---|---|---|
| decode tok/s, 3 prompts | 144 / 139 / 159 | 84 / 94 / 102 | 199 / 203 / 169 |
| accepted per draft, mean accepted length | 0.51-0.55, 3.0-3.2 | 0.71-0.79, 3.2-3.4 | 0.57-0.81 (343/423, 260/419, 268/470) |
| expert cache hit rate (distinct lookups) | 93.4% | 92.5-93.2% | 95.1 / 95.1 / 97.7% |
| missed experts per layer per round | ~1.6 (75k misses / ~990 rounds / 48) | | 0.4-1.0 computed on the CPU |
| missed experts **skipped** (not computed) | 0 | 0 | **53,128 over the three prompts** (`route tail skip`) |
| bytes moved for misses | 144 GB (47 MB per token, ~4 ms per round inline) | | misses computed on the CPU, overlapped |

Findings:
- **Strata's gate does not transfer.** llama.cpp rebuilds and re-captures the CUDA graph whenever the verify width
  changes, so letting the draft length vary with the draft probability costs more than the better acceptance gains
  (84-102 vs 139-159 tok/s). Fixed width stays the right setting for llama.cpp.
- **Strata misses fewer experts because it does not compute all of them.** `STRATA_ROUTE_TAIL_SKIP=7` is on by default
  on CUDA (`generate.cpp` ~11174, `expert_source.cpp` ~4930): a non-resident expert that every token of the verify
  window routes at rank 7, 8 or 9 (the three smallest of the top-10 weights) is neither copied nor computed; its
  contribution is zero, without renormalising the others. Over these prompts that removed 53,128 expert computations,
  roughly 70% of what its misses would have been, which is why its logs show 0.65 "CPU experts" per layer-window
  against llama.cpp's 1.6. Strata's own docs measure the cost: KL 0.003-0.006 against the exact run, perplexity x1.005-1.011,
  and they turn it off when >=90% of the experts are cached. llama.cpp computes every routed expert exactly.
- The remaining misses Strata does compute run on the CPU pool concurrently with the GPU (it waits ~1.4-2 ms per
  5-token window); llama.cpp copies each miss over PCIe inside the graph (~55 us per expert, ~4 ms per round).
- So the two engines are not yet "the same strategy" in two places: tail skipping (an accuracy trade) and miss
  computation off the critical path (engineering). Together they account for most of the 139-159 vs 169-203 gap; the
  rest is the draft side (4 separate full-vocabulary draft decodes) and kernel launch gaps, per the nsys section.

Options from here: (a) implement the same optional tail skip in the device lookup (`moe_cache_assign` knows each id's
rank; a skipped miss gets weight 0 and a resident slot), expected to remove ~70% of the fetch cost in MTP mode at the
same small accuracy cost Strata accepts, off by default; (b) hits-first expert matmul with the fetch on a parallel
graph branch, exact, larger change.

## Update, 2026-11-03: tail skip and hits-first split implemented

Both are opt-in through environment variables (the server reads them at start):

- `LLAMA_MOE_CACHE_TAIL_SKIP=R` (Strata's `STRATA_ROUTE_TAIL_SKIP`, use 7): in the device lookup (`moe_cache_assign`), a
  missing expert that every token of the ubatch routes at rank R or lower is not fetched; its ids point at a zero
  expert kept in an extra bank slot, so the matmuls produce zero for it and the router weights are left as they are.
  The log reports "missing tail experts skipped". Needs the zero slot inside the matmuls' expert range: the cache view
  is now `n_slots + 1` experts wide (the MMQ kernel, used for IQ3_S/IQ2_S experts at 5 tokens, sizes its per-expert
  tables by that width; the first version crashed there with an illegal address).
- `LLAMA_MOE_CACHE_SPLIT=1` (Strata's "VRAM hits first"): the lookup writes two slot tensors, resident experts and
  missing experts (the other set pointing at the zero expert), forks the PCIe copy of the misses onto a side CUDA
  stream, and the graph runs the expert chain twice: resident experts immediately, then `ggml_moe_cache_join`
  (waits for the copy), then the missing experts, and adds the two. Exact: same experts, same weights. A CUDA graph
  chunk cannot end between fork and join (`moe_fork_open`).

**Tail skip, plain decode, teacher-forced 2,958 positions (`scripts/teacher_force.py`, exact vs skip 7):**

| | exact | skip 7 |
|---|---|---|
| expert misses over the run | 40,728 (77.7 GB uploaded) | 21,733 (41.4 GB), 28,353 skipped |
| hit rate | 96.9% | 98.3% |
| top-1 agreement with exact | | 97.94% (61 of 2,958 differ, all near-ties: median top-2 margin 0.11) |
| KL top-20 mean / median / p99 | | 0.0032 / 0.0005 / 0.032 |

That is the same distance as between llama.cpp and Strata (0.0033, 97.6%), and the same as Strata reports for its own
skip. In plain decode the skip halves the misses; speed numbers for MTP are below.

**Speed (commit 9a9f82d2d, same three 1024-token prompts, `scripts/mtp_trace.sh`; plain via `quick_hit.sh`):**

| | plain decode tok/s | MTP-4 tok/s (3 prompts) | misses in MTP (3 prompts) |
|---|---|---|---|
| exact (both off) | 107-110 | 144 / 139 / 159 | 76k |
| tail skip 7 | | **159 / 158 / 176** | 45k (+57k skipped) |
| split | 97-100 | 134 / 129 / 147 | 76k |
| split + tail skip 7 | | 145 / 146 / 160 | 45k |

- The tail skip gives +10% in MTP mode at Strata's accuracy cost; Strata on these prompts: 199 / 203 / 169.
- The split is exact (identical acceptance counts with and without it, teacher forcing 100% / KL 0) but slower, because
  the second chain still computes a zero expert for every resident pair (10 distinct zero slots, real VRAM reads), which
  costs about as much as the pass it was meant to overlap. To pay off it needs the expert kernels (`mul_mat_vec_q` with
  ids, `mul_mat_vec_q_moe`, and the MMQ id helper for 5-32 token ubatches) to exit early on zero ids and write zeros, or a
  compacted miss list (gather the few missing pairs, compute them, scatter-add). Left opt-in and documented as such.
- Debugging note: both features first crashed inside MMQ with an illegal address. Cause: several ids of one token pointed at
  the same zero slot, and `mm_ids_helper` counts duplicates in `nex_prev` but compacts one per token, so the expert
  bounds overlap. One zero slot per rank (slot `n_slots + rank`) keeps a token's ids distinct. Use `CUDA_LAUNCH_BLOCKING=1`
  to get the faulting kernel instead of a later cuBLAS "internal operation failed".

## Update, 2026-11-03: ablation of Strata's two miss tricks, same prompt on both engines

Bench mode on both sides (`scripts/strata_bench.sh`, `scripts/llama_bench_tf.sh`): the 89-token teacher-forcing prompt,
600 greedy tokens, Strata `--spec 4 --spec-min-p 0.5`, llama.cpp MTP-4 fixed width, 20 GB-class caches. Strata's
`--no-pool` exits with code 2 in this configuration, so "no CPU compute" is `--pcie-frac 1.0` (every miss read by the
GPU over PCIe, as llama.cpp does). Logs: `builds-bench/strata-bench-*.log`, `llama-bench-*.log`, `*-ablation.txt`.

| configuration | Strata tok/s | llama.cpp tok/s |
|---|---|---|
| tail skip on (Strata default on CUDA serve; `LLAMA_MOE_CACHE_TAIL_SKIP=7`) | **204.3** (0.45 misses/layer-window) | **161.4** |
| tail skip off, misses computed on the CPU pool | 187.7 (0.96 misses/layer-window) | n/a |
| tail skip off, misses over PCIe by the GPU (both tricks off) | 181.0 | 135.9 |

- On Strata the two tricks are worth 13% together: the tail skip 8-9%, the CPU pool about 3.5% over reading misses over
  PCIe. On llama.cpp the tail skip is worth 19%, because its misses are on the critical path.
- **With both tricks off on both sides Strata is still 33% faster (181 vs 136), with both on 27% (204 vs 161).** So the
  miss handling is not where most of the remaining gap is. Per round: Strata 600 tokens in 244 windows = 2.46 tokens per
  window at 12.0 ms per window; llama.cpp 2.97 tokens per round at 18.4 ms per round. llama.cpp accepts more per
  round and loses on the cost of the round, which the nsys section attributes to the 5-token verify graph (dense
  kernels, state copies, launch gaps) and to the four separate full-vocabulary draft decodes with their CPU overhead.
- Note on the earlier stage comparison: Strata's bench mode does not apply the tail skip by default (serve mode does), so
  the T1/T5 stage tables in the nsys section were measured with the skip off.

## Update, 2026-11-03: why Strata still wins with both miss tricks off (181 vs 136)

Same 89-token prompt, 600 greedy tokens, both tricks off on both sides (`STRATA_ROUTE_TAIL_SKIP=0 --pcie-frac 1.0`;
llama.cpp exact). Logs: `builds-bench/strata-bench-off-t5.log`, `strata-bench-off-t15.log`, `llama-bench-nmax*.log`,
`nsys-mtp4b*` (llama.cpp round profile on the current build).

| | tokens per round | ms per round | tok/s |
|---|---|---|---|
| Strata, gated widths (`--spec 4 --spec-min-p 0.5`: windows T1 26, T2 29, T3 46, T4 137) | 2.51 | 13.9 | **181** |
| Strata, width fixed at 5 tokens (`STRATA_SPEC_T_CYCLE=5`) | 2.46 | 16.9 (GPU stages 14.84, drafting 1.05) | 146 |
| llama.cpp, 4 drafts fixed (5-token verify) | 2.97 | 21.9 (verify graph 16.4, drafts 2.8, other GPU 0.4, idle 2.6) | 136 |
| llama.cpp, 3 drafts fixed | 2.84 | | **146** |
| llama.cpp, 2 drafts / 5 drafts fixed | 2.40 / 3.20 | | 142 / 133 |

1. **The gate is worth 24% to Strata (146 -> 181).** With the draft-probability gate most windows are 4 tokens and a
   fifth are 1-3 tokens, the yield per window stays 2.5 tokens but the window costs 13.9 ms instead of 16.9. Strata can
   do this because it captured one CUDA graph per width up front; llama.cpp re-captures on every width change (84-102
   tok/s with the same gate). Multi-width graphs are therefore the largest remaining item, not a 5-10% one.
2. **At a fixed width llama.cpp is on par once the draft length matches the acceptance: 3 drafts give 146 tok/s, equal
   to Strata's fixed-5 run.** llama.cpp accepts more drafts per round (2.84-2.97 vs 2.46 at the same 4 drafts; its MTP
   head is Q8_0, Strata's Q2_0) but pays more per round.
3. Where the 5 ms per round went at equal width (21.9 vs 16.9), from the nsys profile against Strata's T5 stage table:
   - verify GPU: 16.4 vs 14.45 ms (+1.9): misses over PCIe 4.9 vs 3.9 (Strata still orders hits before the PCIe group),
     expert hits 3.2 vs 2.6, dense/attention/GDN equal at ~6.1, and about 2.7 ms of norms, elementwise ops and copies plus
     launch gaps that Strata fuses into its stages (its copy+combine is 0.26);
   - drafting: 2.8 vs 1.05 ms (+1.75): four separate draft decodes with the full 248k-vocabulary head against one
     captured MTP pass with a vocabulary subset;
   - CPU-side idle: 3.0 vs 1.0 ms (+2.0): per-round sampling, five graph launches and the draft context's own overhead.

Practical setting now: `--spec-draft-n-max 3` (and `LLAMA_MOE_CACHE_TAIL_SKIP=7` if Strata-level accuracy is acceptable).

## Update, 2026-11-03: per-width captured graphs (LLAMA_GRAPH_WIDTH_CACHE=1)

Implemented in `llama_context`: decode ubatches of up to 8 tokens each get a slot with its own `ggml_backend_sched`
(own compute buffers), its own pair of `llm_graph_result`s and reuse state (`width_slots`, `sched_cur`). The CUDA
backend already keys captured graphs by the graph's first node, so each slot keeps its own captured graph. Slots are
dropped whenever the main graph results are reset. Both the main and the draft context use them. VRAM: about 100 MiB
for 5 widths at 32k context. Outputs identical.

| 89-token prompt, 600 tokens, exact | tok/s |
|---|---|
| fixed 4 drafts, with or without the cache | 136.5 |
| gate `--spec-draft-p-min 0.5`, no cache (graph rebuilt and re-captured every width change) | 82.7 |
| gate 0.5, with the cache (203 graphs reused in ~200 rounds) | 133.4 |
| gate 0.5 + cache + tail skip 7 | 143.3 (fixed 4 drafts + tail skip: 161.4) |

The re-capture penalty is gone, but **the gate does not pay in llama.cpp the way it pays in Strata (146 -> 181)**: a
4th draft is accepted 29% of the time (acc per pos 0.78 / 0.57 / 0.40 / 0.29), and dropping it saves only the
verify's marginal width while each draft step (0.7 ms) and the per-round CPU work (2.6 ms) are paid regardless; at
5.4 ms of fixed per-round overhead the narrower rounds roughly break even. Strata's round overhead is about 2 ms, so
for it narrow windows are cheap. Conclusion: fixed `--spec-draft-n-max 3` remains the best llama.cpp setting, and the
next lever is the per-round overhead (draft steps and CPU idle), after which the gate and the width cache become
useful. The cache is kept opt-in.

Branch pushed: `https://github.com/erokhins/llama.cpp` branch `moe-cache-perf` (remote `erokhins` in `src\llama-merge`).

## Update, 2026-11-03: drafter optimisation

Per round the draft side cost 2.8 ms of GPU (4 draft steps of 0.61 ms, 0.39 of each the 248k-vocabulary head GEMV) plus
CPU work per step (logits copy to the host, sampling). Same 89-token prompt, 600 tokens, exact unless noted
(`scripts/llama_bench_tf.sh`, logs `builds-bench/llama-bench-*.log`, `draftopt-run*.txt`):

| setting (4 drafts unless noted) | tok/s | accepted / generated |
|---|---|---|
| baseline | 136.5 | 397 / 807 |
| `--spec-draft-backend-sampling` (draft argmax on the GPU) | 135.9 | same |
| + `--backend-sampling` (main model too) | 138.9 | same |
| `LLAMA_MTP_VOCAB=98304` (draft head over the first 98k ids, rest -inf) | 141.1 | 396 / 811 |
| `LLAMA_MTP_VOCAB=65536` / `32768` | 137.6 / 134.7 | 390 / 831, 381 / 867 (acceptance loss outweighs the saving) |
| vocab 98k + both backend sampling flags | 144.9 | 396 / 811 |
| **3 drafts + vocab 98k + backend sampling (best exact)** | **156.4** | 388 / 631 |
| best + `LLAMA_MOE_CACHE_TAIL_SKIP=7` | **170.0** | 384 / 645 |

- `LLAMA_MTP_VOCAB` (commit 126eced13) is a view of the first N rows of the draft model's output matrix plus a -inf
  pad, so logits keep their shape; BPE ids are frequency ordered and ids < 98304 cover 99.7% of the generated tokens, so
  acceptance is unchanged at that cut. Smaller cuts lose drafts faster than they save GEMV time.
- Backend sampling exists upstream (`-bs` / `--spec-draft-backend-sampling`) and removes the 1 MB logits copy and the
  CPU argmax per step; outputs identical.
- Against Strata on this prompt: exact llama.cpp 156 vs Strata with both miss tricks off 181 (its draft gate alone is
  worth 24% there, see above); with the tail skip 170 vs Strata's default 204. The remaining difference is the
  per-round CPU idle (~2.6 ms vs ~1 ms) and the four separate draft decodes; fusing the draft steps into one captured
  pass is the next structural item, after which the gate and the width cache become profitable here too.

Recommended llama.cpp settings now: `-c 32768 -np 2 -b 2048 -ub 2048 --moe-cache-mib 24000 -md mtp-...Q8_0.gguf
--spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-n-ubatch 64 --backend-sampling --spec-draft-backend-sampling`
with `LLAMA_MTP_VOCAB=98304`, plus `LLAMA_MOE_CACHE_TAIL_SKIP=7` when Strata-level accuracy is acceptable.

## Update, 2026-11-03: host-side profile of a decode step, llama.cpp vs Strata

Instrumentation (commit 21ff027de, all env-gated): `LLAMA_DECODE_TIMING=1` (phases of `llama_decode` per context, GPU
wait in `synchronize()`), `GGML_SCHED_TIMING=1` (inside `ggml_backend_sched_compute_splits`), `GGML_CUDA_GRAPH_TIMING=1`
(CUDA graph update check / execute, eager vs capture vs replay). Run: 89-token prompt, 1200 tokens, 3 drafts, draft
vocab 98k, backend sampling, width cache on. Logs `builds-bench/llama-bench-dectiming*.log`, `emb-*.log`.

**One draft decode (the MTP context), steady state, host microseconds:**

| phase | us | note |
|---|---|---|
| batch init + memory context + apply | ~45 | KV slot search for 1 token |
| graph reuse check, inputs (`tensor_set` into host buffers) | ~8 | reused with the width cache (52 us of rebuild per call without it) |
| scheduler compute, of which: | ~130 | |
| - CUDA graph replay (update check 3.6 + launch 8) | 12 | cheap |
| - split input upload: pre-sync 37 + `cudaMemcpyAsync` 69 + post-sync 21 | ~125 | **the inputs live in a host buffer and are uploaded by the scheduler every step** |
| outputs (sampled token copy) | 12 | |
| GPU wait in `synchronize()` | 314 | the draft graph's GPU time; sequential dependency, cannot overlap |
| **total per draft step** | **~0.5 host + 0.31 GPU wait ≈ 0.65 ms** | Strata: ~0.3 ms per draft step all-in (1.05 ms/round for 3-4 drafts) |

Moving `token_embd.weight` to the GPU (`-ot/-otd token_embd.weight=CUDA0`) removes the CPU split (1 split instead of 2)
but not the upload: the remaining inputs (token ids, positions, the hidden state handed over from the main model) are
still host tensors copied per step, so the cost is unchanged (and 1.5 GB of cache is lost; not worth it).

**The main context per verify decode:** memory context 285 us (KV slot search for 4-5 tokens), inputs 87, outputs 45,
scheduler ~150 besides the GPU time; graph rebuilds are the expensive part, see below.

**CUDA graph warmup cost under Windows WDDM.** The backend captures a graph only after two consecutive stable
executions; the executions before that run eagerly, and an eager launch costs about 19 us of host time here. The
small draft graph (~100 kernels) costs ~2.3 ms per eager run, the main graph (~2,300 kernels) ~43 ms, plus ~5 ms per
capture. Any change of graph properties pays that: in this run 164 eager small-graph runs and 9 eager large-graph
runs (the KV cache grows in steps of 256 cells, `n_pad = 256` in `llama-kv-cache.cpp`, and every growth changes the
graph for every width slot of both contexts), about 0.45 s of 7.6 s, 6%.

**Comparison with Strata's host side** (its own counters, bench mode): plan 0.05 ms, dispatch ~0.06, commit 0.009,
host 0.02 per round; drafting 1.05 ms per round including GPU; windows captured once per width at start-up and never
rebuilt (its KV is resident/streamed without changing the graph). Per round llama.cpp spends about 2.6 ms of GPU-idle
host time where Strata spends about 1 ms, and the difference is made of many small CUDA API round trips (uploads and
syncs at 20-70 us each on WDDM), not of one slow stage.

**Done (commit after a8812af73):** `GGML_CUDA_GRAPH_WARMUP=0` (capture a changed graph immediately) and `LLAMA_KV_PAD=N`.
1200 tokens, same settings: 158.2 (default) -> 161.2 (warmup off) / 160.0 (pad 2048) / **162.3 (both)**; eager runs of the
large graph 9 -> 0 and of the small graph 164 -> 0. Smaller than the 6% estimated: the rebuild storms were real but the KV
padding was not their main trigger (pad 2048 only cut the large-graph rebuilds 9 -> 7, the small-graph ones 164 -> 158), so
most of the draft context's property changes come from elsewhere (to be found: likely n_outputs / recurrent-state inputs of
the update ubatches). Both switches are opt-in with upstream defaults.

Proposals, in order of value per effort:
1. (done, see above) Capture CUDA graphs after one stable execution instead of two, or immediately for graphs the width cache owns
   (`warmup_complete` logic in `ggml-cuda.cu`): saves ~45 ms per main-graph change; env-tunable.
2. Rebuild less often: raise the KV padding from 256 cells so the graph changes every few thousand tokens, or keep the
   graph independent of `n_kv` for the small ubatches.
3. Device-resident inputs for the small decodes: place the draft's input tensors on the GPU (one synchronous
   `cudaMemcpy` of a packed input block instead of the scheduler's sync + async upload + sync), and hand the main
   model's hidden state to the draft device-to-device instead of through host memory. ~0.65 ms per round.
4. Then the draft graph's own GPU time (0.31 ms vs Strata ~0.2): smaller MTP experts (Q4) if acceptance allows.

## Machine

- RTX 5090 32 GB (driver 610.88, CUDA UMD 13.3).
- Ryzen 9 9950X3D, 189.6 GB RAM.
- Windows 11.
- No compiler toolchain: no CMake, MSVC or CUDA toolkit. A winget install was started and stopped before anything
  installed.

## Model

- Qwen3.8-Flash-Next, GSQ-RCO IQ3_S quant (ISTA-DASLab), arch `qwen4exp`.
  - 512 experts, 10 active; hyper-connections, PLE n-gram table, sparse-attention indexer, SSM layers.
- Files, both engines use them unchanged:
  - `C:\Users\erokh\AppData\Local\Strata\Strata-data\models\IQ3_S\Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf`
    (54.8 GB: trunk and experts).
  - `...-00002-of-00002.gguf` (28.8 GB: the PLE n-gram table `per_layer_token_embd`).
- Contains 9 `Q2_0` tensors (ggml type 42) on `ffn_down_exps`. Mainline llama.cpp supports Q2_0 since July 2026.
- API key used for both servers: `sk-strata-9MmYQgcDgRmmDjoHwwgktuWCCiALwvDM`.

## Setup

### Strata (original server)

- Strata Terminal 0.1.55 and engine `components\engine\0.1.55-windows-x86_64-cuda-cuda13\strata.exe`.
- Profile `launch-iq3_s-128k`; config `C:\Users\erokh\AppData\Local\Strata\launch\iq3_s-128k.json`.
- Engine args: `--spec 4 --spec-min-p 0.5 --mtp <Strata-data\mtp\rt> --expert-profile ... --expert-cache auto
  --kv int8 --kv-resident 32768 --prefill auto:32768 --max-context 131072 --batch 4`.
  The HTTP server runs with `--parallel 4`.
- Start time: the first-ever start took 1008 s (one-time tuning). Later starts take **~26 s**.
- Start: `C:\Users\erokh\AppData\Local\Strata\terminal\current\python\python.exe C:\Users\erokh\AppData\Local\Strata\terminal\current\strata.py runtime start --profile launch-iq3_s-128k --wait 1800`
  (`--wait` must be ≤ 1800).
- Stop: same command with `runtime stop --profile launch-iq3_s-128k`. Status: `runtime status --profile ...`.
- A `strata launch` TUI process (from the user's original session) may still be open. It's harmless.

### llama.cpp

- Prebuilt mainline **b11960** (= v0.8.0), Windows CUDA 13.4 x64, in `llamacpp-b11960\bin\`.
  Zip SHA-256s were verified against GitHub's digests.
- Base command used for every run (see `scripts/run_config.sh`):

  ```
  llama-server.exe -m <shard1.gguf> --host 0.0.0.0 --port 8080 --api-key <key> --alias qwen3.8-flash-next-iq3_s \
    -c 131072 -np 4 --kv-unified -ctk q8_0 -ctv q8_0 -fa on --jinja -t 16 \
    -b 4096 -ub 4096 --load-mode none   [+ extra flags]
  ```
- Best single-user extras: `--moe-cache-mib 8192 -md C:\Users\erokh\testing\mtp\mtp-Qwen3.8-Flash-Next-Q8_0.gguf --spec-type draft-mtp --spec-draft-n-max 2`.
- Best for concurrent serving: `--moe-cache-mib 12288`, without MTP.
- MTP head: Unsloth's **self-contained** `MTP/mtp-Qwen3.8-Flash-Next-Q8_0.gguf` (3.9 GB) in `mtp\`.
  SHA-256 `58a47d6a…8d1` matches HF. The "shared" variants need Unsloth's fork; the self-contained one works in mainline.
- Load time is ~14 s. `--fit` (on by default) places expert layers automatically.

## Results

### Throughput (`scripts/bench.py`; tok/s)

| Config | Gen, short prompt | Gen after 30k prompt | Prompt read, 30k | 4 requests at once |
|---|---|---|---|---|
| **Strata** | **183** | **172** | **7,260** (server-side) | **161** |
| llama.cpp, defaults | 45 | 42 | 393 | 60 |
| + `--load-mode none -b/-ub 4096` | 41.5 | 39 | 2,780 | 56 |
| + 4 GB expert cache | 70.4 | 63 | 1,830 | 90 |
| + 8 GB expert cache | 75.7 | 68 | 1,570 | 104 |
| + 12 GB expert cache | 79.3 | 71 | 1,400 | 114 |
| 8 GB cache, `-ub 2048` | 80.5 | 70.6 | 1,140 | 116 |
| 8 GB cache + MTP, 1 draft token | 75.3 | 65 | 1,400 | 90 |
| **8 GB cache + MTP, 2 draft tokens** | **86.2** | **72.7** | 1,410 | 74 |
| 8 GB cache + MTP, 3 draft tokens | 83.5 | 71 | 1,400 | 62 |

- **The expert cache is the big lever.** It costs prompt-reading speed, because it takes VRAM that `--fit` would
  otherwise spend keeping expert layers on the GPU.
- **MTP helps little here.** Drafts are accepted at a normal rate (67–81%), but checking several tokens at once
  routes to many different experts, all read from CPU RAM. It hurts concurrent throughput, so only use it single-user.
- **Measurement caveat:** `bench.py` used `http://localhost`. On Windows that adds **~2 s per request** (IPv6 `::1` is
  tried first, but the servers listen on IPv4 only). Affected and unaffected numbers:
  - Affected: client-side time to first token and client-measured prompt-reading rates; most of the "prompt read"
    column is client-measured.
  - Unaffected: generation speeds and server-side prompt speeds.
  - **Use `127.0.0.1` from now on.**

### Why Strata is faster (inferred from logs, not measured directly)

1. **Expert placement.** About 94% of Strata's expert lookups hit VRAM. Its misses still run on the GPU, with weights
   read over PCIe, and its cache starts warm from `expert-profile.bin`. ~~llama.cpp computes every miss on the CPU,
   bound by DDR5 bandwidth~~ (**wrong, see the update at the top**: llama.cpp also uploads misses over PCIe and
   computes on the GPU; its cost was per-layer host round trips and a cold, unseeded cache, which PR #268 fixes).
2. **Speculative decoding.** Strata drafts 4 tokens with 70–80% acceptance. Because everything is on the GPU, checking
   them is almost free.
3. **Prompt reading.** Strata uses chunks of up to 32k tokens. It also keeps only 32k tokens of KV cache in VRAM and
   streams the rest from RAM, which leaves more VRAM for experts.

### Same model? Correctness checks

- **Static:**
  - Strata reads experts and quantized dense tensors directly from the GGUF, by byte offset.
  - All 777 BF16/F32 tensors it copies into `packs\iq3_s\dense.bin` are byte-identical to the GGUF
    (`scripts/verify_pack.py`).
  - Its chat template is identical to the GGUF's.
- **Free greedy generation** (`scripts/gen_compare.py`, 8 prompts, temperature 0, compared after whitespace trim):

  | Pair | Identical outputs |
  |---|---|
  | llama.cpp vs itself | 7/8 |
  | llama.cpp MTP+cache vs plain | 3/8 |
  | llama.cpp vs Strata | 3–4/8 |
  | Strata vs itself | 3/8 |

  Divergences always start at near-ties. Factual answers were identical in all runs. Strata keeps a trailing `\n` on
  reasoning text and llama.cpp strips it, so compare after trimming.
- **Teacher-forced next token** (`scripts/teacher_force.py`):
  - Setup: an 89-token prompt plus a 2,958-token greedy reference. Every prefix was fed to both engines as raw token
    IDs, and each returned its top-20 log-probabilities.

  | Metric | llama vs llama | llama vs Strata |
  |---|---|---|
  | top-1 agreement | 100% | 97.6% (72 disagreements) |
  | KL, top-20: mean / median / 99th percentile | 0 | 0.0033 / 0.0005 / 0.033 |
  | difference in forced-token log-probability: median / 90th / 99th percentile | 0 | 0.004 / 0.07 / 0.22 |
  | perplexity on the reference | 1.2884 | 1.2911 (Strata) |

  - Disagreements happen where llama.cpp's top-2 log-probability gap has a median of 0.08 (max 0.65).
  - Mean KL stays flat (0.0024–0.0039) across 500-token windows, so there's no context-dependent drift.
  - **Verdict: the same model; the differences are only numerical.**

## Important findings

- **Expert cache and MTP are both in mainline already.**
  - Expert cache: `--moe-cache-mib`, from #29887 and #30637. It supersedes PR #27861.
  - qwen4exp MTP: #29761. It supersedes PR #27836.
  - The clone in `src\llama.cpp` (checked out at `b11960`, PR branches fetched as `pr-mtp` and `pr-expcache`) is
    **unused**. Both PR merges conflicted and were aborted.
- **Strata's HTTP API can't do teacher forcing.** It has only chat endpoints, no raw prompt or prefill, and logprobs
  under `--parallel` only work with `max_tokens 1`.
  - `teacher_force.py strata` spawns `strata.exe --serve` itself (the server's args minus `--batch 4`).
  - It drives the stdin protocol directly: send `GEN 1 logprobs=20 id,id,...`, get back `T <id>` / `LP <lp> id:lp ...`
    / `DONE`.
  - The HTTP server must be stopped first.
- llama.cpp `/completion` accepts token-ID arrays (`"prompt": [ids]`) with `n_probs`. Its `/tokenize` and
  `/apply-template` endpoints build the reference prompt.

## Files

| Path | What |
|---|---|
| `scripts/bench.py` | throughput benchmark (decode, prefill at 4k/15k/30k, long-context decode, 4-request concurrency) |
| `scripts/run_config.sh NAME [flags]` | start llama.cpp with the base config plus flags, benchmark it, leave it running |
| `scripts/gen_compare.py` | free greedy generation, collect and compare (`nolp` arg for Strata) |
| `scripts/teacher_force.py` | `ref` / `llama` / `strata` / `compare` teacher-forced pipeline |
| `scripts/verify_pack.py` | byte-compare Strata's `dense.bin` against the GGUF |
| `scripts/gguf_info.py` | GGUF header and tensor-type dump |
| `scripts/run_bin.sh BIN_DIR NAME [flags]` | like run_config.sh for any build dir; 127.0.0.1; output in `builds-bench/` |
| `scripts/quick_tg.sh BIN_DIR NAME [flags]` | 1-minute decode sanity check of a build |
| `scripts/srv.py` | start/stop helper used by the two scripts above |
| `scripts/compare.py LABEL=results.json ...` | side-by-side table of bench results |
| `src/llama-merge/` | **PR #268 merged onto b11960, the build to use**; `src/llama-base`, `src/llama-pr268`: A/B trees |
| `C:\Users\erokh\tools\` | per-user MSVC/CUDA/CMake toolchain, `buildenv.sh`, `fetch_msvc.py` |
| `builds-bench/results-*.json`, `builds-bench/server-*.log` | this update's benchmark outputs and logs |
| `results-strata.json`, `results-llamacpp-*.json` | benchmark outputs |
| `gen/*.json` | free-generation outputs |
| `tf/ref.json`, `tf/llama.json`, `tf/llama-2.json`, `tf/strata.json` | teacher-forcing data |
| `llamacpp-b11960/server-*.log` | llama.cpp server logs, one per config |
| `mtp/mtp-Qwen3.8-Flash-Next-Q8_0.gguf` | MTP draft head for llama.cpp |

## How to start things

- **Strata:** `strata.py runtime start --profile launch-iq3_s-128k --wait 1800` (full path above); ~30 s.
- **llama.cpp, best config now (PR #268 on b11960, no MTP):**
  `bash scripts/run_bin.sh C:/Users/erokh/testing/src/llama-merge/build/bin NAME --moe-cache-mib 12288`
  runs the benchmark and stops the server. To only serve, copy the command from `scripts/run_bin.sh`
  (`--host 127.0.0.1`). Env knobs of the PR: `GGML_SCHED_PREFETCH=0`, `LLAMA_MOE_CACHE_DEVICE=0`, `LLAMA_MOE_CACHE_SEED=0`.
  Rebuild after source changes: `source /c/Users/erokh/tools/buildenv.sh; ninja -C src/llama-merge/build llama-server`.
- **llama.cpp prebuilt b11960, old best single-user config:**
  `bash scripts/run_config.sh mycfg --moe-cache-mib 8192 -md C:/Users/erokh/testing/mtp/mtp-Qwen3.8-Flash-Next-Q8_0.gguf --spec-type draft-mtp --spec-draft-n-max 2`.
  This also runs the benchmark. To start it without benchmarking, copy the base command above.
- Only one engine fits on the GPU at a time. Stop one before starting the other: `taskkill //IM llama-server.exe //F`,
  or `runtime stop` for Strata.

## Possible next steps

- Re-run `bench.py` against `127.0.0.1` for clean time-to-first-token and client prompt-reading numbers.
- Tune llama.cpp's prompt-reading vs generation trade-off: try `--fit off -ncmoe N` together with the expert cache.
- Watch upstream llama.cpp. GPU-side computation of expert-cache misses, or a warm-started cache, would close most of
  the gap.
- Teacher-force a second, different text (code, or a non-model-generated document) so the reference isn't one that
  llama.cpp generated itself.
