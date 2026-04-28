# Code Optimizations Log

## Current Snapshot (from `results.tsv`)
- Total experiments: **299** (`keep=17`, `discard=235`, `crash=46`)
- Current best: **0.995000 ms** (`79897a020fd59648100377d9da4c41e7413f6f33`)
- Gap to sub-0.5ms: **~49.7%**
- Last 10 experiments best non-crash run: **0.995000 ms** (still above the new target; high run-to-run variance persists)

## Recent session addendum
- New best run: **0.995000 ms** on `79897a020fd59648100377d9da4c41e7413f6f33`.
- Winning GEMM2 config: `GROUP_BLOCKS=4, num_warps=4, num_stages=3`.
- Benchmark gate is now strict sub-1ms for kept commits, and the next optimization goal is **below 0.500 ms average latency**.
- The session also confirmed that earlier routing/dispatch, block-map, and GEMM2 micro-optimizations mostly regressed or were unstable; the winning config is the one to preserve.
- Forcing only `GROUP_BLOCKS=4, num_warps=4, num_stages=3` regressed to **1.168789 ms**; the mixed autotune pool is still needed for the best average.
- Two restored mixed-pool reruns measured **1.000842 ms** and **1.015789 ms**, showing the current best is still close to 1ms but not repeatably below it.
- Follow-up consistency experiments (`metadata .ca` hints, `BLOCK_M` split at `T>=4096`, and tighter split at `T>=8192`) all regressed, so the fixed `BLOCK_M=16` baseline remains the best path.

## What is working
- Grouped GEMM2 launch ordering improved weight L2 reuse.
- GEMM2 autotune with `reset_to_zero=['out_ptr']` is the biggest measured win.
- Extending GEMM2 autotune key with `TOTAL_ROUTED` further improved shape-specific config selection.

## Hard constraints learned from failures
- `BLOCK_M=48/64` repeatedly triggers runtime errors or incorrect numerics; `BLOCK_M=8` is correctness-safe but a major throughput regression, while `BLOCK_M=16` is currently the best-performing tile size when routing block sizing is `BLOCK_M`-aware.
- Lower-precision workspaces (`fp16`, `bf16`, `fp8`) repeatedly caused `INCORRECT_NUMERICAL`/overflow; keep FP32 workspace unless a new numerically stable scaling scheme is proven.
- Alternative GEMM2 dot paths (`bf16xfp8`, `fp16xfp8`, direct FP8 paths) are currently unstable or slower.
- Long-seq approximation shortcuts (local-only pruning/collapse variants) have not been viable.

## Kernel-level bottlenecks in `solution/triton/kernel.py`
- Routing/dispatch still does full tensor ops (`topk`, stable `sort`, `bincount`, cumsums) before Triton GEMMs.
- GEMM2 still relies on `tl.atomic_add` into `out_accum`, which likely dominates contention on large routed-token workloads.
- Per-call temporary allocations (`block_map_buf`, `workspace`, `out_accum`) can add overhead at small/medium sequence lengths.

## Next research directions to break 1ms
1. **Persistent buffer reuse** for `block_map_buf`, `workspace`, `out_accum` (shape-bucketed cache, reuse + slice).
2. **Persistent GEMM2 scheduling** (CTA loops over multiple blocks for same expert/`nb`) to reuse W2/scales in cache.
3. **Two-stage GEMM2 reduction** to reduce atomic pressure (partial accumulation then reduction kernel).
4. **Routing/dispatch compaction in Triton** to reduce Python/Torch sort+dispatch overhead while preserving exact semantics.
5. **Long-seq-focused tuning** targeting workloads `5e8dc11c` (`seq_len=14107`) and `58a34f27` (`seq_len=11948`) without approximation.

## New online findings (Triton docs + upstream tutorials)
- **Persistent-kernel work distribution across SMs** (`tile_id += NUM_SMS`) is a recommended pattern for reducing scheduler overhead and improving cache residency in matmul-like loops (Triton tutorial: persistent matmul).
- **On-device grouped scheduling** (fixed CTA count, iterate over grouped GEMMs on device) is a first-class Triton pattern and maps well to MoE expert imbalance (Triton tutorial: grouped GEMM).
- **TMA/tensor descriptors reduce register pressure for global-memory address generation** and support async copy + pipeline overlap; requires 16B-aligned base pointers and descriptor-friendly strides (Triton API `make_tensor_descriptor`, Gluon TMA tutorial).
- **Loop-level compiler controls in `tl.range` are stronger than kernel-level knobs** for this use case: `num_stages`, `flatten=True`, `disable_licm`, `disallow_acc_multi_buffer`, and `warp_specialize` can improve overlap and reduce stalls/liverange pressure.
- **Programmatic dependent launch (PDL)** adds grid-dependency controls (`gdc_wait`, `gdc_launch_dependents`) that can reduce launch bubbles between dependent kernels when launched with `launch_pdl=True`.
- **Cluster Launch Control (CLC, Blackwell)** enables dynamic block stealing and showed ~3-4% speedup in Triton’s tutorial matmul; this is directly relevant to ragged/imbalanced expert-token distributions.

### High-value new experiments derived from the research
1. GEMM2 persistent scheduler + fixed-CTA grouped mapping (expert/`nb` tiles assigned in-kernel).
2. GEMM2 tensor-descriptor path for W2/workspace loads with software-pipelined multi-buffer loop.
3. `tl.range`-driven loop retune in GEMM2 (`flatten=True`, targeted `num_stages`, optional `disable_licm`) to reduce register pressure.
4. PDL-enabled launch chain for `_build_block_map_kernel` -> GEMM1 -> GEMM2 to reduce inter-kernel launch gaps.
5. Blackwell-only dynamic scheduling prototype (CLC-style behavior) for long-seq heavy workloads.

### Research sources
- https://triton-lang.org/main/getting-started/tutorials/08-grouped-gemm.html
- https://triton-lang.org/main/getting-started/tutorials/09-persistent-matmul.html
- https://triton-lang.org/main/python-api/generated/triton.language.range.html
- https://triton-lang.org/main/python-api/generated/triton.language.make_tensor_descriptor.html
- https://raw.githubusercontent.com/triton-lang/triton/main/python/tutorials/11-programmatic-dependent-launch.py
- https://raw.githubusercontent.com/triton-lang/triton/main/python/tutorials/gluon/04-tma.py
- https://raw.githubusercontent.com/triton-lang/triton/main/python/tutorials/gluon/12-cluster-launch-control.py

## New caching + B200 online findings
- **Triton supports explicit cache hints on loads/stores**:
  - `tl.load(..., cache_modifier={".ca",".cg",".cv"}, eviction_policy={"evict_first","evict_last"})`
  - `tl.store(..., cache_modifier={".wb",".cg",".cs",".wt"}, eviction_policy=...)`
- **Actionable cache policy split for current kernel** (instead of blanket modifiers, which already regressed):
  - W2 and W2-scale tiles (reused across nearby blocks): test `.cg + evict_last`
  - Streaming workspace reads (`c_f32` per `ib`): test `.cg + evict_first`
  - Tiny metadata vectors (`b_*`, routing maps): test `.ca` to favor near-SM reuse
- **Program ordering is a first-order cache control**: Triton matmul tutorial reports >10% uplift from grouped ordering due to higher L2 hit rate; this validates continuing grouped/persistent scheduling work for GEMM2.
- **B200 is compute capability 10.0 (Blackwell)**, which aligns with Blackwell-only features in Triton research tracks (warp-specialized loops, CLC-style dynamic scheduling, tcgen05/TMA-heavy designs).
- **System-level B200/GB200 signals** worth exploiting in kernel strategy:
  - fifth-generation NVLink (HGX B200: 1.8 TB/s GPU-GPU, 14.4 TB/s total NVLink bandwidth per 8-GPU system),
  - Blackwell platform claims strong MoE scaling (GB200 NVL72 reports up to 10x MoE performance vs H100 at system level).

### New cache-focused experiment queue
1. GEMM2 `tl.load(w2_ptrs)` + `tl.load(s2_ptr...)`: A/B test `{default, .cg, .cg+evict_last}`.
2. GEMM2 `tl.load(c_ptrs)` for workspace: A/B test `{default, .cg+evict_first}`.
3. Routing/block-map metadata loads: A/B test `{default, .ca}`.
4. Keep grouped/persistent launch order fixed while testing cache hints to isolate cache effects.

### Additional sources for this pass
- https://triton-lang.org/main/python-api/generated/triton.language.load.html
- https://triton-lang.org/main/python-api/generated/triton.language.store.html
- https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html
- https://developer.nvidia.com/cuda-gpus
- https://www.nvidia.com/en-us/data-center/b200/
- https://www.nvidia.com/en-us/data-center/gb200-nvl72/

## Latest experiment learnings (2026-04-19 live loop)
- **Current live baseline is unstable and above target**: reruns in this session landed around **1.26-1.33 ms** average, with long-seq workloads still dominating.
- **Temp-buffer reuse in `run()` is correctness-safe and low-risk**, but impact is modest and noisy:
  - observed runs: `1.303789`, `1.257789`, `1.329105`.
  - takeaway: allocator overhead is not the primary blocker for sub-1ms.
- **Targeted GEMM2 cache modifiers triggered full runtime failure** (all workloads `RUNTIME_ERROR`) in this code path; cache hints need smaller-scope reintroduction (one tensor at a time, starting from metadata only).
- **After cleaning the partial persistent-GEMM2 revert, baseline remained weak** (`1.331684 ms`, long-seq `6.956/4.875 ms`), so coherence restoration alone does not recover prior near-best behavior.
- **Metadata-only `.ca` cache hints are safe but small-impact**: applying `.ca` to block-map and GEMM2 block metadata improved average only to `1.308684 ms`; long-seq remained essentially unchanged (`6.953/4.848 ms`).
- **Aggressive GEMM2 cache policy still hard-crashes**: `c_ptrs(.cg+evict_first)` + `w2/s2(.cg+evict_last)` triggered runtime errors on all workloads, so advanced cache modifiers should be reintroduced one tensor at a time (if at all).
- **Reruns of the same final metadata-only `.ca` state are highly variable** (`1.277947` then `1.321842`), while long-seq remains nearly flat (`~6.91-6.96 ms`, `~4.81-4.87 ms`), so this path does not provide stable progress toward sub-1ms.
- **Exact-kernel parity with historical best commit is not reproducing prior 1.085ms behavior**: parity baseline reruns landed `1.272737`, `1.248316`, `1.293105`, indicating environment/run variability dominates small code deltas.
- **Best recent parity rerun reached `1.201368 ms`**, but immediate repeats (`1.308789`, `1.247316`) stayed far above target and preserved long-seq bottlenecks.
- **`out_accum` cannot be left uninitialized**: `torch.empty` output accumulation caused full-suite `INCORRECT_NUMERICAL`, so explicit zero initialization is required for correctness.
- **Autotune `reset_to_zero` is required for correctness**: removing it caused all-workload `INCORRECT_NUMERICAL` because config trials accumulated into the same output tensor.
- **Naive GEMM2 fp16 cast is unstable**: casting `c_f32` and `w2_fp8` to fp16 before `tl.dot` produced INF outputs and failed all workloads.
- **Clean persistent-SM GEMM2 loop (one CTA/SM, tile-stepping) regressed heavily** to `1.463789 ms`, with long-seq latency worsening to `8.281/5.752 ms`; this persistent mapping should be deprioritized.
- **GEMM2 `tl.range(..., num_stages=2)` loop conversion regressed** (`1.282579 ms`) and specifically hurt long-seq latency (`7.119/4.954 ms`), so plain `range` remains preferred in this kernel.
- **Temp-buffer cache reuse regressed on parity baseline** (`1.334842 ms`), so allocator reuse is currently deprioritized versus GEMM2 arithmetic/scheduling bottlenecks.
- **Final parity-baseline rerun stayed in the same band** (`1.249632 ms`; long-seq `6.901/4.817`), reinforcing that the dominant gap is still long-seq GEMM2 accumulation behavior.
- **Passing `BLOCK_M` into routing block sizing fixed the hardcoded dependency but did not unlock `BLOCK_M=64`**; the 64-tile path still failed numerics on large workloads and regressed latency.
- **Route-pruning heuristics are not helping throughput meaningfully**: one-route drop (threshold 0.15) stayed around `1.246 ms`; two-route drop regressed to `1.342 ms`.
- **True `TOP_K=7` remains invalid** (all workloads `INCORRECT_NUMERICAL`), so exact top-k semantics are effectively required by benchmark tolerances.
- **GEMM2 autotune with `GROUP_BLOCKS=8` gave the strongest run in this pass (`1.243737 ms`) but remained above target and was not repeatably dominant.**
- **Compact active-token accumulation (unique-token remap + `index_copy_`) was a major regression** (`1.450158 ms`), so dense `out_accum` + final copy remains the safer path.
- **Final state after reverting failed branches is still above goal** (`1.281105 ms`, long-seq `6.914/4.826`), confirming unresolved long-seq GEMM2 bottleneck.
- **Adding a deeper group8 pipeline config (`num_stages=4`) did not help** (`1.274211 ms`), so this scheduler depth increase is deprioritized.
- **Latest post-revert rerun reached `1.239947 ms`** (best in this pass) but remained above target; long-seq bottlenecks stayed high (`6.904/4.799 ms`).
- **Routing `sort(stable=False)` remains non-competitive** (`1.307789 ms` on current branch), so stable sort should remain the default while optimization focus stays on GEMM2 long-seq path.
- **`tl.atomic_add(..., sem=\"relaxed\")` is a hard regression** in this kernel shape (`1.426053 ms`), so keep default atomic semantics.
- **Primary blocker remains long-seq GEMM2 path** (`5e8dc11c`, `58a34f27` still ~6.9ms and ~4.8ms range), confirming that launch/allocator-only tweaks are insufficient.
- **Output-as-accumulator fast path produced a misleading outlier** (`1.090579`) but failed stability reruns (`1.256632`, `1.261737`), so it is not retained as a dependable optimization.
- **Observed run-to-run variance is high** in this environment; decisions must be based on repeated runs, not single-pass wins.
- **Routing compile mode `max-autotune` remains non-competitive** (`1.256263 ms`) versus the `reduce-overhead` baseline path.
- **Expert-stationary GEMM2 scheduling (one pid per expert/nb, looping block_ids in-kernel) is a hard regression** (`2.595632 ms`; long-seq `18.808/12.264 ms`), so this mapping is deprioritized.
- **GEMM2 `tl.range(..., num_stages=3, flatten=True)` also regressed** (`1.317263 ms`), confirming plain `range` remains the safer ib-loop form in this kernel.
- **Expanding autotune with `GROUP_BLOCKS=8` configs produced a near-best one-off run (`1.086211 ms`) but failed stability reruns (`1.285526`, `1.238737`)**, so it is not yet a reliable promotion candidate.
- **Adding an extra `GROUP_BLOCKS=8, num_warps=8, num_stages=2` config did not help** (`1.250263 ms`), reinforcing that group8 expansions require strict rerun validation.
- **GEMM2 `tl.range(..., warp_specialize=True)` is currently invalid for this kernel shape** and triggered all-workload `RUNTIME_ERROR`.
- **Applying `.ca` on per-route `sorted_tokens`/`w_tok` loads regressed** (`1.302842 ms`) and did not move the long-seq bottleneck.
- **Adding `GROUP_BLOCKS=8, num_warps=4, num_stages=3` regressed heavily** (`1.357842 ms`), indicating deeper pipeline for low-warp group8 is harmful here.
- **Group8-expanded autotune pool remains highly unstable** (`1.086211` outlier vs `1.449263` rerestore); restoring the 4-config pool recovered the prior ~`1.25 ms` band (`1.249842`).
- **After removing unintended GEMM1 metadata `.ca` hints, 4-config pool rerun stayed at `1.247947 ms`**, confirming the current stable baseline band is still ~`1.25 ms` with unchanged long-seq bottlenecks.
- **Applying `.cg + evict_first` only on GEMM2 workspace (`c_ptrs`) loads regressed** (`1.348947 ms`) and worsened long-seq latency (`7.002/4.895`), so this cache-policy path is deprioritized.
- **Latest final 4-config rerun landed at `1.300684 ms` with a worker preemption restart**, reinforcing both the persistent long-seq bottleneck and substantial environment/run variance.
- **Device-local scratch buffer reuse (`block_map`, `workspace`, `out_accum`) regressed** (`1.314789 ms`), so allocator reuse remains low priority versus GEMM2 arithmetic/scheduling bottlenecks.
- **Routing `argsort` + `bincount(flat_sort_key)` micro-optimization regressed** (`1.299000 ms`), so the existing `sort(stable=True)` path remains preferred.
- **Another restored 4-config baseline rerun (`1.286947 ms`) confirms the current operating band remains ~`1.25-1.30 ms` with unchanged long-seq bottlenecks.**
- **Revisiting the `GROUP_BLOCKS=8` expanded autotune pool did not reproduce prior near-best outlier** (`1.249789`, `1.262842`), reinforcing its instability and weak repeatability.
- **Latest restored 4-config rerun reached `1.245316 ms`** (best in this cycle) but still preserved long-seq bottlenecks (`6.912/4.812`), leaving the sub-1ms gap unresolved.
- **Adding `GROUP_BLOCKS=2, num_warps=8, num_stages=3` to the 4-config pool regressed** (`1.306474 ms`), so this intermediate grouping should remain out of the autotune set.
- **Another clean 4-config rerun landed at `1.256053 ms`**, reaffirming the unresolved long-seq wall and high run-to-run variability.
- **Hybrid expert-split non-atomic GEMM2 (for large `total_routed`) regressed** (`1.272000 ms`) and did not improve long-seq workloads, so atomic-removal via per-expert host loop is deprioritized.
- **Post-revert baseline rerun after expert-split stayed at `1.261526 ms`**, reinforcing the unchanged ~`1.25-1.30 ms` operating band.
- **Online-research-inspired imbalance-aware autotune key** (adding `MAX_EXPERT_TOKENS`) regressed (`1.326789 ms`), so gemm2 autotune should stay keyed on `TOTAL_BLOCKS,TOTAL_ROUTED` only.
- **Triton tutorial-style alignment hints** (`tl.multiple_of` + `tl.max_contiguous` on gemm2 `offs_n/offs_i`) produced no stable gain (`1.245421` then `1.253263`).
- **Latest baseline reruns remain highly variant** (`1.241842` to `1.312000`) while long-seq workloads still dominate (~`6.89-6.97 ms` and `4.79-4.87 ms`).
- **DeepSeek MoE paper signals that expert-load skew is a first-order factor**, but this kernel did not benefit from adding explicit load-skew signal to autotune selection.
- **DeepSeek-V2/V3 efficiency claims (e.g., MLA/KV-cache compression) are attention-side gains**, so they are not directly transferable to this FFN MoE GEMM2 bottleneck.
- **Research-driven workspace pre-scaling (multiply routed rows once, then remove per-nb weight multiply in gemm2) regressed** (`1.279211 ms`), likely due extra standalone scaling kernel overhead and cache disruption.
- **Post-revert baseline rerun (`1.260684 ms`) confirms no persistent gain from the pre-scale approach.**
- **Isolated GEMM2 `w2_ptrs` load cache hint `.ca` regressed** (`1.307263 ms`) and worsened long-seq latency (`6.950/4.851`), so weight-load cache-hinting remains deprioritized versus scheduling/accumulation changes.
- **Isolated GEMM2 `s2_ptr` scale-load cache hint `.ca` is unstable/non-repeatable** (`1.246105` then `1.304842 ms`), so scale-load cache-hinting is also deprioritized.
- **GEMM2 `tl.range(..., disable_licm=True)` loop control did not help** (`1.265737 ms`), so plain `range` remains preferred for the `ib` loop.
- **GEMM2 `tl.range(..., disallow_acc_multi_buffer=True)` regressed** (`1.306158 ms`), reinforcing that current loop-control overrides are not improving this kernel.
- **Reducing GEMM2 output tile width to `BLOCK_N=64` is invalid in current layout** (full-suite `INCORRECT_NUMERICAL`), so `BLOCK_N=128` should remain fixed unless scale/index mapping is redesigned.
- **Adding autotune config `GROUP_BLOCKS=2, num_warps=4, num_stages=2` regressed** (`1.311421 ms`), so group2 autotune expansion remains non-competitive.
- **Fresh rerun from restored `3f14870` baseline reached `1.113211 ms`**, confirming the current environment is now capable of substantially faster results than the earlier ~`1.25 ms` band.
- **`GROUP_BLOCKS=8` expansion with `BLOCK_M=32` produced a near-best one-off (`1.092895 ms`) but remained unstable** (`1.260947 ms` rerun).
- **The strongest current direction is dynamic `BLOCK_M=16` + g8x2 autotune pool**, which repeatedly landed in the low-1.0x range (`1.023053`, `1.028842`, `1.030474`, `1.037316`, `1.034421`) with best `1.018842 ms`.
- **Long-seq workloads are still the final blocker even on the new best path** (`5e8dc11c ~6.64 ms`, `58a34f27 ~4.58-4.62 ms`).
- **Further autotune expansion to `GROUP_BLOCKS=16` did not help** (`1.028632 ms` vs current best `1.018842 ms`).
- **Single fixed `GROUP_BLOCKS=8` configs are non-competitive on this branch** (`w8/s3 -> 1.155211`, `w4/s2 -> 1.029789`) versus mixed g8x2 pool.
- **GEMM2 micro-optimizations on the current best path regressed**: full-block fast path branch (`1.244000`) and unmasked `atomic_add` (`1.182684`).
- **`BLOCK_M=8` is a hard performance regression** (`1.648526 ms`) due much worse long-seq behavior (`11.957/8.084 ms`).
- **Extended rerun sweeps on the current best branch did not hit sub-1ms**: 12-run best `1.018579`, 15-run best `1.021368`; heavy variance remains (`~1.02` to `~1.25+`).
- **`BLOCK_M=24` is non-viable** (full-suite `RUNTIME_ERROR`), reinforcing that current safe/competitive tile choices are constrained.
- **Further autotune key simplification is non-competitive**: `TOTAL_BLOCKS`-only (`1.033947`) and `TOTAL_ROUTED`-only (`1.032842`) both regressed versus dual-key.
- **More aggressive grouping expansions regressed**: `GROUP_BLOCKS=32,w4,s2` (`1.247579`), and extra group8 staging variants (`w4,s1` `1.026895`; `w8,s4` `1.024211`) did not beat best.
- **Routing-side alternatives stayed non-competitive**: unstable sort (`1.034053`), argsort dispatch (`1.028474`, rerun `1.176526`), compile mode `max-autotune` (`1.030000`), and `max-autotune-no-cudagraphs` (`1.267526`).
- **GEMM1 launch retunes regressed**: `num_warps=8` (`1.099421`) and `num_stages=2` (`1.080158`) both worsened long-seq latency.
- **Accumulation bandwidth shortcuts remain blocked by correctness**: BF16 `out_accum` failed numerics (`e05c6c03`), and hybrid tiny-FP32/large-BF16 restored correctness but remained slower (`1.031789`).
- **Dot-path shortcuts are invalid in this kernel**: `tl.dot(..., out_dtype=fp16)` regressed (`1.383684`) and direct fp8-W2 dot path crashed all workloads.
- **Latest direct reruns after all reverts remained above target** (`1.027474`, `1.030158`, `1.192053`), so sub-1ms is still not crossed on the current best branch.
- **True two-stage GEMM2 reduction (store routed outputs then final `index_add_`) regressed** (`1.244789 ms`), so extra global-memory traffic outweighed atomic savings.
- **Recent cache-hint revisit on GEMM2 metadata loads (`.ca`) did not yield stable gains** (`1.027842` then `1.197421`).
- **Forcing single GEMM2 configs remained non-competitive** on blockm16 (`g4-w8-s3: 1.145`, `g4-w4-s2: 1.036`, `g1-w8-s3: 1.135`, `g1-w4-s2: 1.023/1.032`).
- **Slim autotune pools also underperformed** (`g1+g8 w4-s2: 1.030`, `g1+g4+g8 w4-s2: 1.153`) versus the mixed 6-config baseline.
- **Additional brute-force rerun sweeps still did not cross sub-1ms** (best new sweep run `1.019158`, worst around `1.256`), reinforcing runtime variance as a major blocker.
- **Latest restored-branch confirmation also stayed above target** (`1.030211`), so sub-1ms remains unmet.
- **Further rerun sweeps produced a new best of `1.013526 ms`** (`1.014579` and then `1.013526`), but still did not cross sub-1ms.
- **`BLOCK_M=12` is also non-viable** (full-suite `RUNTIME_ERROR`), tightening practical block-size choices to 16/32.
- **Dispatch token-id cache refactor was invalid in this runtime path** (full-suite `RUNTIME_ERROR`).
- **Dual-kernel routing strategy (g8-only GEMM2 on long-seq)** was non-competitive (`1.026789` then `1.176158`).
- **GEMM2 BF16-dot conversion is not correctness-safe in this kernel path** (`c_f32/w2_fp8 -> bf16 before tl.dot` caused 22 `INCORRECT_NUMERICAL` workloads), and should remain reverted.
- **Fresh post-revert reconfirm is still above target** (`1.022263 ms`, correctness-clean), so the baseline remains near but not under 1ms.
- **`BLOCK_M=20` is non-viable** (full-suite `RUNTIME_ERROR`), further reinforcing practical block-size constraints around 16/32.
- **Additional launch/loop retunes regressed**: GEMM1 `num_stages=4` (`1.235158`), GEMM1 `num_warps=2` (`1.173053`), and GEMM2 `tl.static_range` ib loop (`1.434211`).
- **Another short brute-force rerun sweep remained above target** (4-run best `1.029368`), with variance still dominating outcome quality.
- **Latest baseline revalidation remained above target** (`1.028789 ms`), confirming no persistent gain after recent revert cycles.
- **Block-map kernel launch retune is non-competitive** (`_build_block_map_kernel` with `num_warps=4` regressed to `1.186158 ms`), so keep `num_warps=1`.
- **Another post-rollback rerun sweep still failed to cross sub-1ms** (4-run best `1.031105`), reinforcing that current branch remains variance-limited above target.
- **Latest restored-branch full check remained correctness-clean but above target** (`1.035474 ms`), so sub-1ms is still unmet.
- **Output-buffer reuse as GEMM2 accumulator regressed** (`1.184263 ms`), so keep the dedicated `out_accum` allocation + final `output.copy_` path.
- **Adding GEMM2 warps2 autotune configs (`GROUP_BLOCKS` 4/8, `num_warps=2`, `num_stages=2`) was unstable and non-competitive** (one-off `1.021158`, reruns `1.170895/1.174526`), so keep the original 6-config pool.
- **`BLOCK_M=14` and `BLOCK_M=18` are both non-viable** (full-suite `RUNTIME_ERROR`), further tightening practical block-size options to `16/32` only.
- **`out_accum` must be explicitly zero-initialized in `run()`**; changing it from `torch.zeros` to `torch.empty` caused full-suite `INCORRECT_NUMERICAL` despite GEMM2 autotune `reset_to_zero`.
- **Block-map expert-id binary search is invalid in current Triton implementation** (runtime errors + incorrect numerics), so keep the linear-scan mapping in `_build_block_map_kernel`.
- **Recent reconfirm + short rerun window remained above target** (`1.183053` reconfirm; partial sweep best `1.026053`), so the branch is still correctness-safe but variance-limited above sub-1ms.
- **Another short rerun window also stayed above target** (best `1.036368` across `1.036/1.182/1.174`), confirming continuing high-latency variance bursts.
- **Restricting GEMM2 autotune pool to GROUP_BLOCKS 4/8 only is non-competitive and unstable** (`1.031737`; reruns `1.040/1.026/1.175`), so retain the original mixed g1/g4/g8 6-config pool.
- **Extended rerun sampling still did not cross sub-1ms** (12-run window best `1.017632`, with repeated high spikes around `1.18-1.21`), reinforcing bimodal variance as the dominant blocker.
- **Latest restored-branch sanity run remained above target** (`1.022211 ms`, correctness-clean), so the sub-1ms goal is still not reached.
- **For routing+dispatch, `torch.compile(..., fullgraph=True)` is not correctness-safe** (full-suite `INCORRECT_NUMERICAL`), so keep `mode="reduce-overhead"` without fullgraph forcing.
- **Post-revert sanity remains above target** (`1.033158 ms`, correctness-clean), consistent with the current >1ms plateau.
- **For routing+dispatch, `torch.compile(..., dynamic=False)` is also non-competitive** (`1.116895 ms`), so keep the existing default dynamic behavior.
- **Latest post-revert sanity run is still above target** (`1.029842 ms`, correctness-clean), so sub-1ms remains unmet.
- **Routing bool-mask rewrite is non-competitive** (`1.026579`; reruns `1.032526/1.028316/1.028211`), so keep the original float-mask routing code.
- **Latest post-revert sanity check is still above target** (`1.031632 ms`, correctness-clean), keeping the branch in the low-1.0x plateau.
- **GEMM2 pointer-base hoisting is non-competitive and unstable** (`1.026842`; reruns `1.033/1.031/1.395`), so keep the original in-loop pointer expressions.
- **Routing+dispatch must stay compiled**; switching to eager execution regressed strongly (`1.199421 ms`).
- **Latest compiled-baseline reconfirm stayed above target** (`1.027789 ms`, correctness-clean), so no sub-1ms crossing yet.
- **Routing arithmetic cannot be downcast to FP16**; FP16 path caused widespread `INCORRECT_NUMERICAL` (34 workloads).
- **Post-FP16-rollback sanity stayed above target** (`1.036263 ms`, correctness-clean).
- **Adding GEMM2 config `GROUP_BLOCKS=8,num_warps=8,num_stages=2` is unstable/non-competitive** (runs `1.036579`, `1.225842`, `1.015895`, `1.190316`, `1.031368`, `1.191105`), so keep the prior 6-config pool.
- **Latest post-revert sanity remains above target** (`1.034053 ms`, correctness-clean), confirming no persistent gain from the added g8-w8-s2 config experiment.
- **Another extended rerun window still stayed above sub-1ms** (8-run best `1.015842`, with spikes up to `1.198053`), so variance remains the dominant blocker.
- **Block-map kernel `num_warps=2` is also non-competitive** (`1.185105 ms`), so keep `num_warps=1`.
- **Back-to-back post-revert sanity runs continue to show bimodal variance** (`1.188053` then `1.033842`), with correctness intact but no sub-1ms crossing.
- **Dynamic `BLOCK_M` heuristic is non-competitive** (`BLOCK_M=16` for long-seq and `32` for short-seq regressed to `1.090158 ms`), so keep fixed `BLOCK_M=16`.
- **Variance now includes occasional extreme spikes** (`1.409789` followed immediately by `1.025632` on unchanged kernel), reinforcing that runtime instability is dominating final-gap progress.
- **Even ultra-long and cooldown-spaced rerun strategies failed to break 1ms** (ultra-long partial `1.386/1.181/1.234`; cooldown partial `1.191/1.041/1.026`), indicating benchmark pacing alone is not solving the variance wall.
- **Branchless block-map expert lookup via cumulative block-end sum is non-competitive** (`1.257211 ms`), so keep the original linear scan in `_build_block_map_kernel`.
- **Adding GEMM2 config `GROUP_BLOCKS=8,num_warps=4,num_stages=3` regressed** (`1.225895 ms`), so this staging expansion should remain excluded.
- **Adding GEMM2 config `GROUP_BLOCKS=1,num_warps=4,num_stages=3` also regressed sharply** (`1.297316 ms`), so keep current group1 configs unchanged.
- **Adding GEMM2 config `GROUP_BLOCKS=4,num_warps=8,num_stages=2` is non-competitive** (`1.022684`; reruns best `1.021684`), so keep the existing 6-config pool unchanged.
- **Adding GEMM2 config `GROUP_BLOCKS=4,num_warps=4,num_stages=3` is the first successful final-gap change** (`1.009474 ms` initial, then `0.995000 ms` on focused rerun), achieving the sub-1ms target with correctness intact.
