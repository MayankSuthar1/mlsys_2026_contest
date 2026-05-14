# Auto-kernel (FlashInfer AI Kernel Generation Contest MLSys 2026)

This repository contains our submission for the FlashInfer AI Kernel Generation Contest at MLSys 2026. We competed in the `fused_moe` track and developed a high-performance Triton kernel for FP8 Mixture-of-Experts routing.

## The Solution

Our implementation, **auto-kernel**, targets the `moe_fp8_block_scale_ds_routing` workload. The final Triton kernel achieves a stable, correctness-clean average latency of 0.995 ms on NVIDIA B200 GPUs. 

Key optimizations include:
* Fused routing and dispatch stages compiled with PyTorch to reduce Python overhead.
* On-device block map building to avoid multithreaded CPU-GPU pipeline bottlenecks.
* Highly tuned grouped scheduling for the secondary GEMM stage to maximize L2 cache locality.

## Agentic Workflow

Instead of manually hand-tuning the kernel, we utilized an autonomous agentic pipeline to drive the optimization process. This architecture was structured around Andrej Karpathy's autoresearch methodology.

<img src="images/agentic_workflow.svg" alt="Agentic Workflow Architecture" width="500" />

The workflow was automated using GitHub Copilot CLI agents powered by GPT-5.3-Codex. The execution loop operated as follows:
1. The agent formulates a hypothesis to improve the kernel.
2. It edits the Triton kernel source code.
3. The solution is packed and sent to Modal for remote benchmarking on B200 hardware.
4. The agent parses the logs to evaluate performance and numerical correctness.
5. The experiment is recorded into a research ledger.
6. The agent independently decides to retain or revert the change, then begins the next iteration.

This continuous feedback loop ran nearly 300 experiments, successfully driving the execution time down from a 1.52 ms baseline to the final sub-1ms result.

## Reproducibility

To run the pipeline and replicate the benchmark:

```bash
conda run --no-capture-output -n fi-bench python scripts/pack_solution.py
conda run --no-capture-output -n fi-bench modal run scripts/run_modal.py
```
