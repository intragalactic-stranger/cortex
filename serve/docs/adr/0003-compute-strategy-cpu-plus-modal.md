# 0003 · Compute: local CPU + Modal GPUs (≈ $200), no managed GPU Kubernetes

- **Status:** Accepted · **Date:** 2026-11-04 · **Related:** D-005, G-13

## Context
The spec assumes rented GPUs (RunPod etc.) and a time-boxed GKE/EKS GPU cluster. Available: a CPU-only dev VM (AMD EPYC 7571, 4 threads, AVX2 only, 15 GiB RAM, ~40 GB disk), ~$200 of Modal credits, and Blaxel.

## Options considered
1. CPU-only project: cheap, but loses the core inference experiments (vLLM/SGLang, FP8, MoE).
2. **Modal for all GPU work**; Kubernetes demonstrated on local k3d with CPU llama.cpp pods plus a Modal-hosted vLLM endpoint behind the gateway.
3. Add a paid GKE/EKS GPU node pool for the autoscaling figure (~$50–150 extra).
4. Blaxel for GPUs: its own comparison pages state it has no GPU instances, so it isn't viable.

## Decision
Option 2. Option 3 stays open (G-13). Blaxel is considered only as a zero-idle-cost CPU host for a tiny live demo (G-09).

## Consequences
+ All high-value inference experiments remain. + Per-second billing suits short benchmark bursts.
− No managed-GPU autoscaling figure; stated honestly in README and interviews as a budget tradeoff.
− Modal adds its own container/network behaviour, so the load client runs in the same container as the engine (G-05).
− Locally: SGLang CPU needs AMX and vLLM CPU targets AVX-512, so the CPU tier is llama.cpp + Ollama.
