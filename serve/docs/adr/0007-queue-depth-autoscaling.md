# 0007 · Autoscale LLM pools on queue depth, not CPU

- **Status:** Accepted · **Date:** 2026-11-24 · **Related:** M5, G-13, ADR 0006, plan M5 task 4

## Context
The M5 primary pool is llama.cpp `llama-server` (Qwen3-0.6B Q4_K_M) on a single-node k3d cluster.
Each replica runs 1 decode thread with 2 slots (`-np 2`). Requests that find no free slot wait inside
the server (`llamacpp:requests_deferred`); requests in a slot are `llamacpp:requests_processing`.
We need a scaling signal that tracks the user-visible problem (waiting for a first token) and that
reacts before the SLO is broken.

## Why CPU utilisation (the default HPA metric) is the wrong signal
1. **It saturates at low load and stays flat.** A llama.cpp replica with one busy slot already pins its
   decode threads near 100% of their CPU quota, and stays there whether 1 or 20 requests are waiting.
   CPU says "full" but not "how much more capacity is needed". HPA's ratio
   `desired = ceil(current * usage / target)` then saturates as well.
2. **It does not see the queue.** Waiting requests use no CPU. The queue (and TTFT) can grow without any
   change in the metric. Queue depth is the leading indicator of TTFT; CPU is not.
3. **It is wrong in the other direction on GPUs.** For vLLM/SGLang the GPU, not the CPU, does the work;
   host CPU is low even when the server is saturated. The analogue there is
   `vllm:num_requests_waiting` (documented for the GPU path; not deployed, G-13).
4. **Memory-bound decode.** Decode speed is set by memory bandwidth (RESULTS §M1: 9–17 GB/s effective on this
   box). Utilisation % does not show that limit.

## Decision
- KEDA `ScaledObject` (Prometheus scaler, `metricType: AverageValue`) on the primary pool with
  **demand = Σ over pods of (requests_processing + requests_deferred)**, smoothed as the mean of the
  summed series over 15 s (`avg_over_time((Σ processing + Σ deferred)[15s:5s])`), and
  **target = slots per replica** (`-np`). The HPA then computes `desired = ceil(demand / slots)`: enough
  replicas that no request has to queue.
- Running requests are included, not only the queue: with only the deferred count, a fully used but
  non-queuing pool reads 0 and gets scaled down, which creates a queue again (oscillation).
- `min 1 / max 3` replicas: 3 × 1 thread leaves 1 vCPU of the 4 for the gateway, the canary, the
  fallback model and the monitoring stack. Scale-up: +1 pod per 15 s, no stabilisation window.
  Scale-down: 60 s stabilisation, −1 pod per 30 s, and a drain-aware `preStop` hook so that removed pods finish
  their in-flight requests.
- KEDA `fallback` holds the current replica count if Prometheus is unreachable
  (`behavior: currentReplicasIfHigher`). It never scales on missing data.

## Amendment (same day, after the first ramp)
The first version summed two separate `max_over_time` terms. A request that moves from deferred to
processing inside the 15 s window was counted twice: 4 users read as 5 and the pool went to 3 replicas
instead of 2 (`results/m5-autoscale/20261124T131550Z`). The second version took the peak of the summed
series (subquery). It still read 4 users as 5 now and then (`results/m5-rollout/20261124T133518Z`): pods
are scraped at slightly different instants, so a request that finished on one pod and the next one that
started on another can both be counted. The pool again settled at 3 replicas for 4 users. The final version
takes the 15 s **mean** of the summed series. This removes the one-scrape spikes and costs a few
seconds of reaction time.

## Consequences
+ The metric reflects what users feel. The timeline figure (`results/m5-autoscale/`) shows replicas following
  queued demand and TTFT recovering after each scale-up.
+ It is engine-agnostic: the same pattern works with vLLM's `num_requests_waiting` + `num_requests_running`.
− The reaction time includes the scrape interval (5 s), the 15 s window, the HPA sync period (15 s, k3s default),
  and pod start (mmap'd weights from the node page cache plus readiness). Scaling is slower than queue
  growth for short bursts; for bursty traffic a higher `minReplicas` is the fix, not a faster metric.
- One node only: a new replica adds CPU threads until the 4 vCPUs are used, so the pool
  stops at 3. On a real cluster the same object would add nodes through the cluster autoscaler.
− Load balancing across replicas uses the ClusterIP Service (connection-level). With long-lived
  keep-alive connections from the gateway, new replicas fill up only as new connections are opened.
  The per-pod series in the run directory show how even the load was.
