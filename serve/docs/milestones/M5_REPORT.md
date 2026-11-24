# M5 · Production-style serving on local k3d (CPU): Report

> **Status:** `DONE` with documented downscaling · **Date:** 2026-11-24 · **Branch:** `serve/m5-k3d` · **Paid compute:** $0 (all local)
> **Results:** [`../../RESULTS.md`](../../RESULTS.md) §M5 · **ADR:** [`0007`](../adr/0007-queue-depth-autoscaling.md) · **Threat model:** [`../THREAT_MODEL.md`](../THREAT_MODEL.md)

GPU Kubernetes is deferred (G-13). Everything below runs on a single-node k3d cluster on the 4 vCPU / 15 GiB
AWS VM, with CPU llama.cpp pods. The gateway falls back to a second self-hosted model (D-021), not to an
external API.

## 1. Summary of what was done

| Deliverable | Status | Evidence |
|---|---|---|
| k3d cluster scripts, `make k3d-up` / `make k3d-down` | Done | `deploy/k3d/` (loopback-only ports, read-only GGUF mount); cluster built twice from the scripts |
| Helm chart: per-model Deployments from `values.yaml` (primary pool, canary, fallback) | Done | `deploy/helm/cortex-serve/`; `helm lint` + `helm template` clean; 8 render tests (`tests/test_helm_chart.py`) |
| Startup / readiness / liveness probes for slow loads; drain-aware `preStop` | Done | `templates/models.yaml`; ⟨ROLLOUT⟩ |
| Weights from a cache through PV/PVC (hostPath, read-only) | Done | `templates/weights.yaml`; pods start in ~6 s from cache |
| PDB, `maxUnavailable: 0`, non-root, read-only rootfs, no SA token | Done | chart + tests |
| NetworkPolicies: only the gateway (and Prometheus) reach model pods; model pods have no egress | Done, verified in-cluster | ⟨NETPOL⟩ |
| LiteLLM gateway (pinned): `cortex-default`, `cortex-judge`, `cortex-router`; virtual keys with rpm / parallel limits; timeouts; retries; cooldowns | Done | `templates/gateway-config.yaml`, `scripts/m5_key.sh` |
| Fallback chain: llama.cpp pool → second self-hosted model (D-021) | Done, demonstrated | ⟨FALLBACK⟩ |
| Disabled Modal vLLM slot (URL to be supplied; inserts itself into the chain, opens egress 443 only when enabled) | Done (not deployed) | `values.yaml` `gateway.modal`, render test |
| KEDA ScaledObject on queue depth + ADR "why not CPU HPA" | Done | ADR 0007; two ramps |
| Prometheus + Grafana (trimmed kube-prometheus-stack), dashboard JSON, alert rules | Done | `dashboards/cortex-serve.json`, `templates/alerts.yaml`; rules load healthy |
| Canary 90/10 with promotion rule (error rate + latency + exact-answer eval) | Done | ⟨CANARY⟩ |
| Autoscaling timeline figure | Done | ⟨AUTOSCALE⟩ |
| Rolling update under load, zero failed requests | ⟨ROLLOUT_STATUS⟩ | ⟨ROLLOUT⟩ |
| `deploy/compose/` + `make up` | Written, config validated, **not run** (disk) | `docker compose config` passes; see §5 |
| Threat model | Done | `docs/THREAT_MODEL.md` |
| Tooling: `gatewaybench` (profile load, per-request outcome log, timeline, balance, pod lifecycle, canary gate) | Done | `src/cortex_serve/gatewaybench/`, 30 unit tests |

⟨BODY⟩
