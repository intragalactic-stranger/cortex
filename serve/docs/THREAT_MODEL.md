# Cortex Serve on Kubernetes: threat model (M5)

Scope: the k3d deployment in `serve/deploy/` (LiteLLM gateway + Postgres, llama.cpp model pods,
Prometheus/Grafana, KEDA) on one VM. Method: assets → trust boundaries → threats (STRIDE-style) →
controls in place → residual risk. Each control below is in the chart or scripts; nothing is assumed.

## Assets
| Asset | Why it matters |
|---|---|
| Gateway master key | Full admin: create/delete keys, read spend, change routing |
| Virtual keys | Model access with per-key rpm / parallel-request limits and expiry |
| Backend API key | Direct access to llama.cpp servers, bypassing gateway limits and logging |
| Prompts and completions | User data; may hold personal or confidential text |
| Model weights (GGUF cache) | Integrity: a swapped file changes model behaviour silently |
| Gateway DB (Postgres) | Key hashes, budgets, spend logs |
| Availability of the pool | The service itself |

## Trust boundaries
1. **Host → cluster:** only three loopback ports are published (`127.0.0.1:4000` gateway, `:9090`
   Prometheus, `:3300` Grafana) plus the API server on `127.0.0.1:6550`. Nothing listens on the VM's
   public interface.
2. **Client → gateway:** bearer-token auth (virtual keys) on every inference route.
3. **Gateway → model pods:** in-cluster, NetworkPolicy-restricted, backend API key.
4. **Monitoring → workloads:** Prometheus scrapes model pods (with the backend key) and the gateway.

## Threats and controls
| # | Threat | Control (where) | Residual risk |
|---|---|---|---|
| T1 | **Spoofing:** client calls a model pod directly, bypassing keys, limits and logs | NetworkPolicy `models-ingress-gateway-only`: ingress to model pods only from gateway pods and Prometheus pods, TCP 8080. llama-server also requires `LLAMA_API_KEY` on every route except `/health` (defense in depth). Verified: a test pod gets a timeout (M5 report §4.4) | Prometheus can reach `/v1/*` on the same port (L4 policy cannot filter by path), but it would need the backend key, which is mounted only through the PodMonitor secret ref |
| T2 | **Key leakage** through the repo | No keys in code or values. `up.sh` generates master/salt/backend/DB secrets with `openssl rand` into Secrets at install time. Load-test keys are created by `m5_key.sh`, expire after 24 h, and are held only in env. gitleaks runs in pre-commit and CI | Secrets are base64 in etcd (k3s default: no encryption at rest). Acceptable on a single-user VM; enable `--secrets-encryption` for shared use |
| T3 | **Elevation:** container escape / privilege abuse from a model server parsing untrusted input | Non-root UIDs (10001 model, 65534 gateway, 70 Postgres), `allowPrivilegeEscalation: false`, all capabilities dropped, `seccompProfile: RuntimeDefault`, read-only root filesystem on model pods, `automountServiceAccountToken: false` (no Kubernetes API token in any workload pod) | Gateway rootfs is writable (LiteLLM writes Prisma/runtime files). The image is large, so its attack surface is too |
| T4 | **Exfiltration** from a compromised model pod | Model pods have **no egress** (empty egress policy, not even DNS). The weights volume is mounted read-only from a read-only hostPath | A compromised gateway could still exfiltrate. Its egress is limited to model pods, Postgres and DNS (internet only if the disabled Modal slot is turned on, then TCP 443 to public IPs only) |
| T5 | **Tampering** with weights | Read-only mount at three layers (k3d bind `:ro`, PV `ReadOnlyMany`, `readOnly: true` volumeMount). The file path and HF revision are recorded as pod annotations and in every run manifest. SHA-256 of each GGUF is cached from M1 (`*.sha256.json`) | No admission-time hash check. A changed file on the host would be served after the next pod start. Next step: an init container that verifies the SHA-256 |
| T6 | **Information disclosure** via metrics | Model `/metrics` needs the backend key. Gateway `/metrics` is unauthenticated (LiteLLM requires a proxy-admin key for it, and Prometheus should not hold admin). It exposes counts, latencies, key aliases and hashed key ids, not prompts | On a shared cluster, serve gateway metrics on an internal-only port or behind a sidecar that adds auth |
| T7 | **Denial of service:** one client floods the pool | Per-key `rpm_limit` and `max_parallel_requests`. Per-backend `rpm` in the router. Request timeout 120 s and stream first-chunk timeout 60 s. The queue is bounded by llama.cpp slots, and KEDA adds replicas up to 3 | Max replicas is capped by the node. Under sustained overload TTFT grows (alert `CortexSlotUsageHigh`) instead of requests being rejected early. Admission control (shed when queue > N) is not implemented |
| T8 | **Repudiation:** no record of who called what | LiteLLM spend logs per key in Postgres; JSON logs | Logs live only in the cluster (no shipping). Prompt logging is off by default (good for privacy, less useful for forensics) |
| T9 | **Supply chain:** mutable image tags | Every image is pinned by digest (`values.yaml`, monitoring values, k3d node image); chart versions are pinned in `versions.env`. Tools are installed from official releases after checksum verification (`docs/VERSIONS.md`) | Digests are trusted on first use (no signature verification, e.g. cosign) |
| T10 | **Prompt injection / unsafe output** | Out of scope for the serving layer (no tools or agents behind the gateway in M5) | Becomes relevant in Phase 2/3 (agents); belongs in the application layer |
| T11 | **Fallback abuse:** fallback hides a broken primary | Fallbacks are counted (`litellm_deployment_successful_fallbacks`), and `CortexModelBackendDown` fires after 1 min with no ready replica | None beyond alert routing (Alertmanager is disabled on the laptop stack) |

## Out of scope / not done
TLS inside the cluster (no mesh; all traffic stays on one node), Alertmanager routing, image signing,
etcd encryption, multi-tenant isolation (one namespace, one team).
