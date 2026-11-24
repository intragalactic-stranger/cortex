# Versions

Resolved and verified at build time. Exact pins for Python packages live in `uv.lock`; engine image digests are added from M1/M2.

| Component | Version | Verified | Notes |
|---|---|---|---|
| uv | 0.13.7 | 2026-11-05 | mandatory package/env manager (D-010) |
| Python | 3.12.8 | 2026-11-05 | installed by uv; `.python-version` = 3.12 |
| pydantic | 2.14.2 | 2026-11-05 | |
| typer | 0.27.3 | 2026-11-05 | memcalc CLI |
| rich | 15.0.0 | 2026-11-05 | |
| httpx | 0.28.1 | 2026-11-05 | Hub access, range requests (later: servebench client) |
| pyyaml | 6.0.3 | 2026-11-05 | |
| ruff | 0.17.3 | 2026-11-05 | |
| mypy | 2.4.1 | 2026-11-05 | `--strict` |
| pytest | 9.2.0 | 2026-11-05 | |
| pre-commit | 4.7.0 | 2026-11-05 | gitleaks hook v8.24.2 |

## Model revisions (fixtures)

| Model | Revision (HF commit) | License |
|---|---|---|
| Qwen/Qwen3-0.6B | c1899de289a04d12100db370d81485cdf75e47ca | Apache-2.0 |
| Qwen/Qwen3-1.7B | 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e | Apache-2.0 |
| Qwen/Qwen3-4B-Instruct-2507 | cdbee75f17c01a7cc42f958dc650907174af0554 | Apache-2.0 |
| Qwen/Qwen3-30B-A3B | ad44e777bcd18fa416d9da3bd8f70d33ebb85d39 | Apache-2.0 |
| Qwen/Qwen3.5-4B | 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a | Apache-2.0 |
| google/gemma-4-E2B-it | 4adf15748de9e1a0097b6274cee9c15836c41c37 | Apache-2.0 |
| google/gemma-4-E4B-it | 4ecc2ba63bcac23ba4a767fc87025b1450f84c9c | Apache-2.0 |

## M5: local Kubernetes stack (verified 2026-11-24)

Tools are user binaries in `~/.local/bin`, downloaded from the official release URLs and checked against
the published SHA-256 before install.

| Tool | Version | SHA-256 of installed binary | Source / checksum file |
|---|---|---|---|
| k3d | v5.9.0 | `06d8f25bc3a971c4eb29e0ff08429b180402db0f4dec838c9eac427e296800a0` | GitHub release `checksums.txt` |
| kubectl | v1.35.10 | `c49a28f1f4efc4d475ed05f47f2ce8e563bf7bef69ed77fe1b924ea6d83b6327` | `dl.k8s.io/.../kubectl.sha256` |
| helm | v4.3.1 (tarball sha verified) | `b66cb19f2b31f9d427632ea6c4b4a1ff89b74ecaeaee323cd166da41a7af53bd` | `get.helm.sh/...tar.gz.sha256sum` |

Charts (pinned in `serve/deploy/k3d/versions.env`): KEDA **2.21.1**, kube-prometheus-stack **93.0.0**
(newest release older than two weeks on 2026-11-24; operator v0.95.1).

Images (all pinned by digest in the chart / values files):

| Image | Tag | Digest (index) |
|---|---|---|
| rancher/k3s (node) | v1.35.10-k3s1 | `sha256:8262ca99a558dca0c0437443421986480b2359f9118f2371359f88a41514522f` |
| ghcr.io/ggml-org/llama.cpp | server (build 12110, commit cd959bae9; same as M1) | `sha256:1145e8de3111dc9358c6d8daffb755c4cf8a862d93391c26ba521b9a810bbe3e` |
| ghcr.io/berriai/litellm-non_root | v1.108.16 | `sha256:9a81470454422473f10882f8de4c2c780bcd4a5719159c133110d437e1b8f7a7` |
| postgres | 17.7-alpine | `sha256:bb377b7239d2774ac8cc76f481596ce96c5a6b5e9d141f6d0a0ee371a6e7c0f2` |
| ghcr.io/kedacore/keda | 2.21.1 | `sha256:d09d92c0a310e3057a3c54df387b99abf2bf31352218dbe1adcb00e807886488` |
| ghcr.io/kedacore/keda-metrics-apiserver | 2.21.1 | `sha256:747e376b3e80dc4f72bf4f7043f47cc4cdb56f34b1aed9531789f474996e1c84` |
| ghcr.io/kedacore/keda-admission-webhooks | 2.21.1 | `sha256:5f4b0b0f7831b9f53738be85e99449a1747a5effa970305fe602455ab0f4fcd3` |
| quay.io/prometheus/prometheus | v3.16.0-distroless | `sha256:9764044e09eb2c9e1abdb12d0f0188d4efb7384b0262d3c080f0fc59045c89ba` |
| quay.io/prometheus-operator/prometheus-operator | v0.95.1 | `sha256:9c2490af910d58ec2abd77642d340e30bd5e2de5c80fd19ab4b49e28dc7bbb83` |
| quay.io/prometheus-operator/prometheus-config-reloader | v0.95.1 | `sha256:39bdbade0c64f7c38bb6d18d6f602b14ad489bc189b709dc9266b49e6647aac1` |
| registry.k8s.io/kube-state-metrics/kube-state-metrics | v2.20.0 | `sha256:42cfe3723a5f058171c627537fb57a3ea0f26e4380fa18555a95cb1a1b4cfc5b` |
| docker.io/grafana/grafana | 13.3.0-distroless | `sha256:540b8afb775bf34aefce417e4329039daa5d44b6f2fbda6881397b1b247ca7df` |
| quay.io/kiwigrid/k8s-sidecar | 2.14.3 | `sha256:52d311dcedd587421a236c4d6c621f05614c40a51386e955e0699dca97ec0880` |

The k3s add-ons (coredns, local-path-provisioner, pause) are the versions bundled with the pinned k3s
node image. Every run manifest (`results/m5-*/*/manifest.json`) records the image IDs that actually ran.
Disk used by the running cluster: ≈ 8.2 GB (containerd content 1.9 GB + unpacked snapshots 5.8 GB +
kubelet 0.4 GB), measured with `du` inside the node.
