"""Render the cortex-serve Helm chart and check the M5 invariants (skipped when helm is not installed)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

CHART = Path(__file__).resolve().parents[1] / "deploy" / "helm" / "cortex-serve"
pytestmark = pytest.mark.skipif(shutil.which("helm") is None, reason="helm not installed")


def render(*sets: str) -> list[dict[str, Any]]:
    args = ["helm", "template", "cortex", str(CHART), "-n", "cortex"]
    for s in sets:
        args += ["--set", s]
    out = subprocess.run(args, capture_output=True, text=True, timeout=60, check=True).stdout
    return [d for d in yaml.safe_load_all(out) if d]


def kind(docs: list[dict[str, Any]], k: str) -> list[dict[str, Any]]:
    return [d for d in docs if d["kind"] == k]


def gateway_config(docs: list[dict[str, Any]]) -> dict[str, Any]:
    [cm] = [d for d in kind(docs, "ConfigMap") if d["metadata"]["name"] == "cortex-gateway-config"]
    cfg: dict[str, Any] = yaml.safe_load(cm["data"]["config.yaml"])
    return cfg


def test_every_image_pinned_by_digest() -> None:
    for d in kind(render(), "Deployment") + kind(render(), "StatefulSet"):
        for c in d["spec"]["template"]["spec"]["containers"]:
            assert "@sha256:" in c["image"], c["image"]


def test_model_deployments_rollout_probes_and_security() -> None:
    docs = render()
    models = [d for d in kind(docs, "Deployment") if d["metadata"]["name"].startswith("cortex-model-")]
    assert {d["metadata"]["name"] for d in models} == {
        "cortex-model-primary",
        "cortex-model-canary",
        "cortex-model-fallback",
    }
    for d in models:
        spec = d["spec"]
        assert spec["strategy"]["rollingUpdate"]["maxUnavailable"] == 0
        pod = spec["template"]["spec"]
        assert pod["securityContext"]["runAsNonRoot"] is True
        assert pod["automountServiceAccountToken"] is False
        [c] = pod["containers"]
        assert c["securityContext"]["readOnlyRootFilesystem"] is True
        assert {"startupProbe", "readinessProbe", "livenessProbe"} <= c.keys()
        # startup budget (period x failures) must cover a slow load; liveness must not be the tight one
        start = c["startupProbe"]["periodSeconds"] * c["startupProbe"]["failureThreshold"]
        live = c["livenessProbe"]["periodSeconds"] * c["livenessProbe"]["failureThreshold"]
        ready = c["readinessProbe"]["periodSeconds"] * c["readinessProbe"]["failureThreshold"]
        assert start >= 120 and live > ready
        assert c["lifecycle"]["preStop"]["exec"]["command"][0] == "sh"
        # bounded host prompt cache (llama.cpp default 8 GiB OOM-killed a 1.5 GiB pod in M5)
        assert c["args"][c["args"].index("--cache-ram") + 1] == "256"
        mounts = {m["name"]: m for m in c["volumeMounts"]}
        assert mounts["weights"]["readOnly"] is True
    # autoscaled pool leaves replicas to the HPA
    primary = next(d for d in models if d["metadata"]["name"] == "cortex-model-primary")
    assert "replicas" not in primary["spec"]


def test_pdb_per_model_and_scaledobject_on_queue_depth() -> None:
    docs = render()
    assert len(kind(docs, "PodDisruptionBudget")) == 4  # 3 models + gateway
    [so] = kind(docs, "ScaledObject")
    trig = so["spec"]["triggers"][0]
    assert trig["type"] == "prometheus" and trig["metricType"] == "AverageValue"
    assert "requests_deferred" in trig["metadata"]["query"] and "requests_processing" in trig["metadata"]["query"]
    assert trig["metadata"]["threshold"] == "2"  # = slots per replica


def test_network_policies_isolate_models() -> None:
    pols = {d["metadata"]["name"]: d["spec"] for d in kind(render(), "NetworkPolicy")}
    models = pols["models-ingress-gateway-only"]
    assert models["egress"] == [] and set(models["policyTypes"]) == {"Ingress", "Egress"}
    sources = models["ingress"][0]["from"]
    assert sources[0]["podSelector"]["matchLabels"] == {"app.kubernetes.io/component": "gateway"}
    assert "namespaceSelector" in sources[1]
    # gateway has no internet egress unless the Modal slot is enabled
    assert not any("ipBlock" in str(rule) for rule in pols["gateway"]["egress"])
    pols_modal = {
        d["metadata"]["name"]: d["spec"]
        for d in kind(
            render("gateway.modal.enabled=true", "gateway.modal.apiBase=https://x/v1", "gateway.modal.model=m"),
            "NetworkPolicy",
        )
    }
    assert any("ipBlock" in str(rule) for rule in pols_modal["gateway"]["egress"])


def test_gateway_aliases_canary_split_and_fallbacks() -> None:
    cfg = gateway_config(render())
    by_alias: dict[str, list[dict[str, Any]]] = {}
    for m in cfg["model_list"]:
        by_alias.setdefault(m["model_name"], []).append(m)
    assert {"cortex-default", "cortex-judge", "cortex-router"} <= by_alias.keys()
    weights = {m["model_info"]["cortex_model"]: m["litellm_params"]["weight"] for m in by_alias["cortex-default"]}
    assert weights == {"primary": 90, "canary": 10}
    fallbacks = {k: v for d in cfg["router_settings"]["fallbacks"] for k, v in d.items()}
    assert fallbacks["cortex-default"] == ["cortex-fallback"]
    assert by_alias["cortex-fallback"][0]["model_info"]["cortex_model"] == "fallback"
    assert "cortex-gpu" not in by_alias  # Modal slot disabled by default
    for m in cfg["model_list"]:
        assert m["litellm_params"]["api_key"] == "os.environ/BACKEND_API_KEY"


@pytest.mark.parametrize(("weight", "expected"), [("0", {"primary"}), ("100", {"canary"})])
def test_canary_rollback_and_promotion_weights(weight: str, expected: set[str]) -> None:
    cfg = gateway_config(render(f"gateway.canary.weight={weight}"))
    got = {m["model_info"]["cortex_model"] for m in cfg["model_list"] if m["model_name"] == "cortex-default"}
    assert got == expected


def test_modal_slot_enters_fallback_chain_when_enabled() -> None:
    cfg = gateway_config(
        render("gateway.modal.enabled=true", "gateway.modal.apiBase=https://x/v1", "gateway.modal.model=m")
    )
    fallbacks = {k: v for d in cfg["router_settings"]["fallbacks"] for k, v in d.items()}
    assert fallbacks["cortex-default"] == ["cortex-gpu", "cortex-fallback"]
    [gpu] = [m for m in cfg["model_list"] if m["model_name"] == "cortex-gpu"]
    assert gpu["litellm_params"]["api_key"] == "os.environ/MODAL_ENDPOINT_KEY"
