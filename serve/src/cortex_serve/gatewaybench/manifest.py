"""Cluster-side manifest: what was actually running (images by digest, model files, chart values)."""

from __future__ import annotations

import json
import subprocess
from typing import Any


def _kubectl(args: list[str], timeout_s: float = 30) -> str | None:
    try:
        out = subprocess.run(["kubectl", *args], capture_output=True, text=True, timeout=timeout_s, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout if out.returncode == 0 else None


def pods_summary(pods_json: dict[str, Any]) -> list[dict[str, Any]]:
    """Reduce ``kubectl get pods -o json`` to name, labels, image and resolved image digest per container."""
    rows = []
    for pod in pods_json.get("items", []):
        meta, status = pod["metadata"], pod.get("status", {})
        ids = {c["name"]: c.get("imageID") for c in status.get("containerStatuses", [])}
        for c in pod["spec"]["containers"]:
            rows.append(
                {
                    "pod": meta["name"],
                    "component": meta.get("labels", {}).get("app.kubernetes.io/component"),
                    "model": meta.get("labels", {}).get("cortex.dev/model"),
                    "version": meta.get("labels", {}).get("cortex.dev/version"),
                    "weights": meta.get("annotations", {}).get("cortex.dev/weights"),
                    "hf_revision": meta.get("annotations", {}).get("cortex.dev/hf-revision"),
                    "container": c["name"],
                    "image": c["image"],
                    "image_id": ids.get(c["name"]),
                    "args": c.get("args"),
                    "resources": c.get("resources"),
                }
            )
    return rows


def cluster_manifest(namespace: str) -> dict[str, Any]:
    pods = _kubectl(["-n", namespace, "get", "pods", "-o", "json"])
    nodes = _kubectl(["get", "nodes", "-o", "json"])
    node_info = []
    if nodes:
        for n in json.loads(nodes).get("items", []):
            info = n["status"].get("nodeInfo", {})
            node_info.append(
                {
                    "name": n["metadata"]["name"],
                    "kubelet": info.get("kubeletVersion"),
                    "runtime": info.get("containerRuntimeVersion"),
                    "capacity": n["status"].get("capacity"),
                }
            )
    return {
        "namespace": namespace,
        "nodes": node_info,
        "containers": pods_summary(json.loads(pods)) if pods else None,
        "helm_values": _helm_values(namespace),
    }


def _helm_values(namespace: str) -> Any:
    try:
        out = subprocess.run(
            ["helm", "-n", namespace, "get", "values", "cortex", "--all", "-o", "json"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return json.loads(out.stdout) if out.returncode == 0 and out.stdout.strip() else None
