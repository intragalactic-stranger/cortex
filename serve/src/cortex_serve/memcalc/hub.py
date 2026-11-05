"""Hugging Face Hub access: resolve a revision, fetch config.json, and read safetensors headers.

Safetensors files start with an 8-byte little-endian header length followed by a JSON header that
lists every tensor's dtype and shape. Two HTTP range requests per shard therefore give exact
parameter counts without downloading any weights.
"""

from __future__ import annotations

import gzip
import json
import os
import struct
import time
from pathlib import Path
from typing import Any

import httpx

HF = "https://huggingface.co"
TIMEOUT = httpx.Timeout(30.0, connect=10.0)
CACHE_DIR = Path(os.environ.get("CORTEX_CACHE", Path.home() / ".cache" / "cortex")) / "memcalc"


def _headers(extra: dict[str, str] | None = None) -> dict[str, str]:
    headers = {"User-Agent": "cortex-memcalc/0.1"}
    token = os.environ.get("HF_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if extra:
        headers.update(extra)
    return headers


def _get(client: httpx.Client, url: str, extra: dict[str, str] | None = None, retries: int = 3) -> bytes:
    for attempt in range(retries):
        try:
            resp = client.get(url, headers=_headers(extra), follow_redirects=True)
            resp.raise_for_status()
            return resp.content
        except (httpx.TransportError, httpx.HTTPStatusError) as exc:
            retriable = isinstance(exc, httpx.TransportError) or exc.response.status_code >= 500
            if not retriable or attempt == retries - 1:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def resolve_revision(client: httpx.Client, repo: str, revision: str | None) -> str:
    url = f"{HF}/api/models/{repo}" + (f"/revision/{revision}" if revision else "")
    info: dict[str, Any] = json.loads(_get(client, url))
    return str(info["sha"])


def fetch_config(client: httpx.Client, repo: str, sha: str) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(_get(client, f"{HF}/{repo}/resolve/{sha}/config.json"))
    return data


def _shard_header(client: httpx.Client, repo: str, sha: str, fname: str) -> dict[str, list[Any]]:
    url = f"{HF}/{repo}/resolve/{sha}/{fname}"
    (length,) = struct.unpack("<Q", _get(client, url, {"Range": "bytes=0-7"}))
    raw: dict[str, Any] = json.loads(_get(client, url, {"Range": f"bytes=8-{7 + length}"}))
    raw.pop("__metadata__", None)
    return {name: [spec["dtype"], spec["shape"]] for name, spec in raw.items()}


def fetch_tensor_table(client: httpx.Client, repo: str, sha: str) -> dict[str, list[Any]]:
    cache = CACHE_DIR / f"{repo.replace('/', '__')}@{sha}.tensors.json.gz"
    if cache.exists():
        cached: dict[str, list[Any]] = json.loads(gzip.decompress(cache.read_bytes()))
        return cached
    try:
        index = json.loads(_get(client, f"{HF}/{repo}/resolve/{sha}/model.safetensors.index.json"))
        shards = sorted(set(index["weight_map"].values()))
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code != 404:
            raise
        shards = ["model.safetensors"]
    table: dict[str, list[Any]] = {}
    for shard in shards:
        table.update(_shard_header(client, repo, sha, shard))
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(gzip.compress(json.dumps(table, sort_keys=True).encode()))
    return table


def client() -> httpx.Client:
    return httpx.Client(timeout=TIMEOUT)
