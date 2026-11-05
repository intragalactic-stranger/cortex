"""Build a ``ModelSpec`` from a local model directory, a config file, or a Hugging Face repo id.

A local model directory may contain:
  config.json               (required)
  tensors.json[.gz]         safetensors header table, ``{name: [dtype, shape]}`` (optional, exact params)
  meta.json                 ``{"repo": ..., "revision": ...}`` (optional, provenance)
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

from cortex_serve.memcalc import hub
from cortex_serve.memcalc.families import build_layers, build_moe, text_config, tied_embeddings
from cortex_serve.memcalc.model import ModelSpec
from cortex_serve.memcalc.params import TensorTable, analytic_params, classify_tensors


def _read_json(path: Path) -> Any:
    data = path.read_bytes()
    if path.suffix == ".gz":
        data = gzip.decompress(data)
    return json.loads(data)


def build_spec(
    name: str, cfg: dict[str, Any], tensors: TensorTable | None = None, revision: str | None = None
) -> ModelSpec:
    layers = build_layers(cfg)
    if tensors is not None:
        params = classify_tensors(tensors, layers, tied_embeddings(cfg))
    else:
        analytic = analytic_params(cfg)
        if analytic is None:
            raise ValueError(
                f"{name}: no safetensors header available and no analytic formula for "
                f"model_type={text_config(cfg).get('model_type')!r}; provide tensors.json"
            )
        params = analytic
    tc = text_config(cfg)
    return ModelSpec(
        name=name,
        revision=revision,
        model_type=str(tc.get("model_type")),
        hidden_size=int(tc["hidden_size"]),
        vocab_size=int(tc["vocab_size"]),
        tied_embeddings=tied_embeddings(cfg),
        max_context=tc.get("max_position_embeddings"),
        layers=layers,
        moe=build_moe(cfg),
        params=params,
    )


def load_local(path: Path) -> ModelSpec:
    if path.is_file():
        return build_spec(path.stem, _read_json(path))
    cfg = _read_json(path / "config.json")
    tensors: TensorTable | None = None
    for candidate in ("tensors.json.gz", "tensors.json"):
        if (path / candidate).exists():
            tensors = {k: (v[0], v[1]) for k, v in _read_json(path / candidate).items()}
            break
    meta: dict[str, Any] = _read_json(path / "meta.json") if (path / "meta.json").exists() else {}
    return build_spec(str(meta.get("repo", path.name)), cfg, tensors, meta.get("revision"))


def load_hub(repo: str, revision: str | None = None, exact_params: bool = True) -> ModelSpec:
    with hub.client() as client:
        sha = hub.resolve_revision(client, repo, revision)
        cfg = hub.fetch_config(client, repo, sha)
        tensors: TensorTable | None = None
        if exact_params:
            table = hub.fetch_tensor_table(client, repo, sha)
            tensors = {k: (v[0], v[1]) for k, v in table.items()}
    return build_spec(repo, cfg, tensors, sha)


def load_model(source: str, revision: str | None = None) -> ModelSpec:
    path = Path(source)
    if path.exists():
        return load_local(path)
    if source.count("/") == 1:
        return load_hub(source, revision)
    raise ValueError(f"{source!r} is neither a local path nor a Hugging Face repo id (org/name)")
