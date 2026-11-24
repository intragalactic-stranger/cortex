"""M3: produce quantized checkpoints with llm-compressor on a Modal L4 (own image, see docs/research M3 §1.4).

    uv run --extra modal modal run serve/deploy/modal/quantize_app.py --recipe fp8-dynamic             # dry run
    uv run --extra modal modal run serve/deploy/modal/quantize_app.py --recipe w4a16-gptq-g128 --calib bfsi --yes

Artifacts land in the ``cortex-hf-cache`` volume under ``/hf/artifacts/<name>/`` with a ``cortex_manifest.json``
(recipe, calibration set + sample-id digest, library versions, git SHA, wall time). vLLM loads them from there.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import modal

QUANT_PINS = [
    "llmcompressor==0.15.0",
    "compressed-tensors==0.20.1",
    "torch==2.14.1",
    "transformers==5.21.0",
    "datasets==5.1.0",
    "accelerate==1.15.0",
    "pandas>=2.2",
    "pyarrow>=18",
]
MODELS = {
    "qwen3-4b": ("Qwen/Qwen3-4B-Instruct-2507", "cdbee75f17c01a7cc42f958dc650907174af0554", "causal"),
    "gemma4-e4b": ("google/gemma-4-E4B-it", "4ecc2ba63bcac23ba4a767fc87025b1450f84c9c", "gemma4"),
}
RATES = {"L4": 0.80, "L40S": 1.95, "A100-80GB": 2.50}  # `modal billing rates`, 2026-11-24
MAX_MINUTES = 60

app = modal.App("cortex-quantize")
hf_cache = modal.Volume.from_name("cortex-hf-cache", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(*QUANT_PINS)
    .env({"HF_HOME": "/hf", "HF_HUB_ENABLE_HF_TRANSFER": "0"})
    .add_local_python_source("cortex_serve")
)


def _calibration(kind: str, tok: object, n: int, max_len: int) -> tuple[object, dict[str, str]]:
    from cortex_serve.quantize import calib
    from datasets import Dataset, load_dataset

    if kind == "generic":
        repo, rev, split = calib.GENERIC
        ds = load_dataset(repo, revision=rev, split=f"{split}[:{n}]")
        texts = [tok.apply_chat_template(m, tokenize=False) for m in ds["messages"]]  # type: ignore[attr-defined]
        ids = [str(i) for i in range(n)]
    elif kind == "bfsi":
        repo, rev, split = calib.BFSI
        rows = load_dataset(repo, revision=rev, split=split).to_pandas().to_dict("records")  # type: ignore[union-attr]
        sp = calib.split_bfsi(rows, n_calib=n, n_heldout=0)
        texts, ids = sp.calib, sp.calib_ids
    else:
        raise ValueError(f"unknown calibration set {kind!r}")
    enc = Dataset.from_dict({"text": texts}).map(
        lambda s: tok(s["text"], max_length=max_len, truncation=True, add_special_tokens=False),  # type: ignore[operator]
        remove_columns=["text"],
    )
    return enc, {
        "calib": kind,
        "calib_repo": repo,
        "calib_revision": rev,
        "n": str(len(texts)),
        "ids": calib.ids_digest(ids),
    }


@app.function(image=image, gpu="L4", memory=32768, volumes={"/hf": hf_cache}, timeout=MAX_MINUTES * 60)
def quantize(
    model_key: str, recipe: str, calib_kind: str, n: int, max_len: int, git_sha: str, git_dirty: bool, gpu: str = "L4"
) -> dict:
    import importlib.metadata as md

    from cortex_serve.quantize.recipes import RECIPES, build
    from llmcompressor import oneshot
    from transformers import AutoModelForCausalLM, AutoTokenizer

    repo, rev, kind = MODELS[model_key]
    spec = RECIPES[recipe]
    name = f"{model_key}-{recipe}" + (f"-{calib_kind}" if spec.needs_calibration else "")
    out = Path("/hf/artifacts") / name
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(repo, revision=rev)
    if kind == "gemma4":
        from llmcompressor.utils import load_context
        from transformers import Gemma4ForConditionalGeneration  # type: ignore[attr-defined]

        with load_context(Gemma4ForConditionalGeneration):
            model = Gemma4ForConditionalGeneration.from_pretrained(repo, revision=rev, dtype="auto")
        ignore = ["lm_head", "re:.*vision.*", "re:.*audio.*"]
    else:
        model = AutoModelForCausalLM.from_pretrained(repo, revision=rev, dtype="auto")
        ignore = ["lm_head"]
    meta: dict[str, str] = {"calib": "none"}
    kwargs: dict[str, object] = {}
    if spec.needs_calibration:
        ds, meta = _calibration(calib_kind, tok, n, max_len)
        kwargs = {"dataset": ds, "max_seq_length": max_len, "num_calibration_samples": n}
    oneshot(model=model, recipe=build(recipe, ignore), **kwargs)
    model.save_pretrained(out, save_compressed=True)
    tok.save_pretrained(out)
    manifest = {
        "artifact": name,
        "base_model": repo,
        "base_revision": rev,
        "recipe": recipe,
        "recipe_description": spec.description,
        **meta,
        "max_seq_length": max_len,
        "versions": {p: md.version(p) for p in ("llmcompressor", "compressed-tensors", "torch", "transformers")},
        "git_sha": git_sha,
        "git_dirty": git_dirty,
        "gpu": gpu,
        "wall_s": round(time.time() - t0, 1),
        "size_bytes": sum(f.stat().st_size for f in out.rglob("*") if f.is_file()),
    }
    (out / "cortex_manifest.json").write_text(json.dumps(manifest, indent=2))
    hf_cache.commit()
    return manifest


@app.local_entrypoint()
def main(
    recipe: str,
    model: str = "qwen3-4b",
    calib: str = "generic",
    n: int = 512,
    max_len: int = 2048,
    yes: bool = False,
    approved_gate: float = 60.0,
    gpu: str = "L4",
) -> None:
    from cortex_serve.quantize.recipes import RECIPES
    from cortex_serve.spend import guarded

    if recipe not in RECIPES or model not in MODELS:
        raise SystemExit(f"recipe must be one of {sorted(RECIPES)}, model one of {sorted(MODELS)}")
    worst = RATES[gpu] * MAX_MINUTES / 60
    print(
        f"[cost] quantize {model} {recipe} ({calib}): {gpu}, hard cap {MAX_MINUTES} min -> worst case ${worst:.2f}",
        flush=True,
    )
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True,
                                text=True, check=False).stdout.strip())  # fmt: skip
    with guarded(f"quantize {model} {recipe}", worst, approved_gate, yes):
        manifest = quantize.with_options(gpu=gpu).remote(model, recipe, calib, n, max_len, sha, dirty, gpu)
    dest = Path("serve/results/quantize") / manifest["artifact"]
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "cortex_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[done] {manifest['artifact']}: {manifest['size_bytes'] / 2**30:.2f} GiB in {manifest['wall_s']:.0f} s")
