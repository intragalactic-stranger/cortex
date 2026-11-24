"""M3 E3: quality of a checkpoint served by vLLM, measured with lm-evaluation-harness in the same container.

    uv run --extra modal modal run serve/deploy/modal/eval_app.py --name qwen3-4b-bf16 \
        --model Qwen/Qwen3-4B-Instruct-2507 --revision cdbee75f17c01a7cc42f958dc650907174af0554 --yes
    uv run --extra modal modal run serve/deploy/modal/eval_app.py --name qwen3-4b-fp8-dynamic \
        --model /hf/artifacts/qwen3-4b-fp8-dynamic --yes

Every variant gets the same items (``--limit`` takes the first N, so comparisons are paired; per-item results are
kept with ``--log_samples`` for paired bootstrap CIs). Tasks and sizes are fixed below (docs/research M3 §2):
loglikelihood tasks go to ``/v1/completions`` (echo + logprobs), generative ones to ``/v1/chat/completions``.
Domain perplexity uses held-out eCFR parts (never seen by the BFSI calibration set, D-026).
"""

from __future__ import annotations

import io
import json
import math
import subprocess
import tarfile
import time
from pathlib import Path

import modal

# same pinned image as ENGINE_IMAGES["vllm"] in serve/deploy/modal/bench_app.py
VLLM_IMAGE = "vllm/vllm-openai:v0.32.0@sha256:befccf0e7e4089aa23926bb6f887a85c0d46c2a99e6de5c5ffdf5cb7957c5b34"
LM_EVAL = "lm_eval[api,ifeval]==0.4.13"
MMLU_PRO_LIMIT = 1000  # leaderboard_mmlu_pro is one task of 12,032 items: first 1,000
BFSI_HELDOUT_N = 200
RATES = {"L4": 0.80, "L40S": 1.95, "H100": 3.95}  # `modal billing rates`, 2026-11-24
MAX_MINUTES = 120

app = modal.App("cortex-eval")
hf_cache = modal.Volume.from_name("cortex-hf-cache", create_if_missing=True)
image = (
    modal.Image.from_registry(VLLM_IMAGE)
    .entrypoint([])
    .run_commands('command -v python || ln -s "$(command -v python3)" /usr/local/bin/python')
    .uv_pip_install(LM_EVAL, "pydantic>=2.10", "pyyaml>=6", "httpx>=0.28", "pandas>=2.2", "datasets>=4")
    # lm-eval must not move vLLM's torch/transformers (research M3 risk 6): fail the build if it did
    .run_commands('python3 -c "import torch, transformers, vllm; print(torch.__version__, transformers.__version__)"')
    .env({"HF_HOME": "/hf", "HF_HUB_ENABLE_HF_TRANSFER": "0", "HF_ALLOW_CODE_EVAL": "1"})
    .add_local_python_source("cortex_serve")
)


def _lm_eval(kind: str, model_args: str, out: Path, tasks: str, limit: int | None, quick: int = 0) -> None:
    limit = quick or limit
    cmd = ["lm-eval", "run", "--model", kind, "--model_args", model_args, "--tasks", tasks, "--seed", "1234",
           "--output_path", str(out), "--log_samples"]  # fmt: skip
    if kind == "local-chat-completions":
        cmd.append("--apply_chat_template")
    if limit:
        cmd += ["--limit", str(limit)]
    print("[eval]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def _domain_ppl(base_url: str, n: int) -> dict[str, float]:
    """Token-level perplexity on held-out eCFR sections via prompt logprobs (echo=True)."""
    import httpx
    from cortex_serve.quantize import calib
    from datasets import load_dataset

    repo, rev, split = calib.BFSI
    rows = load_dataset(repo, revision=rev, split=split).to_pandas().to_dict("records")
    held = calib.split_bfsi(rows, n_calib=0, n_heldout=n).heldout
    nll, n_tok = 0.0, 0
    with httpx.Client(timeout=300) as client:
        for text in held:
            body = {"model": "model", "prompt": text[:6000], "max_tokens": 1, "echo": True, "logprobs": 0}
            r = client.post(f"{base_url}/v1/completions", json={**body, "temperature": 0})
            r.raise_for_status()
            lps = [x for x in r.json()["choices"][0]["logprobs"]["token_logprobs"] if x is not None]
            lps = lps[:-1]  # drop the 1 generated token, keep prompt tokens
            nll -= sum(lps)
            n_tok += len(lps)
    return {"bfsi_heldout_ppl": math.exp(nll / n_tok), "bfsi_heldout_tokens": float(n_tok),
            "bfsi_heldout_sections": float(len(held))}  # fmt: skip


@app.function(image=image, gpu="L4", volumes={"/hf": hf_cache}, timeout=MAX_MINUTES * 60)
def evaluate(name: str, model: str, revision: str, tokenizer: str, max_len: int, extra: list[str],
             git: dict[str, object], quick: int = 0, add_bos: bool = False, parts: str = "all") -> bytes:  # fmt: skip
    from cortex_serve.servebench.config import ArtifactSpec
    from cortex_serve.servebench.engines.process import ProcessEngine

    out = Path("/tmp/eval") / name
    out.mkdir(parents=True, exist_ok=True)
    art = ArtifactSpec(name=name, repo=model, revision=revision, family=tokenizer, quant=name)
    args: dict[str, object] = {"max_model_len": max_len, "gpu_memory_utilization": 0.9}
    if extra:
        args["extra"] = extra
    eng = ProcessEngine("vllm", VLLM_IMAGE, args, "/hf/gguf", port=8000)
    t0 = time.time()
    eng.start(art)
    try:
        base = "http://127.0.0.1:8000"
        # local checkpoints carry their own tokenizer files; hub models use the base repo's tokenizer
        tok = f"tokenizer={model if model.startswith('/') else tokenizer}"
        common = f"model=model,{tok},num_concurrent=32,max_retries=3,timeout=600"
        # Gemma needs <bos>; lm-eval's API models tokenize loglikelihood requests without special tokens by default
        ll = f"{common},base_url={base}/v1/completions" + (",add_bos_token=True" if add_bos else "")
        want = set(parts.split(",")) if parts != "all" else {"loglik", "gen", "domain"}
        if "loglik" in want:
            _lm_eval("local-completions", ll, out / "loglik", "arc_challenge,wikitext", None, quick)
            _lm_eval("local-completions", ll, out / "hellaswag", "hellaswag", 2000, quick)
            _lm_eval("local-completions", ll, out / "mmlu_pro", "leaderboard_mmlu_pro", MMLU_PRO_LIMIT, quick)
        if "gen" in want:
            _lm_eval("local-chat-completions", f"{common},base_url={base}/v1/chat/completions", out / "gen",
                     "gsm8k,ifeval", None, quick)  # fmt: skip
        domain = _domain_ppl(base, quick or BFSI_HELDOUT_N) if "domain" in want else None
        (out / "engine.log").write_text(eng.logs())
    finally:
        eng.stop()
    gpu_name = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                              capture_output=True, text=True, check=False).stdout.strip()  # fmt: skip
    qm = Path(model) / "cortex_manifest.json"
    meta = {"name": name, "model": model, "revision": revision, "max_model_len": max_len, "extra": extra,
            "engine_image": VLLM_IMAGE, "lm_eval": LM_EVAL, "gpu": gpu_name, "wall_s": round(time.time() - t0),
            "git": git, "domain": domain, "quick_limit": quick, "add_bos": add_bos, "parts": parts,
            "quantize_manifest": json.loads(qm.read_text()) if qm.exists() else None}  # fmt: skip
    mname = "eval_manifest.json" if parts == "all" else f"eval_manifest_{parts.replace(',', '+')}.json"
    (out / mname).write_text(json.dumps(meta, indent=2))
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(out, arcname=name)
    return buf.getvalue()


@app.local_entrypoint()
def main(name: str, model: str, revision: str = "local", tokenizer: str = "Qwen/Qwen3-4B-Instruct-2507",
         max_len: int = 4096, extra: str = "",
         gpu: str = "L4", quick: int = 0, add_bos: bool = False, parts: str = "all", yes: bool = False,
         approved_gate: float = 60.0) -> None:  # fmt: skip
    from cortex_serve.spend import guarded

    worst = RATES[gpu] * MAX_MINUTES / 60
    print(f"[cost] eval {name}: {gpu}, hard cap {MAX_MINUTES} min -> worst case ${worst:.2f}", flush=True)
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True,
                                text=True, check=False).stdout.strip())  # fmt: skip
    fn = evaluate.with_options(gpu=gpu)
    with guarded(f"eval {name}", worst, approved_gate, yes):
        data = fn.remote(name, model, revision, tokenizer, max_len, extra.split() if extra else [],
                         {"sha": sha, "dirty": dirty}, quick, add_bos, parts)  # fmt: skip
    dest = Path("serve/results/quality-gpu" + ("-quick" if quick else ""))
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        tar.extractall(dest, filter="data")
    print(f"[done] {name}: results under {dest}/{name}/")
