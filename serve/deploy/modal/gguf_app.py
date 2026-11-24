"""M3 GGUF track on a Modal L4 with the pinned llama.cpp ``full-cuda`` image (docs/research M3 §3).

    uv run --extra modal modal run serve/deploy/modal/gguf_app.py --job qwen --yes     # E3 GGUF formats
    uv run --extra modal modal run serve/deploy/modal/gguf_app.py --job gemma --yes    # M1 Gemma PPL anomaly

qwen:  HF bf16 -> bf16 GGUF -> imatrix (generic, BFSI) -> Q8_0 / Q5_K_M / Q4_K_M / Q4_K_M+imatrix(generic|bfsi).
       Quality = llama-perplexity on WikiText-2 (all chunks, ctx 512) + KL divergence vs the bf16 GGUF, and
       perplexity on held-out eCFR parts. (llama-server has no prompt logprobs, so lm-eval MC tasks can't run on GGUF.)
gemma: four-way test from research M3 §4: QAT-bf16, QAT-Q4_0 (Google), IT-bf16, unsloth IT-Q8_0. Same WikiText-2
       setting as M1 (ctx 512, 20 chunks) plus all chunks; KLD of each quant vs its own bf16 parent.
"""

from __future__ import annotations

import io
import json
import re
import subprocess
import tarfile
import time
from pathlib import Path

import modal

FULL_CUDA = (
    "ghcr.io/ggml-org/llama.cpp:full-cuda-b12110"
    "@sha256:87803fec70c8046e04e051e49199894ad77e61cf74fd1dceb9ad5ee2f75de228"
)
QWEN = ("Qwen/Qwen3-4B-Instruct-2507", "cdbee75f17c01a7cc42f958dc650907174af0554")
GEMMA = {
    "qat-bf16": ("google/gemma-4-E2B-it-qat-q4_0-unquantized", "6befbaca7398925921802abd1f277b495b78b738", None),
    "it-bf16": ("google/gemma-4-E2B-it", "4adf15748de9e1a0097b6274cee9c15836c41c37", None),
    "qat-q4_0": (
        "google/gemma-4-E2B-it-qat-q4_0-gguf",
        "675cff42a74c774d6cb76f76d8eacb49b48c9b93",
        "gemma-4-E2B_q4_0-it.gguf",
    ),
    "it-q8_0": ("unsloth/gemma-4-E2B-it-GGUF", "0314792d7f1f7e229411f620751375812bb9faf2", "gemma-4-E2B-it-Q8_0.gguf"),
}
QUANTS = ["Q8_0", "Q5_K_M", "Q4_K_M"]
L4_USD_PER_HOUR = 0.80
MAX_MINUTES = 120
OUT = Path("/hf/gguf-m3")

app = modal.App("cortex-gguf")
hf_cache = modal.Volume.from_name("cortex-hf-cache", create_if_missing=True)
image = (
    modal.Image.from_registry(FULL_CUDA)
    .entrypoint([])
    .run_commands(
        "rm -f /usr/lib/python3*/EXTERNALLY-MANAGED",
        'command -v python || ln -s "$(command -v python3)" /usr/local/bin/python',
    )
    # Gemma 4 conversion needs transformers 5 (llama.cpp quantize README); Qwen3 converts with either.
    .uv_pip_install(
        "transformers==5.21.0",
        "pydantic>=2.10",
        "pyyaml>=6",
        "httpx>=0.28",
        "polars>=1.20",
        "pandas>=2.2",
        "pyarrow>=18",
        "datasets>=4",
        "huggingface_hub>=0.26",
    )
    .env({"HF_HOME": "/hf", "HF_HUB_ENABLE_HF_TRANSFER": "0"})
    .add_local_python_source("cortex_serve")
)

_KLD = re.compile(r"Mean\s+KLD:\s*([\d.]+)\s*±\s*([\d.]+)")
_TOP1 = re.compile(r"Same top p:\s*([\d.]+)\s*±\s*([\d.]+)\s*%")


def _sh(cmd: list[str], log: Path) -> str:
    print("[run]", " ".join(cmd), flush=True)
    out = subprocess.run(cmd, capture_output=True, text=True, check=False)
    text = out.stdout + out.stderr
    log.write_text(text)
    if out.returncode:
        raise RuntimeError(f"{cmd[0]} exited {out.returncode}: {text[-2000:]}")
    return text


def _convert(repo: str, rev: str, dest: Path, logs: Path) -> Path:
    from huggingface_hub import snapshot_download

    if not dest.exists():
        src = snapshot_download(repo, revision=rev, allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt"])
        _sh(["python3", "/app/convert_hf_to_gguf.py", src, "--outtype", "bf16", "--outfile", str(dest)],
            logs / f"convert-{dest.stem}.log")  # fmt: skip
        hf_cache.commit()
    return dest


def _ppl(model: Path, data: Path, logs: Path, tag: str, chunks: int | None = None, kld: str = "") -> dict[str, object]:
    from cortex_serve.servebench.quality.perplexity import parse_ppl

    cmd = ["/app/llama-perplexity", "-m", str(model), "-f", str(data), "-c", "512", "-ngl", "99"]
    if chunks:
        cmd += ["--chunks", str(chunks)]
    if kld == "base":
        cmd += ["--kl-divergence-base", str(logs / "base.kld")]
    elif kld:
        cmd = ["/app/llama-perplexity", "-m", str(model), "--kl-divergence-base", kld, "--kl-divergence", "-ngl", "99"]
    text = _sh(cmd, logs / f"ppl-{tag}.log")
    ppl = parse_ppl(text)
    rec: dict[str, object] = {"ppl": ppl[0] if ppl else None, "ppl_err": ppl[1] if ppl else None}
    if m := _KLD.search(text):
        rec.update(kld=float(m.group(1)), kld_err=float(m.group(2)))
    if m := _TOP1.search(text):
        rec.update(same_top1_pct=float(m.group(1)))
    if m := re.search(r"add_bos_token\s*=\s*(\w+)", text):
        rec["add_bos"] = m.group(1)
    return rec


def _texts(work: Path) -> dict[str, Path]:
    from cortex_serve.quantize import calib
    from cortex_serve.servebench.quality.datasets import wikitext2_raw
    from datasets import load_dataset
    from transformers import AutoTokenizer

    files = {"wiki": wikitext2_raw(work / "wiki.test.raw")}
    rows = load_dataset(calib.BFSI[0], revision=calib.BFSI[1], split=calib.BFSI[2]).to_pandas().to_dict("records")
    sp = calib.split_bfsi(rows, n_calib=512, n_heldout=200)
    (work / "bfsi-calib.txt").write_text("\n\n".join(sp.calib))
    (work / "bfsi-heldout.txt").write_text("\n\n".join(sp.heldout))
    tok = AutoTokenizer.from_pretrained(QWEN[0], revision=QWEN[1])
    g = load_dataset(calib.GENERIC[0], revision=calib.GENERIC[1], split=f"{calib.GENERIC[2]}[:512]")
    (work / "generic-calib.txt").write_text(
        "\n\n".join(tok.apply_chat_template(m, tokenize=False) for m in g["messages"])
    )
    files.update(bfsi_calib=work / "bfsi-calib.txt", bfsi_heldout=work / "bfsi-heldout.txt",
                 generic_calib=work / "generic-calib.txt")  # fmt: skip
    return files


def _qwen(work: Path, logs: Path) -> dict[str, object]:
    t = _texts(work)
    base = _convert(*QWEN, OUT / "qwen3-4b-bf16.gguf", logs)
    imat = {}
    for kind in ("generic", "bfsi"):
        imat[kind] = OUT / f"qwen3-4b-imatrix-{kind}.gguf"
        if not imat[kind].exists():
            _sh(["/app/llama-imatrix", "-m", str(base), "-f", str(t[f"{kind}_calib"]), "-o", str(imat[kind]),
                 "-ngl", "99", "-c", "512"], logs / f"imatrix-{kind}.log")  # fmt: skip
    variants: dict[str, Path] = {"bf16": base}
    for q in QUANTS:
        variants[q.lower()] = OUT / f"qwen3-4b-{q}.gguf"
    for kind in ("generic", "bfsi"):
        variants[f"q4_k_m-imat-{kind}"] = OUT / f"qwen3-4b-Q4_K_M-imat-{kind}.gguf"
    for name, path in variants.items():
        if path.exists() or name == "bf16":
            continue
        qtype = name.split("-")[0].upper()
        cmd = ["/app/llama-quantize"]
        if "imat" in name:
            cmd += ["--imatrix", str(imat[name.rsplit("-", 1)[1]])]
        _sh([*cmd, str(base), str(path), qtype], logs / f"quantize-{name}.log")
    hf_cache.commit()
    res: dict[str, object] = {}
    for name, path in variants.items():
        wiki = _ppl(path, t["wiki"], logs, f"wiki-{name}", kld="base" if name == "bf16" else "")
        if name != "bf16":
            wiki.update(_ppl(path, t["wiki"], logs, f"kld-{name}", kld=str(logs / "base.kld")))
        res[name] = {"bytes": path.stat().st_size, "wikitext2": wiki,
                     "bfsi_heldout": _ppl(path, t["bfsi_heldout"], logs, f"bfsi-{name}")}  # fmt: skip
    return res


def _gemma(work: Path, logs: Path) -> dict[str, object]:
    from cortex_serve.servebench.quality.datasets import wikitext2_raw
    from huggingface_hub import hf_hub_download

    wiki = wikitext2_raw(work / "wiki.test.raw")
    paths: dict[str, Path] = {}
    for tag, (repo, rev, file) in GEMMA.items():
        if file is None:
            paths[tag] = _convert(repo, rev, OUT / f"gemma4-e2b-{tag}.gguf", logs)
        else:
            paths[tag] = Path(hf_hub_download(repo, file, revision=rev))
    res: dict[str, object] = {}
    for parent, child in (("qat-bf16", "qat-q4_0"), ("it-bf16", "it-q8_0")):
        kbase = logs / f"{parent}.kld"
        for tag in (parent, child):
            r = {"m1_setting_20_chunks": _ppl(paths[tag], wiki, logs, f"wiki20-{tag}", chunks=20)}
            if tag == parent:
                _sh(["/app/llama-perplexity", "-m", str(paths[tag]), "-f", str(wiki), "-c", "512", "-ngl", "99",
                     "--kl-divergence-base", str(kbase)], logs / f"kldbase-{tag}.log")  # fmt: skip
                r["all_chunks"] = _ppl(paths[tag], wiki, logs, f"wiki-{tag}")
            else:
                r["all_chunks"] = _ppl(paths[tag], wiki, logs, f"kld-{tag}", kld=str(kbase))
            res[tag] = r
    return res


@app.function(image=image, gpu="L4", memory=49152, volumes={"/hf": hf_cache}, timeout=MAX_MINUTES * 60)
def run_job(job: str, git: dict[str, object]) -> bytes:
    work, logs = Path("/tmp/work"), Path(f"/tmp/gguf-{job}")
    work.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    try:
        results = _qwen(work, logs) if job == "qwen" else _gemma(work, logs)
        status = "ok"
    except Exception as exc:
        results, status = {"error": repr(exc)}, "failed"
    meta = {"job": job, "status": status, "image": FULL_CUDA, "gpu": "L4", "wall_s": round(time.time() - t0),
            "git": git, "results": results}  # fmt: skip
    (logs / "summary.json").write_text(json.dumps(meta, indent=2))
    hf_cache.commit()
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(logs, arcname=f"gguf-{job}")
    return buf.getvalue()


@app.local_entrypoint()
def main(job: str, yes: bool = False, approved_gate: float = 60.0) -> None:
    from cortex_serve.spend import guarded

    if job not in ("qwen", "gemma"):
        raise SystemExit("--job must be qwen or gemma")
    worst = L4_USD_PER_HOUR * MAX_MINUTES / 60
    print(f"[cost] gguf {job}: L4, hard cap {MAX_MINUTES} min -> worst case ${worst:.2f}", flush=True)
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True,
                                text=True, check=False).stdout.strip())  # fmt: skip
    with guarded(f"gguf {job}", worst, approved_gate, yes):
        data = run_job.remote(job, {"sha": sha, "dirty": dirty})
    dest = Path("serve/results/quality-gguf-gpu") / time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        tar.extractall(dest, filter="data")
    summary = json.loads((dest / f"gguf-{job}" / "summary.json").read_text())
    print(f"[done] gguf {job}: {summary['status']} in {summary['wall_s']} s -> {dest}/gguf-{job}/")
