"""Copy the base repo's non-weight files (processor/chat template/generation config) into M3 artifact dirs.

    uv run --extra modal modal run serve/deploy/modal/fix_artifact_files.py \
        --artifacts gemma4-e4b-w4a16-gptq-g32-sym-generic,gemma4-e4b-w4a16-rtn-g32-sym

vLLM loads Gemma 4 as a multimodal model and needs ``processor_config.json`` etc. even for text-only serving;
llm-compressor's ``save_pretrained`` only wrote the tokenizer. CPU-only, a few cents.
"""

from __future__ import annotations

import modal

BASES = {"gemma4-e4b": ("google/gemma-4-E4B-it", "4ecc2ba63bcac23ba4a767fc87025b1450f84c9c")}
KEEP = (".json", ".jinja", ".model", ".txt")
SKIP = ("config.json", "model.safetensors.index.json")  # the artifact's own quantized config must not be replaced

app = modal.App("cortex-fix-artifacts")
hf_cache = modal.Volume.from_name("cortex-hf-cache")
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("huggingface_hub>=0.26").env({"HF_HOME": "/hf"})


@app.function(image=image, volumes={"/hf": hf_cache}, timeout=600)
def fix(artifacts: list[str]) -> list[str]:
    import shutil
    from pathlib import Path

    from huggingface_hub import hf_hub_download, list_repo_files

    done = []
    for name in artifacts:
        repo, rev = next(v for k, v in BASES.items() if name.startswith(k))
        dest = Path("/hf/artifacts") / name
        for f in list_repo_files(repo, revision=rev):
            if f.endswith(KEEP) and f not in SKIP and "/" not in f and not (dest / f).exists():
                shutil.copy(hf_hub_download(repo, f, revision=rev), dest / f)
                done.append(f"{name}/{f}")
    hf_cache.commit()
    return done


@app.local_entrypoint()
def main(artifacts: str) -> None:
    for line in fix.remote(artifacts.split(",")):
        print("[copied]", line)
