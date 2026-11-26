"""Make the published M3 model repos public (or private again) on Hugging Face.

    uv run --extra modal modal run serve/deploy/modal/hf_visibility.py            # dry run: show current state
    uv run --extra modal modal run serve/deploy/modal/hf_visibility.py --public --yes

Runs in a small CPU container so the token never leaves the Modal secret.
"""

from __future__ import annotations

import modal

SECRET = "huggingface-secret"
USER = "intragalactic-stranger"
REPOS = [
    "Qwen3-4B-Instruct-2507-FP8-Dynamic",
    "Qwen3-4B-Instruct-2507-W8A8-INT8",
    "Qwen3-4B-Instruct-2507-W4A16-GPTQ",
    "Qwen3-4B-Instruct-2507-W4A16-AWQ",
    "Qwen3-4B-Instruct-2507-W4A16-GPTQ-BFSI",
]

app = modal.App("cortex-hf-visibility")
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("huggingface_hub>=0.26")


@app.function(image=image, secrets=[modal.Secret.from_name(SECRET)], timeout=300)
def visibility(make_public: bool | None) -> list[dict[str, object]]:
    import os

    from huggingface_hub import HfApi

    api = HfApi(token=os.environ["HF_TOKEN"])
    out: list[dict[str, object]] = []
    for name in REPOS:
        repo_id = f"{USER}/{name}"
        if make_public is not None:
            api.update_repo_settings(repo_id, private=not make_public, repo_type="model")
        info = api.model_info(repo_id)
        out.append({"repo": repo_id, "private": info.private, "files": len(info.siblings or [])})
    return out


@app.local_entrypoint()
def main(public: bool = False, private: bool = False, yes: bool = False) -> None:
    target: bool | None = True if public else (False if private else None)
    if target is not None and not yes:
        raise SystemExit("dry run only: add --yes to change visibility")
    for row in visibility.remote(target):
        print(row)
