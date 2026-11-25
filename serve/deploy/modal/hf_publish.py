"""Publish one M3 artifact from the Modal volume to Hugging Face (CPU container; the token stays in the Modal secret).

    uv run --extra modal modal run serve/deploy/modal/hf_publish.py \
        --artifact qwen3-4b-fp8-dynamic --repo Qwen3-4B-Instruct-2507-FP8-Dynamic --yes

The model card is read from ``serve/results/hf_cards/<repo>.md`` (built by ``serve/scripts/hf_cards.py``).
Without ``--yes`` it only prints what it would upload. Each publish is a separate, explicit owner decision (D-022).
"""

from __future__ import annotations

from pathlib import Path

import modal

SECRET = "huggingface-secret"
USER = "intragalactic-stranger"
app = modal.App("cortex-hf-publish")
hf_cache = modal.Volume.from_name("cortex-hf-cache")
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("huggingface_hub>=0.26")


@app.function(image=image, secrets=[modal.Secret.from_name(SECRET)], volumes={"/hf": hf_cache}, timeout=3600)
def publish(artifact: str, repo: str, card: str, private: bool) -> dict[str, object]:
    import os

    from huggingface_hub import HfApi

    api = HfApi(token=os.environ["HF_TOKEN"])
    folder = Path("/hf/artifacts") / artifact
    repo_id = f"{USER}/{repo}"
    api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
    (folder / "README.md").write_text(card)
    commit = api.upload_folder(repo_id=repo_id, folder_path=str(folder), commit_message=f"Cortex Serve M3: {artifact}")
    files = sorted(p.name for p in folder.iterdir())
    return {"repo": repo_id, "commit": commit.oid, "files": files, "private": private}


@app.local_entrypoint()
def main(artifact: str, repo: str, private: bool = False, yes: bool = False) -> None:
    card_path = Path("serve/results/hf_cards") / f"{repo}.md"
    card = card_path.read_text()
    print(f"[publish] {artifact} -> https://huggingface.co/{USER}/{repo} ({'private' if private else 'PUBLIC'})")
    if not yes:
        raise SystemExit("[publish] dry run only. Re-run with --yes after the owner approved this upload.")
    print(publish.remote(artifact, repo, card, private))
