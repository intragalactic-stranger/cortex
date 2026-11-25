"""Check the Hugging Face secret on Modal without revealing it: key present, account name, token role.

uv run --extra modal modal run serve/deploy/modal/hf_check.py
"""

from __future__ import annotations

import modal

SECRET = "huggingface-secret"
app = modal.App("cortex-hf-check")
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("huggingface_hub>=0.26")


@app.function(image=image, secrets=[modal.Secret.from_name(SECRET)], timeout=120)
def check() -> dict[str, object]:
    import os

    from huggingface_hub import HfApi

    keys = sorted(k for k in os.environ if "HF" in k or "HUGGING" in k)
    token = os.environ.get("HF_TOKEN")
    if not token:
        return {"keys_present": keys, "ok": False, "reason": "no HF_TOKEN key in the secret"}
    me = HfApi(token=token).whoami()
    auth = me.get("auth", {}).get("accessToken", {})
    return {"keys_present": keys, "ok": True, "user": me.get("name"), "token_role": auth.get("role")}


@app.local_entrypoint()
def main() -> None:
    print(check.remote())
