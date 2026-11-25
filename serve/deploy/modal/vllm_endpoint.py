"""M5 GPU fallback: our M3 FP8 Qwen3-4B served by vLLM on a Modal L4 as an OpenAI-compatible web endpoint.

    modal deploy serve/deploy/modal/vllm_endpoint.py     # prints the URL
    modal app stop cortex-gpu-endpoint                   # after the demo

Safety (D-030, D-052): requests need the API key held in the Modal secret ``cortex-vllm-key`` (vLLM ``--api-key``);
at most one container; it scales to zero after 2 idle minutes, so an idle deployment costs nothing.
"""

from __future__ import annotations

import os
import subprocess

import modal

VLLM_IMAGE = "vllm/vllm-openai:v0.32.0@sha256:befccf0e7e4089aa23926bb6f887a85c0d46c2a99e6de5c5ffdf5cb7957c5b34"
MODEL_DIR = "/hf/artifacts/qwen3-4b-fp8-dynamic"
SERVED_NAME = "cortex-gpu"
PORT = 8000

app = modal.App("cortex-gpu-endpoint")
hf_cache = modal.Volume.from_name("cortex-hf-cache")
image = (
    modal.Image.from_registry(VLLM_IMAGE)
    .entrypoint([])
    .run_commands('command -v python || ln -s "$(command -v python3)" /usr/local/bin/python')
    .env({"HF_HOME": "/hf", "HF_HUB_ENABLE_HF_TRANSFER": "0"})
)


@app.function(
    image=image,
    gpu="L4",
    volumes={"/hf": hf_cache},
    secrets=[modal.Secret.from_name("cortex-vllm-key")],
    max_containers=1,
    scaledown_window=120,
    timeout=3600,
)
@modal.concurrent(max_inputs=64)
@modal.web_server(port=PORT, startup_timeout=900)
def serve() -> None:
    cmd = [
        "vllm", "serve", MODEL_DIR, "--served-model-name", SERVED_NAME, "--host", "0.0.0.0", "--port", str(PORT),
        "--max-model-len", "4096", "--gpu-memory-utilization", "0.9", "--api-key", os.environ["VLLM_API_KEY"],
    ]  # fmt: skip
    subprocess.Popen(cmd)
