"""Build Hugging Face model cards for the M3 artifacts from committed manifests and results.

uv run python serve/scripts/hf_cards.py   # writes serve/results/hf_cards/<name>.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import polars as pl
from cortex_serve.quantize.card import build_card, quality_table

sys.path.insert(0, str(Path(__file__).parent))
from quality_gpu_table import load

RES = Path("serve/results")
OUT = RES / "hf_cards"
USER = "intragalactic-stranger"
QWEN_TAGS = ["qwen3", "quantized", "compressed-tensors", "llm-compressor", "vllm"]


def speed_md(fmt_l4: str | None, fmt_h100: str | None) -> str:
    def point(pattern: str, level: float) -> tuple[float, float] | None:
        files = sorted(RES.glob(pattern))
        if not files:
            return None
        df = pl.read_parquet(files[-1]).filter((pl.col("workload") == "chat-balanced") & (pl.col("level") == level))
        return (float(df["tpot_p50_s"].median()) * 1000, float(df["output_throughput_tok_s"].median()))

    rows = ["| GPU | Metric | bf16 | This model |", "|---|---|---|---|"]
    if fmt_l4:
        b1, v1 = point("e1-l4-vllm/*/points.parquet", 1), point(f"e2-l4-vllm-{fmt_l4}/*/points.parquet", 1)
        b64, v64 = point("e1-l4-vllm/*/points.parquet", 64), point(f"e2-l4-vllm-{fmt_l4}/*/points.parquet", 64)
        if b1 and v1:
            rows.append(f"| NVIDIA L4 | ms per output token, 1 user | {b1[0]:.1f} | {v1[0]:.1f} |")
        if b64 and v64:
            rows.append(f"| NVIDIA L4 | output tokens/s, 64 users | {b64[1]:,.0f} | {v64[1]:,.0f} |")
    if fmt_h100:
        b1, v1 = point("e2-h100-vllm-bf16/*/points.parquet", 1), point(f"e2-h100-vllm-{fmt_h100}/*/points.parquet", 1)
        b128 = point("e2-h100-vllm-bf16/*/points.parquet", 128)
        v128 = point(f"e2-h100-vllm-{fmt_h100}/*/points.parquet", 128)
        if b1 and v1:
            rows.append(f"| NVIDIA H100 80GB | ms per output token, 1 user | {b1[0]:.2f} | {v1[0]:.2f} |")
        if b128 and v128:
            rows.append(f"| NVIDIA H100 80GB | output tokens/s, 128 users | {b128[1]:,.0f} | {v128[1]:,.0f} |")
    return "\n".join(rows) if len(rows) > 2 else "Not measured for this artifact."


def vllm_usage(repo: str, extra: str = "") -> str:
    return f"```bash\nvllm serve {USER}/{repo}{extra}\n```"


CARDS = {
    "Qwen3-4B-Instruct-2507-FP8-Dynamic": dict(
        art="qwen3-4b-fp8-dynamic",
        l4="fp8-dynamic",
        h100="fp8-dynamic",
        title="Qwen3-4B-Instruct-2507 · FP8 dynamic (W8A8)",
        summary="FP8 E4M3 weights (per channel) and dynamically quantized FP8 activations (per token). Data-free.",
        caveats=[
            "FP8 compute needs compute capability ≥ 8.9 (Ada L4/L40S, Hopper H100). Older GPUs fall back to weight-only FP8.",
            "Quality matched both bf16 and Qwen's own FP8 release within the confidence intervals.",
        ],
    ),
    "Qwen3-4B-Instruct-2507-W8A8-INT8": dict(
        art="qwen3-4b-w8a8-int8-generic",
        l4="w8a8-int8",
        h100="w8a8-int8",
        title="Qwen3-4B-Instruct-2507 · W8A8 INT8 (SmoothQuant + GPTQ)",
        summary="INT8 weights (per channel) and dynamic INT8 activations (per token), SmoothQuant strength 0.8 then GPTQ.",
        caveats=["Runs on CUTLASS INT8 kernels in vLLM (Ampere and newer)."],
    ),
    "Qwen3-4B-Instruct-2507-W4A16-GPTQ": dict(
        art="qwen3-4b-w4a16-gptq-g128-generic",
        l4="w4a16-gptq",
        h100="w4a16-gptq",
        title="Qwen3-4B-Instruct-2507 · W4A16 GPTQ (group 128)",
        summary="INT4 weights, group size 128, symmetric, static activation order; activations stay bf16.",
        caveats=[
            "Weight-only INT4 is 2.7× faster per token than bf16 on an L4, but slower than bf16 on a busy H100 "
            "(dequantization becomes the bottleneck at high batch). Prefer FP8 for high-concurrency H100 serving.",
            "Perplexity +5.5%; MMLU-Pro −2.3 (inside the CI); reasoning and instruction tasks within noise.",
        ],
    ),
    "Qwen3-4B-Instruct-2507-W4A16-AWQ": dict(
        art="qwen3-4b-w4a16-awq-asym-generic",
        l4="w4a16-awq",
        h100=None,
        title="Qwen3-4B-Instruct-2507 · W4A16 AWQ (asymmetric, group 128)",
        summary="AWQ activation-aware scales followed by round-to-nearest INT4, group 128, asymmetric (zero point).",
        caveats=[
            "H100 speed not measured for this variant.",
            "Quantized on an L40S: AWQ's scale search ran out of memory on a 24 GB L4 at 512 × 2,048 calibration.",
        ],
    ),
    "Qwen3-4B-Instruct-2507-W4A16-GPTQ-BFSI": dict(
        art="qwen3-4b-w4a16-gptq-g128-bfsi",
        l4=None,
        h100=None,
        title="Qwen3-4B-Instruct-2507 · W4A16 GPTQ calibrated on US financial regulation (research artifact)",
        summary="Same recipe as the generic GPTQ release, calibrated only on eCFR Titles 12 and 17 (banking, securities).",
        caveats=[
            "**Research artifact, not a general-purpose model.** Domain-only calibration halved the in-domain "
            "perplexity damage (+3.0% vs +7.3%) but weakened format-following: only 45% of GSM8K answers keep "
            "the `#### N` format (80% for generic calibration), so strict GSM8K drops 38 points while arithmetic "
            "(flexible extract) drops 3.2. IFEval −3.9. Calibrate on a domain + general-chat mix instead.",
            "Speed is identical to the generic GPTQ release (same format).",
        ],
    ),
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    base = load("qwen3-4b-bf16")
    for repo, c in CARDS.items():
        man = json.loads((RES / "quantize" / c["art"] / "cortex_manifest.json").read_text())
        q = quality_table(load(c["art"]), base, "bf16")
        card = build_card(f"{USER}/{repo}", man, c["title"], c["summary"], q, speed_md(c["l4"], c["h100"]),
                          c["caveats"], QWEN_TAGS, vllm_usage(repo))  # fmt: skip
        (OUT / f"{repo}.md").write_text(card)
        print("wrote", OUT / f"{repo}.md")


if __name__ == "__main__":
    main()
