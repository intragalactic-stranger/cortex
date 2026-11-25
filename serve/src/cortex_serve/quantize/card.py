"""Hugging Face model cards for M3 artifacts, built only from committed manifests and measured results."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

REPO_URL = "https://github.com/intragalactic-stranger/cortex"

QUALITY_ROWS = (
    ("arc_challenge", "ARC-Challenge (acc_norm, 1,172)", True),
    ("hellaswag", "HellaSwag (acc_norm, first 2,000)", True),
    ("leaderboard_mmlu_pro", "MMLU-Pro (logprob MC, first 1,000)", True),
    ("gsm8k", "GSM8K 5-shot (strict)", True),
    ("gsm8k_flex", "GSM8K 5-shot (flexible extract)", True),
    ("ifeval", "IFEval (prompt-level strict)", True),
    ("arc_cot", "ARC-Challenge, reasoning-first (n = 500, own scorer)", True),
    ("wikitext", "WikiText-2 word perplexity (lower is better)", False),
    ("bfsi_ppl", "Held-out US financial regulation perplexity (lower is better)", False),
)


def _fmt(value: float, pct: bool) -> str:
    return f"{value * 100:.1f}" if pct else f"{value:.3f}"


def quality_table(variant: Mapping[str, tuple[float, float | None]], base: Mapping[str, tuple[float, float | None]],
                  base_label: str) -> str:  # fmt: skip
    """Markdown table: metric | this model | base | delta (only metrics present in both)."""
    lines = [f"| Benchmark | This model | {base_label} | Δ |", "|---|---|---|---|"]
    for key, label, pct in QUALITY_ROWS:
        if key not in variant or key not in base:
            continue
        v, se = variant[key]
        b, _ = base[key]
        ci = f" ±{1.96 * se * 100:.1f}" if (se is not None and pct) else ""
        delta = f"{(v - b) * 100:+.1f}" if pct else f"{(v / b - 1) * 100:+.1f}%"
        lines.append(f"| {label} | {_fmt(v, pct)}{ci} | {_fmt(b, pct)} | {delta} |")
    return "\n".join(lines)


def build_card(
    repo_id: str,
    manifest: Mapping[str, Any],
    title: str,
    summary: str,
    quality_md: str,
    speed_md: str,
    caveats: list[str],
    tags: list[str],
    usage: str,
) -> str:
    """YAML front matter + body. Every number in ``quality_md``/``speed_md`` comes from ``serve/results``."""
    base = manifest["base_model"]
    versions = ", ".join(f"{k} {v}" for k, v in manifest.get("versions", {}).items())
    calib = manifest.get("calib", "none")
    calib_line = (
        "none (data-free)"
        if calib == "none"
        else f"`{manifest['calib_repo']}` @ `{manifest['calib_revision'][:8]}`, {manifest['n']} samples × "
        f"{manifest['max_seq_length']} tokens (sample-id digest `{manifest['ids']}`)"
    )
    front = "\n".join(
        ["---", "license: apache-2.0", f"base_model: {base}", "base_model_relation: quantized", "tags:"]
        + [f"  - {t}" for t in tags]
        + ["---", ""]
    )
    body = (
        f"""# {title}

{summary}

Produced and evaluated in the [Cortex Serve]({REPO_URL}) project (milestone M3). Every number below is
reproducible from a committed config and manifest; full tables are in `serve/RESULTS.md` §M3.

## How it was made

| | |
|---|---|
| Base model | [`{base}`](https://huggingface.co/{base}) @ `{manifest["base_revision"][:8]}` |
| Method | {manifest["recipe_description"]} |
| Calibration | {calib_line} |
| Tools | {versions} |
| Hardware / time | {manifest.get("gpu", "?")}, {manifest.get("wall_s", 0):.0f} s |
| Size on disk | {manifest["size_bytes"] / 2**30:.2f} GiB |
| Source commit | [`{str(manifest["git_sha"])[:8]}`]({REPO_URL}/commit/{manifest["git_sha"]}) |

## Usage

{usage}

## Quality (same items for every variant; ± = 95% CI from lm-eval's standard error)

Served with vLLM 0.32.0, scored with lm-evaluation-harness 0.4.13 in the same container.

{quality_md}

## Speed (vLLM 0.32.0, Qwen chat workload 512 in / 256 out, closed-loop concurrency)

{speed_md}

## Caveats

"""
        + "\n".join(f"- {c}" for c in caveats)
        + f"""

## License

Apache-2.0, inherited from [`{base}`](https://huggingface.co/{base}).
"""
    )
    return front + body
