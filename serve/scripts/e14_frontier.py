"""E14: cost frontier. $ per 1M SLO-meeting output tokens vs quality, for every measured serving configuration.

    uv run python serve/scripts/e14_frontier.py      # -> results/reports/m7-e14/{frontier.csv,frontier.png}
                                                     #    results/web/{configs,curves,quality}.json (site data)

Cost = hourly price of the hardware ÷ the best SLO-meeting output throughput (goodput tok/s) over the measured load
levels, so it is the price of tokens that met the latency target, at the best load we measured. Quality = mean of
GSM8K (5-shot, flexible extract) and IFEval (prompt-level strict): the two generative tasks that are valid for every
model family here (Gemma 4's loglikelihood benchmarks are not, RESULTS §M3.6).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from quality_gpu_table import load

RES = Path("serve/results")
OUT = RES / "reports" / "m7-e14"
WEB = RES / "web"
CPU_USD_PER_HOUR = 0.1504  # AWS t3a.xlarge on-demand, us-east-1 (list price; the M1 box)
GPU_USD_PER_HOUR = {"L4": 0.80, "L40S": 1.95, "H100": 3.95, "H200": 3.95, "H100x2": 7.90}  # H200 billed as H100

# artifact name in run dirs -> quality-gpu result name
QUALITY_OF = {
    "qwen3-4b-bf16": "qwen3-4b-bf16",
    "qwen3-4b-f16-gguf": "qwen3-4b-bf16",
    "qwen3-4b-fp8-dynamic": "qwen3-4b-fp8-dynamic",
    "qwen3-4b-fp8-official": "qwen3-4b-fp8-official",
    "qwen3-4b-w8a8-int8": "qwen3-4b-w8a8-int8-generic",
    "qwen3-4b-w4a16-gptq-g128": "qwen3-4b-w4a16-gptq-g128-generic",
    "qwen3-4b-w4a16-awq-asym": "qwen3-4b-w4a16-awq-asym-generic",
    "qwen3-30b-a3b-bf16": "qwen3-30b-a3b-bf16",
    "qwen3.5-4b-bf16": "qwen3.5-4b-bf16",
    "gemma4-e4b-bf16": "gemma4-e4b-bf16",
}
OFFSETS = {  # label positions (points) so the dense clusters stay readable
    "qwen3-4b-fp8-official|vllm|H100": (6, 22),
    "qwen3-4b-bf16|vllm|H100": (-10, -42),
    "qwen3-4b-w4a16-gptq-g128|vllm|H100": (10, -26),
    "qwen3-4b-bf16|vllm|H100x2": (8, 24),
    "qwen3-4b-w4a16-gptq-g128|vllm|L4": (-60, -34),
    "qwen3-4b-bf16|vllm|L4": (-20, -42),
    "qwen3.5-4b-bf16|vllm|L4": (-30, 26),
    "gemma4-e4b-bf16|vllm|L4": (-40, -40),
    "qwen3-4b-f16-gguf|llamacpp|L4": (-60, 30),
    "qwen3-30b-a3b-bf16|vllm|H100": (8, 6),
}
SKIP_RUNS = ("e8-", "e4-", "e5-", "e6-", "e7-", "e10-", "smoke", "e6-cpu", "e13-llama-bench", "quality")


def _gpu_label(manifest: dict[str, object]) -> str:
    host = manifest.get("host", {})
    gpu = str(host.get("gpu", "")) if isinstance(host, dict) else ""
    if not gpu or gpu == "None":
        return "CPU"
    n = gpu.count("NVIDIA")
    name = next((g for g in ("H200", "H100", "L40S", "L4") if g in gpu), gpu.split(",")[0])
    return f"{name}x{n}" if n > 1 else name


def collect() -> pl.DataFrame:
    rows = []
    for man_path in sorted(RES.glob("*/*/manifest.json")):
        run = man_path.parent.parent.name
        if any(run.startswith(s) for s in SKIP_RUNS) or (man_path.parent / "SUPERSEDED.md").exists():
            continue
        if (man_path.parent / "INVALID.md").exists() or not (man_path.parent / "points.parquet").exists():
            continue
        manifest = json.loads(man_path.read_text())
        gpu = _gpu_label(manifest)
        price = CPU_USD_PER_HOUR if gpu == "CPU" else GPU_USD_PER_HOUR.get(gpu)
        if price is None:
            continue
        pts = pl.read_parquet(man_path.parent / "points.parquet")
        wl = "chat-balanced" if gpu != "CPU" else "chat-short"
        pts = pts.filter(pl.col("workload") == wl)
        if pts.is_empty():
            continue
        for (engine, artifact), g in pts.group_by(["engine", "artifact"]):
            per_level = g.group_by("level").agg(
                pl.col("goodput_tok_s").median(), pl.col("output_throughput_tok_s").median(),
                pl.col("tpot_p50_s").median(), pl.col("ttft_p95_s").median(),
            ).sort("level")  # fmt: skip
            best = per_level.sort("goodput_tok_s", descending=True).row(0, named=True)
            rows.append({
                "run": run, "stamp": man_path.parent.name, "engine": engine, "artifact": artifact, "gpu": gpu,
                "usd_per_hour": price, "workload": wl, "best_level": best["level"],
                "goodput_tok_s": best["goodput_tok_s"], "throughput_tok_s": best["output_throughput_tok_s"],
                "tpot_p50_ms_1user": float(per_level["tpot_p50_s"][0]) * 1000,
                "curve": per_level.to_dicts(),
            })  # fmt: skip
    return pl.DataFrame(rows)


def quality(name: str) -> dict[str, float]:
    q = load(name)
    out = {k: v[0] for k, v in q.items()}
    if "gsm8k_flex" in out and "ifeval" in out:
        out["composite"] = (out["gsm8k_flex"] + out["ifeval"]) / 2
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    WEB.mkdir(parents=True, exist_ok=True)
    df = collect()
    qual = {a: quality(n) for a, n in QUALITY_OF.items() if (Path("serve/results/quality-gpu") / n).exists()}
    df = df.with_columns(
        (pl.col("usd_per_hour") / (pl.col("goodput_tok_s") * 3600) * 1e6).alias("usd_per_1m_slo_tokens"),
        pl.col("artifact").map_elements(lambda a: qual.get(a, {}).get("composite"), return_dtype=pl.Float64)
        .alias("quality"),
    )  # fmt: skip
    # keep the best measured run per (engine, artifact, gpu): experiments sampled different load levels
    df = df.sort("goodput_tok_s").group_by(["engine", "artifact", "gpu"]).last()
    ok = df.filter(pl.col("goodput_tok_s") > 0)
    table = ok.select("gpu", "engine", "artifact", "best_level", "goodput_tok_s", "usd_per_1m_slo_tokens", "quality",
                      "tpot_p50_ms_1user", "run", "stamp").sort("usd_per_1m_slo_tokens")  # fmt: skip
    table.write_csv(OUT / "frontier.csv")

    # Pareto frontier: lower cost and higher quality (configs with a quality score)
    scored = table.filter(pl.col("quality").is_not_null()).sort("usd_per_1m_slo_tokens")
    frontier, best_q = [], -1.0
    for r in scored.iter_rows(named=True):
        if r["quality"] > best_q:
            frontier.append(r)
            best_q = r["quality"]
    fig, ax = plt.subplots(figsize=(11, 6.2))
    colors = {"L4": "#2f6fd0", "H100": "#c2410c", "H200": "#7c3aed", "H100x2": "#b45309", "L40S": "#0f766e"}
    label_these = {"qwen3-4b-fp8-official|vllm|H100", "qwen3-4b-bf16|vllm|H100", "qwen3-4b-w4a16-gptq-g128|vllm|H100",
                   "qwen3-4b-bf16|vllm|H100x2", "qwen3-30b-a3b-bf16|vllm|H100", "qwen3-4b-w4a16-gptq-g128|vllm|L4",
                   "qwen3-4b-bf16|vllm|L4", "qwen3.5-4b-bf16|vllm|L4", "gemma4-e4b-bf16|vllm|L4",
                   "qwen3-4b-f16-gguf|llamacpp|L4"}  # fmt: skip
    short = {
        "qwen3-4b-": "4B ",
        "qwen3-30b-a3b-": "30B-A3B MoE ",
        "qwen3.5-4b-": "Qwen3.5-4B ",
        "gemma4-e4b-": "Gemma4-E4B ",
    }
    for r in scored.iter_rows(named=True):
        ax.scatter(r["usd_per_1m_slo_tokens"], r["quality"] * 100, color=colors.get(r["gpu"], "k"), s=46, zorder=3)
        key = f"{r['artifact']}|{r['engine']}|{r['gpu']}"
        if key in label_these:
            name = r["artifact"]
            for a, b in short.items():
                name = name.replace(a, b)
            ax.annotate(f'{name} · {r["engine"]} · {r["gpu"]}', (r["usd_per_1m_slo_tokens"], r["quality"] * 100),
                        fontsize=7.5, xytext=OFFSETS.get(key, (5, 4)), textcoords="offset points",
                        arrowprops={"arrowstyle": "-", "lw": 0.5, "color": "#888"})  # fmt: skip
    ax.plot([r["usd_per_1m_slo_tokens"] for r in frontier], [r["quality"] * 100 for r in frontier], "--", color="#16a34a",
            label="Pareto frontier", zorder=2)  # fmt: skip
    ax.axhspan(78.7, 83.5, color="#94a3b8", alpha=0.12, zorder=0, label="±2.4-point CI band around 4B bf16")
    for g, c in colors.items():
        if g in set(scored["gpu"].to_list()):
            ax.scatter([], [], color=c, label=g)
    ax.set_xscale("log")
    ax.set_ylim(76, 88)
    ax.set_xlabel("$ per 1M output tokens that met the latency SLO (log scale; best measured load)")
    ax.set_ylabel("quality: mean of GSM8K (flexible) and IFEval, %")
    ax.set_title("E14 cost frontier: price of SLO-meeting tokens vs quality (chat 512 in / 256 out)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0)
    fig.tight_layout()
    fig.savefig(OUT / "frontier.png", dpi=130)

    # site data
    (WEB / "configs.json").write_text(json.dumps(table.to_dicts(), indent=1))
    (WEB / "frontier.json").write_text(
        json.dumps([r["artifact"] + "|" + r["engine"] + "|" + r["gpu"] for r in frontier])
    )
    (WEB / "curves.json").write_text(json.dumps(
        [{k: r[k] for k in ("run", "engine", "artifact", "gpu", "workload", "curve")} for r in df.iter_rows(named=True)],
        indent=1))  # fmt: skip
    (WEB / "quality.json").write_text(json.dumps({n: quality(n) for n in sorted(
        p.name for p in Path("serve/results/quality-gpu").iterdir() if p.is_dir())}, indent=1))  # fmt: skip
    print(table.select("gpu", "engine", "artifact", "best_level", "goodput_tok_s", "usd_per_1m_slo_tokens", "quality"))
    print("frontier:", [f"{r['artifact']}|{r['engine']}|{r['gpu']}" for r in frontier])


if __name__ == "__main__":
    main()
