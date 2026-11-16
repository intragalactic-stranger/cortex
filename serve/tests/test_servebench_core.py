"""Metrics, workloads, engines, parsers."""

from __future__ import annotations

import random

import pytest
from cortex_serve.servebench.client import RequestResult
from cortex_serve.servebench.collect import parse_prometheus, summarize_samples
from cortex_serve.servebench.config import SLO, ArtifactSpec, WorkloadSpec
from cortex_serve.servebench.engines.llamacpp import parse_memory, server_args
from cortex_serve.servebench.engines.ollama import OllamaEngine
from cortex_serve.servebench.metrics import meets_slo, percentile, summarize
from cortex_serve.servebench.quality.arc import parse_answer, wilson
from cortex_serve.servebench.quality.perplexity import parse_ppl
from cortex_serve.servebench.tokenizer import WhitespaceTokenizer
from cortex_serve.servebench.workloads import exact_tokens, generate

SLO_ = SLO(ttft_s=1.0, tpot_s=0.1)


def _res(ttft: float, e2e: float, out: int, err: str | None = None) -> RequestResult:
    r = RequestResult(request_id=0, workload="w", input_tokens=100, max_tokens=out)
    r.ttft_s, r.e2e_s, r.output_tokens, r.error = ttft, e2e, out, err
    r.chunk_times_s = [ttft + i * (e2e - ttft) / max(out - 1, 1) for i in range(out)]
    return r


def test_percentile_matches_numpy_definition() -> None:
    xs = [1.0, 2.0, 3.0, 4.0]
    assert percentile(xs, 50) == 2.5
    assert percentile(xs, 95) == pytest.approx(3.85)
    assert percentile([], 50) is None


def test_goodput_counts_only_slo_meeting_requests() -> None:
    fast = _res(0.5, 0.5 + 9 * 0.05, 10)  # TTFT 0.5 s, TPOT 50 ms -> meets
    slow_ttft = _res(1.5, 2.0, 10)  # TTFT too slow
    slow_tpot = _res(0.5, 0.5 + 9 * 0.2, 10)  # TPOT 200 ms too slow
    failed = _res(0.1, 0.2, 10, err="HTTP 500")
    assert meets_slo(fast, SLO_) and not meets_slo(slow_ttft, SLO_) and not meets_slo(slow_tpot, SLO_)
    row = summarize([fast, slow_ttft, slow_tpot, failed], wall_s=10.0, slo=SLO_)
    assert row["errors"] == 1 and row["requests"] == 4
    assert row["slo_attainment"] == 0.25
    assert row["goodput_tok_s"] == 1.0  # 10 tokens / 10 s
    assert row["output_throughput_tok_s"] == 3.0  # 30 tokens from 3 OK requests / 10 s


def test_exact_token_lengths_and_determinism() -> None:
    tok = WhitespaceTokenizer()
    text = exact_tokens(tok, random.Random(0), 57)
    assert len(tok.encode(text)) == 57
    spec = WorkloadSpec(name="w", input_tokens=120, output_tokens=16, input_jitter=0.1)
    a = generate(spec, tok, 5, seed=7)
    b = generate(spec, tok, 5, seed=7)
    assert [r.prompt for r in a] == [r.prompt for r in b]
    assert all(108 <= r.input_tokens <= 132 for r in a)
    assert len({r.prompt for r in a}) == 5  # distinct prompts: no accidental prefix-cache hits


def test_shared_prefix_workload() -> None:
    tok = WhitespaceTokenizer()
    reqs = generate(WorkloadSpec(name="sp", input_tokens=200, output_tokens=8, shared_prefix_tokens=150), tok, 3, 1)
    prefixes = {" ".join(r.prompt.split()[:150]) for r in reqs}
    assert len(prefixes) == 1


def test_llamacpp_args() -> None:
    args = server_args(
        {"threads": 4, "parallel": 2, "ctx_per_slot": 2048, "ignore_eos": True, "extra": ["-v"]}, "a/b.gguf"
    )
    assert args[:2] == ["-m", "/models/a/b.gguf"]
    assert args[args.index("-t") : args.index("-t") + 2] == ["-t", "4"]
    assert "--ignore-eos" in args and "--metrics" in args and "-v" in args
    with pytest.raises(ValueError):
        server_args({"bogus": 1}, "x.gguf")


def test_ollama_modelfile() -> None:
    eng = OllamaEngine("ollama/ollama:x", {"threads": 4}, "/m")
    art = ArtifactSpec(name="a", repo="org/repo", revision="r", file="f.gguf", family="org/base", quant="q4")
    assert eng.modelfile(art) == "FROM /models/org_repo/f.gguf\nPARAMETER num_thread 4\n"


def test_parse_memory_uses_last_load_and_sums_swa() -> None:
    log = (
        "print_info: file size   = 100.00 MiB (4.00 BPW)\nllama_kv_cache: size =  999.00 MiB (  8 cells\n"
        "print_info: file size   = 372.65 MiB (5.24 BPW)\n"
        "llama_kv_cache: size =  128.00 MiB (  4096 cells, 4 layers)\n"
        "llama_kv_cache: size =   20.00 MiB (  512 cells, 20 layers)\n"
        "sched_reserve:        CPU compute buffer size =    30.02 MiB\n"
    )
    mem = parse_memory(log)
    assert mem["model_file_mib"] == 372.65 and mem["model_bpw"] == 5.24
    assert mem["kv_mib"] == 148.0 and mem["kv_caches"] == 2
    assert mem["compute_buffer_mib"] == 30.02


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("D. rift zones", "D"),
        ("Answer: A\n\nThe correct answer is", "A"),
        ("A, B, C, D.", None),
        ("A. B.", None),
        ("  \n C", "C"),
        ("I think", None),
    ],
)
def test_arc_parser(text: str, expected: str | None) -> None:
    assert parse_answer(text, ["A", "B", "C", "D"]) == expected


def test_wilson_interval() -> None:
    lo, hi = wilson(50, 100)
    assert lo == pytest.approx(0.404, abs=0.001) and hi == pytest.approx(0.596, abs=0.001)


def test_parse_ppl_and_prometheus_and_samples() -> None:
    assert parse_ppl("...\nFinal estimate: PPL = 19.8040 +/- 0.53\n") == (19.804, 0.53)
    metrics = parse_prometheus(
        "# HELP x\nllamacpp:requests_processing 2\nllamacpp:kv_cache_usage_ratio 0.25\nbad line x\n"
    )
    assert metrics == {"llamacpp:requests_processing": 2.0, "llamacpp:kv_cache_usage_ratio": 0.25}
    s = summarize_samples(
        [
            {"t": 0.0, "memory_bytes": 2**20, "cpu_usage_usec": 0.0},
            {"t": 2.0, "memory_bytes": 3 * 2**20, "cpu_usage_usec": 4e6},
        ]
    )
    assert s["peak_memory_mib"] == 3.0 and s["avg_cpu_cores"] == 2.0


def test_draft_model_requires_spec_type() -> None:
    # llama-server's --spec-type defaults to "none": -md alone loads the draft and never drafts (E6, 2026-11-16).
    with pytest.raises(ValueError, match="spec_type"):
        server_args({"draft_model": "d.gguf"}, "x.gguf")
    args = server_args({"draft_model": "d.gguf", "spec_type": "draft-simple"}, "x.gguf")
    assert args[args.index("-md") + 1] == "/models/d.gguf"
    assert args[args.index("--spec-type") + 1] == "draft-simple"


def test_draft_stats_sums_server_timings() -> None:
    from cortex_serve.servebench.runner import draft_stats

    rs = [RequestResult(request_id=i, workload="w", input_tokens=10, max_tokens=4) for i in range(3)]
    rs[0].server_timings = {"draft_n": 49.0, "draft_n_accepted": 46.0}
    rs[1].server_timings = {"draft_n": 51.0, "draft_n_accepted": 34.0}
    stats = draft_stats(rs)  # rs[2] has no timings (e.g. errored request)
    assert stats == {"draft_tokens": 100.0, "draft_accepted": 80.0, "draft_acceptance": 0.8}
    assert draft_stats(rs[2:])["draft_acceptance"] is None
