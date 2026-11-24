"""GPU-path pieces that can be tested without a GPU: commands, log parsers, telemetry parsing, budgets."""

from __future__ import annotations

import pytest
from cortex_serve.servebench.collect import summarize_gpu
from cortex_serve.servebench.config import CREDIT_CEILING_USD, ArtifactSpec, BudgetSpec, budget_check
from cortex_serve.servebench.engines import ProcessEngine, make_engine
from cortex_serve.servebench.engines.process import parse_sglang_memory, parse_vllm_memory, sglang_cmd, vllm_cmd

ART = ArtifactSpec(
    name="q", repo="Qwen/Qwen3-4B-Instruct-2507", revision="cdbee75f", family="Qwen/Qwen3-4B", quant="bf16"
)


def test_vllm_command() -> None:
    cmd = vllm_cmd(ART, {"max_model_len": 8192, "gpu_memory_utilization": 0.9, "enable_prefix_caching": True}, 8000)
    assert cmd[:3] == ["vllm", "serve", "Qwen/Qwen3-4B-Instruct-2507"]
    assert cmd[3:5] == ["--revision", "cdbee75f"]
    assert "--max-model-len" in cmd and cmd[cmd.index("--max-model-len") + 1] == "8192"
    assert "--enable-prefix-caching" in cmd and "True" not in cmd
    assert "--served-model-name" in cmd


def test_sglang_command() -> None:
    cmd = sglang_cmd(ART, {"mem_fraction_static": 0.85, "disable_radix_cache": True}, 30000)
    assert cmd[:3] == ["python3", "-m", "sglang.launch_server"]
    assert "--mem-fraction-static" in cmd and "--disable-radix-cache" in cmd and "--enable-metrics" in cmd


def test_process_llamacpp_command_uses_local_port() -> None:
    gguf = ART.model_copy(update={"file": "x.gguf", "repo": "org/r"})
    eng = make_engine("llamacpp", "img", {"threads": 8, "parallel": 4}, "/hf/gguf", launch="process")
    assert isinstance(eng, ProcessEngine)
    cmd = eng.run_args(gguf)
    assert cmd[0] == "/app/llama-server" and cmd[cmd.index("--port") + 1] == "8000"
    assert cmd[cmd.index("--host") + 1] == "127.0.0.1"
    assert cmd[cmd.index("-m") + 1] == "/hf/gguf/org_r/x.gguf"


def test_parse_vllm_log() -> None:
    log = (
        "INFO [gpu_worker.py:270] Available KV cache memory: 10.25 GiB\n"
        "INFO [kv_cache_utils.py:1200] GPU KV cache size: 74,640 tokens\n"
        "INFO [kv_cache_utils.py:1204] Maximum concurrency for 8,192 tokens per request: 9.11x\n"
        "INFO Model loading took 7.5123 GiB and 12.3 seconds\n"
    )
    assert parse_vllm_memory(log) == {
        "kv_available_gib": 10.25,
        "kv_capacity_tokens": 74640.0,
        "max_concurrency_ctx": 8192.0,
        "max_concurrency": 9.11,
        "model_load_gib": 7.5123,
    }


def test_parse_sglang_log() -> None:
    log = "KV Cache is allocated. #tokens: 74329, K size: 5.06 GB, V size: 5.06 GB\nmax_total_num_tokens=74329\n"
    mem = parse_sglang_memory(log)
    assert mem["kv_capacity_tokens"] == 74329 and mem["kv_gb"] == pytest.approx(10.12)


def test_gpu_telemetry_summary() -> None:
    s = summarize_gpu("35, 18000, 60.5\n95, 21000, 70.5\ngarbage\n")
    assert s == {"samples": 2, "gpu_util_mean_pct": 65.0, "gpu_mem_peak_mib": 21000.0, "gpu_power_mean_w": 65.5}
    assert summarize_gpu("") == {"samples": 0}


def test_budget_worst_case() -> None:
    assert BudgetSpec(gpu="L4", usd_per_hour=0.8, max_minutes=30).max_usd == pytest.approx(0.4)


def test_budget_check_gates_and_ceiling() -> None:
    assert budget_check(10.0, 5.0, 60.0) is None
    assert "stage gate" in (budget_check(58.0, 5.0, 60.0) or "")
    assert budget_check(58.0, 5.0, 120.0) is None
    assert "ceiling" in (budget_check(242.0, 5.0, CREDIT_CEILING_USD) or "")
