# M3 · Quantization artifacts + quality (E2, E3, QAT vs PTQ): Report

> **Status:** `DONE` except HF publishing (needs your yes and an HF token, D-022/D-027) · **Date:** 2026-11-24 · **Branch:** `serve/m3-quant` · **Modal spend:** ≈ $17 (credits; card $0.00)
> **Results:** [`../../RESULTS.md`](../../RESULTS.md) §M3 · **Research:** [`../research/M3_quantization-tooling.md`](../research/M3_quantization-tooling.md)

## 1. Summary of what was done

| Deliverable | Status | Evidence |
|---|---|---|
| Quantization app (llm-compressor 0.15.0, own image) + recipes: FP8-dynamic, W8A8-INT8, W4A16 GPTQ g128, AWQ, GPTQ/RTN g32 | Done | `deploy/modal/quantize_app.py`, `cortex_serve/quantize/` |
| Calibration builders: generic (UltraChat) and BFSI (eCFR Titles 12+17, part-level split) | Done | `cortex_serve/quantize/calib.py`, tests |
| 7 vLLM-format artifacts with manifests (recipe, calibration digest, versions, git SHA) | Done | `results/quantize/*/cortex_manifest.json` |
| GGUF track: bf16 conversion, imatrix (generic, BFSI), Q8_0/Q5_K_M/Q4_K_M ± imatrix | Done | `deploy/modal/gguf_app.py`, `results/quality-gguf-gpu/` |
| E3 quality: lm-eval (ARC, HellaSwag, MMLU-Pro, GSM8K, IFEval, WikiText) + BFSI held-out PPL, same items for every variant | Done | `deploy/modal/eval_app.py`, `results/quality-gpu/` |
| E2 speed: 6 formats on L4, 5 on exact H100 (vLLM); 5 GGUFs on L4 (llama.cpp) | Done | `results/e2-*` |
| Generic vs BFSI calibration ablation (GPTQ and imatrix) | Done | RESULTS §M3.2–3 |
| Cross-check vs Qwen's official FP8 | Done | RESULTS §M3.2, §M3.4 |
| QAT vs PTQ (Gemma 4 E4B: Google QAT W4A16 g32 vs our GPTQ and RTN at the same scheme) | Done | RESULTS §M3.6 |
| Gemma 4 E2B perplexity anomaly from M1 | Resolved | RESULTS §M3.5 |
| Publish artifacts to Hugging Face with model cards | ⏳ | Needs your HF token secret and a yes per publish |

## 2. Acceptance criteria (Phase 1 plan, M3)

| Criterion | Met? | Notes |
|---|---|---|
| All formats produced | ✅ | FP8, W8A8, W4A16 GPTQ/AWQ, GGUF Q8/Q5/Q4 ± imatrix, Gemma PTQ g32 |
| …and published | ⏳ | D-022: each publish asked separately |
| Format × eval quality table with deltas and CIs | ✅ | RESULTS §M3.2, §M3.3, §M3.6 |
| E2 curves | ✅ | RESULTS §M3.4, `results/e2-*` |
| Ablation result stated as measured | ✅ | Domain calibration helps in-domain PPL, hurts format-following |

## 3. Key numbers

| | Value |
|---|---|
| 8-bit quality cost (FP8 / INT8) | within CI on every task; PPL +0.1–1.3% |
| 4-bit quality cost (GPTQ / AWQ) | PPL +4.7–5.5%, MMLU-Pro −1.9 to −2.3 (inside CI) |
| L4: 4-bit vs bf16 | 2.7× faster per token at 1 user, 2.5× goodput at 4 users |
| H100: 4-bit vs bf16 at 64–128 users | slower (weight-only dequant becomes compute) |
| H100: FP8 vs bf16 goodput at 64 users | +30% |
| GGUF imatrix | −31% KL divergence at Q4_K_M, same speed |
| BFSI calibration | in-domain PPL damage +3.0% (vs +7.3%), GSM8K strict −38 (format), flexible −3 |
| Gemma 4 E4B, maths / instructions | GPTQ PTQ ≈ bf16 ≈ Google QAT; RTN −13.6 / −6.8 |

## 4. Deviations from plan
1. **AWQ ran on an L40S** (OOM on a 24 GB L4 with 512 × 2,048 calibration), keeping calibration identical to GPTQ (D-034).
2. **Gemma GPTQ needed llm-compressor's "basic" pipeline**; the default sequential tracer fails on Gemma 4 E4B ("No statistics available"). A data-free RTN baseline was added at the same scheme.
3. **Loglikelihood benchmarks are not usable for Gemma 4 IT** (ARC 25% = chance even for bf16; WikiText word PPL 252): its raw-text likelihoods are degenerate. Gemma is compared on generative tasks (GSM8K, IFEval, reasoning-first ARC with our own scorer) instead. Kept on record, not used for conclusions.
4. **MMLU-Pro uses the logprob leaderboard variant, first 1,000 items**; HellaSwag first 2,000 (cost).
5. **HF publishing not done yet** (needs your token and yes).

## 5. Decisions I made (veto any)

| | Decision | Why |
|---|---|---|
| ⚑ | Report GSM8K strict *and* flexible-extract | Strict alone turned a format change into a fake "maths collapse" |
| ⚑ | KL divergence (vs own parent) is the primary GGUF damage metric, PPL secondary | PPL and KLD disagreed on the imatrix; KLD is the direct measure |
| ⚑ | Gemma loglikelihood results set aside as not meaningful; no-BOS runs moved to `quality-gpu-invalid/` with a README | Chance-level even for bf16 |
| ⚑ | Reasoning-first ARC with our own scorer for chat-only models | lm-eval's `arc_challenge_chat` expects a bare letter; Gemma reasons first |
| | Same eval items for every variant (`--limit` = first N) | Paired comparison |
| | Artifacts stored on the Modal volume, not HF, until you approve publishing | D-022 |

## 6. Challenges & surprises
1. **"The best quantization" depends on the GPU.** 4-bit wins everywhere on the bandwidth-poor L4 and loses under load on the H100, where FP8's real 8-bit compute wins.
2. **A huge "GSM8K collapse" was a formatting change**: the BFSI-calibrated model stopped writing `#### N`. Looking at raw answers before believing a number saved a false headline.
3. **Gemma 4 IT models have degenerate raw-text likelihoods**, so standard multiple-choice benchmarks measure nothing on them. Even the missing `<bos>` turned out to be only part of the story.
4. **Careful PTQ (GPTQ) matched Google's QAT on chat tasks; naive RTN did not.** The QAT checkpoint's big perplexity advantage is a property of a differently trained model, not of 4-bit robustness.
5. **Tooling failures (each cheap, each fixed with a test or a guard):** a KLD reference file of tens of GB packed into results; multimodal processor files missing from artifacts; `polars` missing from the eval image; a deadline guard that skipped every single-config run; `pkill -f` matching the tool's own shell.

## 7. Explain-back checklist
1. How do GPTQ and AWQ differ? Why does AWQ need a quantizer after it?
2. Why are activation outliers the hard part of W8A8, and what does SmoothQuant move?
3. What does group size (32 vs 128) trade? Why does Machete not run Google's g32 checkpoint on H100?
4. FP8 E4M3: what are the bits, and why is "dynamic per-token" activation scaling data-free?
5. Why is W4A16 faster than bf16 on an L4 at 64 users but slower on an H100 at 128?
6. Why did BFSI-only calibration hurt GSM8K strict but barely flexible-extract?
7. What is an importance matrix, and why does it cut KLD without changing speed?
8. Why is KL divergence against the parent a better damage measure than perplexity?
9. QAT vs PTQ: what does QAT train, and why did it not beat GPTQ on chat tasks here?
10. Why are loglikelihood benchmarks meaningless for Gemma 4 IT?
11. Why did Qwen's official block-FP8 have worse TTFT on L4 but not on H100?

## 8. Next
M4 experiments are running (E4, E5, E6, E7, E9, E10, E12, E15). Publishing to HF waits for your token and yes.
