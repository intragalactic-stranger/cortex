# M3 research: quantization tooling, eval harness, GGUF, reference checkpoints (2026-11-24)

> **Method:** versions come from the PyPI JSON API, GitHub releases/tags (`gh api`), the GHCR registry API and the HF Hub API. Code facts were read from sources at the pinned tags: llm-compressor `0.15.0`, compressed-tensors `0.20.1`/`0.17.0`, vLLM `v0.32.0`, llama.cpp `b12110`, lm-eval `v0.4.13`. Nothing ran on a GPU. Anything not read from a primary source is marked **UNVERIFIED**.
> **Pin rule (same as M2):** pin releases that are at least about 2 weeks old.

---

## 0. Decisions this research supports

1. **Quantize in a separate Modal image.** Do not quantize inside the vLLM image: vLLM 0.32.0 pins `compressed-tensors==0.17.0` and llm-compressor 0.15.0 pins `==0.20.1`. These are hard `==` pins, so the two cannot be installed together (§1.4).
2. **Use an L4 for every quantization job.** Run GPTQ/AWQ on a 4B model with 512 samples on an L4 ($0.80/h). Expect under about $1 per artifact (estimate, §1.5).
3. **Two lm-eval model types, both against the vLLM server running in the same container:**
   - `local-completions` for loglikelihood tasks (HellaSwag, ARC-Challenge, `leaderboard_mmlu_pro`).
   - `local-chat-completions` for generative tasks (GSM8K, IFEval, the generative `mmlu_pro`).
   - **llama-server cannot return prompt logprobs**, so GGUF quality uses `llama-perplexity` instead: PPL, KLD vs bf16, and its built-in `--hellaswag`/`--multiple-choice` (§2, §3).
4. **Match Google's QAT scheme in the QAT-vs-PTQ study.** Google's Gemma 4 QAT W4A16 is **group_size 32, symmetric** (not 128). The PTQ baseline must use the same scheme for a like-for-like comparison (§4).
5. **On Gemma 4, use GPTQ, not stock AWQ.** llm-compressor has no AWQ (or SmoothQuant) mapping for `Gemma4ForConditionalGeneration`. It silently falls back to Llama-style mappings, which are wrong for Gemma's `pre_feedforward_layernorm` (§1.3d).

---

## 1. llm-compressor

### 1.1 Versions
| Package | Version | Released | Source |
|---|---|---|---|
| `llmcompressor` | **0.15.0** (latest stable) | PyPI 2026-10-28, GH release 2026-10-29 | PyPI JSON, `gh api …/releases` |
| `compressed-tensors` (required) | **==0.20.1** | 2026-10-28 | llmcompressor `requires_dist` |
| `torch` | `>=2.10.0,<=2.14.1` | | same |
| `transformers` | `>=5.15.0,<=5.21.0` | | same |
| `datasets` / `accelerate` / `auto-round` | `>=4.8.4` / `>=1.15.0` / `>=0.15.1` | | same |
| Python | `>=3.11` | | |

- Ignore the patch releases `0.7.1.5`, `0.9.0.6`, `0.10.0.5`, `0.12.0.3` and `0.14.0.1`. They are maintenance backports of old lines published in Oct–Nov, not newer features.
- 0.15.0 headline features: HIGGS mixed precision, block-wise QAD, layerwise offloading, MXFP6.

### 1.2 Import paths in 0.15.0 (changed vs older tutorials)
- `from llmcompressor import oneshot`
- `from llmcompressor.modifiers.quantization import QuantizationModifier`: RTN; used for FP8_DYNAMIC and as the quantizer after AWQ.
- `from llmcompressor.modifiers.gptq import GPTQModifier`
- `from llmcompressor.modifiers.transform.smoothquant import SmoothQuantModifier` (**moved** under `transform`)
- `from llmcompressor.modifiers.transform.awq import AWQModifier` (**moved**).
  - The old `llmcompressor.modifiers.awq.AWQModifier` is now a deprecated shim. It returns `[AWQ transform, QuantizationModifier]`.
  - **AWQ is native.** It is a transform that only applies scales, so it must be followed by a quantizer: `QuantizationModifier` (RTN) or `GPTQModifier`.

Preset schemes in compressed-tensors 0.20.1 (`quant_scheme.py`):

| Preset | Weights | Activations |
|---|---|---|
| `W4A16` | int4, group 128, symmetric | none |
| `W4A16_ASYM` | int4, group 128, asymmetric (zero point) | none |
| `W8A8` (alias `INT8`) | int8 per-channel, symmetric, static | int8 per-token, dynamic |
| `FP8_DYNAMIC` | fp8 per-channel | fp8 per-token, dynamic |

`GPTQModifier` defaults: `actorder="static"`, `block_size=128`, `dampening_frac=0.01`. Static actorder needs no runtime `g_idx`, so the checkpoint stays Machete/Marlin-friendly.

### 1.3 Minimal recipes (adapted from the `examples/` folder at tag 0.15.0)

Common setup. For BFSI calibration (§5), replace `ds` with the tokenized domain set. The prebaked alias `dataset="perfectblend", splits="train[:512]"` also works.
```python
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot

MODEL, REV = "Qwen/Qwen3-4B-Instruct-2507", "cdbee75f17c01a7cc42f958dc650907174af0554"
model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REV, dtype="auto")
tok = AutoTokenizer.from_pretrained(MODEL, revision=REV)
N, L = 512, 2048
ds = load_dataset("HuggingFaceH4/ultrachat_200k", split=f"train_sft[:{N}]").shuffle(seed=42)
ds = ds.map(lambda ex: {"text": tok.apply_chat_template(ex["messages"], tokenize=False)})
ds = ds.map(lambda s: tok(s["text"], max_length=L, truncation=True, add_special_tokens=False),
            remove_columns=ds.column_names)

def save(d):
    model.save_pretrained(d, save_compressed=True)
    tok.save_pretrained(d)
```

**(a) FP8 dynamic.** Data-free (no calibration set needed); takes minutes.
```python
from llmcompressor.modifiers.quantization import QuantizationModifier
oneshot(model=model, recipe=QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"]))
save("Qwen3-4B-Instruct-2507-FP8-Dynamic")
```

**(b) W8A8 INT8 (SmoothQuant + GPTQ).** Qwen3 has a SmoothQuant mapping (`Qwen3ForCausalLM → DEFAULT_SMOOTHQUANT_MAPPINGS`).
```python
from llmcompressor.modifiers.gptq import GPTQModifier
from llmcompressor.modifiers.transform.smoothquant import SmoothQuantModifier
recipe = [SmoothQuantModifier(smoothing_strength=0.8),
          GPTQModifier(targets="Linear", scheme="W8A8", ignore=["lm_head"])]
oneshot(model=model, dataset=ds, recipe=recipe, max_seq_length=L, num_calibration_samples=N)
save("Qwen3-4B-Instruct-2507-W8A8-INT8")
```

**(c) W4A16 GPTQ, group 128**
```python
from llmcompressor.modifiers.gptq import GPTQModifier
oneshot(model=model, dataset=ds, recipe=GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"]),
        max_seq_length=L, num_calibration_samples=N)
save("Qwen3-4B-Instruct-2507-W4A16-G128")
```

**(d) AWQ.** Use the `W4A16_ASYM` scheme from the upstream example, or `GPTQModifier` in place of `QuantizationModifier` for AWQ+GPTQ.
```python
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform.awq import AWQModifier
recipe = [AWQModifier(duo_scaling="both"),
          QuantizationModifier(targets=["Linear"], scheme="W4A16_ASYM", ignore=["lm_head"])]
oneshot(model=model, dataset=ds, recipe=recipe, max_seq_length=L, num_calibration_samples=N)
save("Qwen3-4B-Instruct-2507-AWQ-W4A16-ASYM")
```

**AWQ mapping coverage:**
- `Qwen3ForCausalLM` is in `AWQ_MAPPING_REGISTRY` (default Llama-style mappings), so the recipe above is correct for Qwen3.
- **Gemma 4 has no AWQ entry.** Only `Gemma2/Gemma3*` are registered. An unknown architecture logs "not found in mappings" and uses the defaults (`post_attention_layernorm → gate/up`). That is wrong for Gemma-style blocks, where `pre_feedforward_layernorm` feeds the MLP.
- If AWQ on Gemma 4 is wanted, pass explicit `mappings=` copied from `_gemma3_mappings`: `v_proj→o_proj`, `pre_feedforward_layernorm→gate/up`, `up_proj→down_proj`. Whether these fit Gemma 4's layer names (including KV-shared layers) is **UNVERIFIED**, so check with a dry run.

**Gemma 4 E4B PTQ to match Google's QAT format (W4A16, group 32, symmetric):**
```python
from transformers import AutoProcessor, Gemma4ForConditionalGeneration
from llmcompressor.utils import load_context
from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
with load_context(Gemma4ForConditionalGeneration):
    model = Gemma4ForConditionalGeneration.from_pretrained("google/gemma-4-E4B-it", revision="4ecc2ba63bcac23ba4a767fc87025b1450f84c9c")
scheme = QuantizationScheme(targets=["Linear"], weights=QuantizationArgs(
    num_bits=4, type="int", strategy="group", group_size=32, symmetric=True, dynamic=False))
recipe = GPTQModifier(config_groups={"group_0": scheme},
                      ignore=["lm_head", "re:.*vision.*", "re:.*audio.*"])  # same as Google's ignore list
```
- Google's ignore list is 250 entries: 249 vision/audio modules plus `lm_head`. Every text `Linear` is quantized, including the per-layer-input projections.
- The `QuantizationScheme`/`config_groups` constructor form follows the compressed-tensors API, but this exact snippet was not run (**UNVERIFIED**).

### 1.4 Co-installation with vLLM 0.32.0 is not possible
| | vLLM 0.32.0 (`requires_dist`) | llmcompressor 0.15.0 |
|---|---|---|
| compressed-tensors | **==0.17.0** | **==0.20.1** |
| torch | ==2.13.0 | 2.10–2.14.1 |
| transformers | >=5.16.1,<5.20.0 | 5.15.0–5.21.0 |

- Even vLLM 0.33/0.34 still pin `compressed-tensors==0.17.0`.
- No llmcompressor release pins 0.17.0 (0.12.0 pins 0.17.1).
- **Use a dedicated quantization image.** Suggested pins, all at least about 2 weeks old:
  ```python
  modal.Image.debian_slim(python_version="3.12").uv_pip_install(
      "llmcompressor==0.15.0", "compressed-tensors==0.20.1", "torch==2.14.1",
      "transformers==5.21.0", "datasets==5.1.0", "accelerate==1.15.0", "auto-round==0.16.1")
  ```
  Write the output to the `cortex-hf-cache` volume, or push it to a private HF repo, then load it from the vLLM image.
- **Format compatibility (checked by reading the schemas):** `QuantizationArgs` in ct 0.17.0 already has `scale_dtype`/`zp_dtype`, and the config models use pydantic's default (extra fields ignored). Checkpoints written by ct 0.20.1 should therefore load in vLLM 0.32.0. This is **UNVERIFIED at runtime**, so smoke-load each artifact once.

### 1.5 Memory and time for a 4B model with 512 samples (estimate)
- **Memory:**
  - llm-compressor onloads one decoder layer at a time ("sequential onloading") and keeps the rest of the model in CPU RAM.
  - GPTQ adds a Hessian about the size of the layer's weights (`docs/guides/memory.md`: Llama-3-8B GPTQ needs about 16 GB CPU and about 1 GB GPU for the layer plus Hessian).
  - Cached calibration activations, 512 × 2048 tokens × hidden size, are about 5.4 GB in bf16 for Qwen3-4B (hidden 2560) and also for Gemma 4 E4B (hidden 2560). Where they live depends on the offload settings.
  - **An L4 (24 GB) is enough.** Request about **32 GB container RAM**. Gemma 4 E4B is 16 GB of bf16 weights including the Per-Layer Embeddings (PLE).
- **Time:**
  - `docs/guides/runtime.md` gives, for Llama-3-8B on an A100: GPTQ ~30 min, AWQ ~30 min, RTN weights-only ~1 min, RTN W+A ~20 min.
  - A 4B model on an L4 should land around **15–40 min for GPTQ or AWQ**, and **under 5 min for FP8_DYNAMIC**.
  - Set the Modal timeout to 60 min. Worst case is $0.80 on an L4. These figures are **estimates, not measured**.

---

## 2. lm-evaluation-harness

| | |
|---|---|
| Version | **`lm_eval` 0.4.13** (PyPI and GH release 2026-08-31). Extras needed: `lm_eval[api,ifeval]==0.4.13` (+`math` only if using the math tasks) |
| CLI | `lm-eval run …` (subcommands `run`/`ls`/`validate`). The legacy `lm_eval --model …` still works because it auto-inserts `run` |
| Deps | No torch. `transformers` is only needed for `tokenizer_backend=huggingface`, and the vLLM image already has it. Safe to `uv_pip_install` into the vLLM image, but confirm torch and transformers versions are unchanged after install |

**Model types (README table + `docs/API_guide.md`):**

| Model type | Endpoint | Request types |
|---|---|---|
| `local-completions` | `/v1/completions` | `generate_until`, `loglikelihood`, `loglikelihood_rolling` |
| `local-chat-completions` | `/v1/chat/completions` | `generate_until` only ("loglikelihood and MCQ tasks are only supported for completion endpoints") |

- For loglikelihood, `local-completions` sends `{"prompt": tokens, "max_tokens": 1, "logprobs": 1, "echo": true}`. It needs `tokenizer_backend=huggingface` (the default) with the model's tokenizer.
- **vLLM 0.32.0 supports this.** `CompletionRequest` has `echo` and `logprobs`. With `echo=True`, `prompt_logprobs` defaults to `logprobs` (`vllm/entrypoints/openai/completion/protocol.py` L373–375). Server cap: `--max-logprobs` defaults to 20, and 1 is needed.
- **llama-server (b12110) does not support this.** It has no `echo` or prompt logprobs; `n_probs` covers generated tokens only. GGUF therefore gets only generative tasks through lm-eval. For MC/PPL, use `llama-perplexity` with `--kl-divergence-base`/`--kl-divergence`, `--hellaswag`, `--winogrande`, `--multiple-choice` (all flags present in `common/arg.cpp` at b12110).

**Tasks (YAMLs at v0.4.13):**

| Benchmark | Task name | Output type | Size | Endpoint |
|---|---|---|---|---|
| MMLU-Pro | `mmlu_pro` (group of 14 `mmlu_pro_<subject>`) | **generate_until**: 5-shot CoT, regex `answer is (X)`, `max_gen_toks 2048` | 12,032 test | chat or completions |
| MMLU-Pro (logprob variant) | `leaderboard_mmlu_pro` | multiple_choice (10 options, 5-shot) | 12,032 | completions + logprobs |
| GSM8K | `gsm8k` (5-shot); also `gsm8k_cot`, `gsm8k_cot_zeroshot` | generate_until | 1,319 test | chat (with `--apply_chat_template`) |
| IFEval | `ifeval` | generate_until, 0-shot, `max_gen_toks 1280` | 541 | chat |
| ARC-Challenge | `arc_challenge` (MC), `arc_challenge_chat` (generative) | multiple_choice / generate_until | 1,172 test | completions / chat |
| HellaSwag | `hellaswag` | multiple_choice (acc, acc_norm) | 10,042 val | completions + logprobs |

Note: the default `mmlu_pro` is **generative** and not loglikelihood-based, with up to 2,048 generated tokens per question. That makes it the most expensive task. Use `leaderboard_mmlu_pro` for a cheap logprob version.

```bash
# loglikelihood tasks against vLLM (same container)
lm-eval run --model local-completions \
  --model_args model=<served-name>,base_url=http://127.0.0.1:8000/v1/completions,tokenizer=Qwen/Qwen3-4B-Instruct-2507,tokenized_requests=True,num_concurrent=16,max_retries=3 \
  --tasks hellaswag,arc_challenge,leaderboard_mmlu_pro --limit 500 --seed 1234 --output_path out/ --log_samples
# generative tasks
lm-eval run --model local-chat-completions \
  --model_args model=<served-name>,base_url=http://127.0.0.1:8000/v1/chat/completions,num_concurrent=16 \
  --tasks gsm8k,ifeval --apply_chat_template --limit 500 --output_path out/ --log_samples
```

- **`--limit`:**
  - An integer is a count and a float in (0,1) is a fraction. It applies per task, so for the `mmlu_pro` group it is per subject: `--limit 50` gives about 700 questions.
  - Docs mark it "for testing only". For reportable subsets, use a fixed `--samples '{"task":[idx…]}'` JSON instead (incompatible with `--limit`), so every quant variant sees the same items.
- **Thinking:**
  - vLLM 0.32.0 has `--default-chat-template-kwargs` (`vllm/entrypoints/launchers/cli_args.py`). Use `'{"enable_thinking": false}'` for Gemma 4.
  - Qwen3-4B-Instruct-2507 is non-thinking only.
  - Whether Gemma 4's template uses exactly that key is **UNVERIFIED**; M2 recorded `enable_thinking` for all engines.
  - lm-eval: `enable_thinking=True` is only valid for generative tasks.

---

## 3. GGUF toolchain at b12110

| | |
|---|---|
| Release | `b12110` published 2026-11-04, commit `cd959bae99ff4bed22bef2604f45241c09bbb8c8` |
| Full image (has convert + quantize + imatrix + perplexity) | `ghcr.io/ggml-org/llama.cpp:full-cuda-b12110@sha256:87803fec70c8046e04e051e49199894ad77e61cf74fd1dceb9ad5ee2f75de228` |
| Server image (pinned in M2, digest re-confirmed) | `…:server-cuda-b12110@sha256:a2963b4817f03eea7e0d6065ea576252322792441ef97aaae178b924520547d6` |
| (light) | `…:light-cuda-b12110@sha256:464290dba73fbed47af9332fcde80356e4739206ade2861f7a7764bfd49d8522` |

- **Contents of `full-cuda` (`.devops/cuda.Dockerfile`):**
  - Base: CUDA 12.8.1, Ubuntu 24.04.
  - `/app` holds all of `build/bin/*`, so `llama-quantize`, `llama-imatrix`, `llama-perplexity` and `llama-server` are all there.
  - Also present: `convert_hf_to_gguf.py`, `conversion/` and `gguf-py/`, with `requirements.txt` pip-installed: `torch==2.11.0` (CPU), **`transformers==4.57.6`**.
  - Entrypoint is `/app/tools.sh` (`--convert|--quantize|--perplexity|--server|…`). For Modal, set `.entrypoint([])` and call the binaries directly.
- **Gemma 4 conversion needs transformers 5.** The quantize README says so: "more and more models (like Gemma 4) require transformers 5. You can safely `pip install -U transformers`". Add a layer to the image, e.g. `transformers==5.21.0`. Qwen3 converts with 4.57.6.
- **Commands (flags read from b12110 sources):**
  ```bash
  python3 /app/convert_hf_to_gguf.py /hf/<snapshot> --outtype bf16 --outfile model-bf16.gguf
  #   --outtype {f32,f16,bf16,q8_0,tq1_0,tq2_0,auto}; auto keeps FP8 weights / else best 16-bit. Also --remote, --mmproj, --fp8-as-q8
  /app/llama-imatrix -m model-bf16.gguf -f calib.txt -o imatrix.gguf -ngl 99 [--chunks N] [--no-ppl] [--parse-special]
  #   default output is now GGUF format (imatrix.gguf); legacy via --output-format dat
  /app/llama-quantize --imatrix imatrix.gguf model-bf16.gguf model-Q4_K_M.gguf Q4_K_M
  #   also: --pure, --leave-output-tensor, --output-tensor-type, --token-embedding-type, --tensor-type <regex>=<type>, --include/--exclude-weights
  /app/llama-perplexity -m model-bf16.gguf -f wiki.test.raw --kl-divergence-base base.kld -ngl 99
  /app/llama-perplexity -m model-Q4_K_M.gguf --kl-divergence-base base.kld --kl-divergence -ngl 99
  ```
- **Changes vs older guides:**
  - The imatrix default output is `imatrix.gguf` (GGUF format), no longer `.dat`.
  - `llama-cli` is now the chat CLI; the old completion CLI is `llama-completion`.
  - The `--imatrix`, `--pure` and `--tensor-type` flag names are unchanged.

---

## 4. Reference checkpoints (HF API, 2026-11-24)

| Repo | Commit SHA | Gated | License | Notes |
|---|---|---|---|---|
| `Qwen/Qwen3-4B-Instruct-2507` | `cdbee75f17c01a7cc42f958dc650907174af0554` | no | Apache-2.0 | matches the project pin |
| `Qwen/Qwen3-4B-Instruct-2507-FP8` | `8591804019c8b22094c3b5b4454e0edc05dffc98` | no | Apache-2.0 | **Not compressed-tensors.** Native `quant_method: fp8`, e4m3, **block 128×128**, dynamic activations. 5.19 GB |
| `google/gemma-4-E4B-it` (bf16) | `4ecc2ba63bcac23ba4a767fc87025b1450f84c9c` | no | Apache-2.0 | 15.99 GB (PLE + vision/audio) |
| `google/gemma-4-E4B-it-qat-w4a16-ct` | `6cd26aaa2357fb2bad8c51699a7558a4d1a965bb` | no | Apache-2.0 | compressed-tensors `pack-quantized`, int4, **group 32, symmetric**, weights only. Ignores only vision/audio + `lm_head`; embeddings stay bf16, which is why it is 11.51 GB. `base_model`: `…-qat-q4_0-unquantized` |
| `google/gemma-4-E4B-it-qat-q4_0-gguf` | `4b4a2c1d584be7264f87aac328a1bc739ce81b6c` | no | Apache-2.0 | `gemma-4-E4B_q4_0-it.gguf` (5.15 GB) + `gemma-4-E4B-it-mmproj.gguf` |
| `google/gemma-4-E4B-it-qat-q4_0-unquantized` | `476025a01dbf99361c062bbeca3d6a76bb4c4566` | no | Apache-2.0 | bf16 weights from the QAT run |
| **`google/gemma-4-E2B-it-qat-q4_0-unquantized`** | `6befbaca7398925921802abd1f277b495b78b738` | no | Apache-2.0 | **bf16** (config dtype bfloat16, no quant config), 10.21 GB, `base_model: google/gemma-4-E2B-it` |
| `google/gemma-4-E2B-it-qat-q4_0-gguf` | `675cff42a74c774d6cb76f76d8eacb49b48c9b93` | no | Apache-2.0 | `gemma-4-E2B_q4_0-it.gguf` (3.35 GB) |
| `google/gemma-4-E2B-it-qat-w4a16-ct` | `971342c08f607aa7779983f6b5289778b5d271a7` | no | Apache-2.0 | |
| `google/gemma-4-E2B-it` (bf16) | `4adf15748de9e1a0097b6274cee9c15836c41c37` | no | Apache-2.0 | |
| `unsloth/gemma-4-E2B-it-GGUF` (for the anomaly) | `0314792d7f1f7e229411f620751375812bb9faf2` | no | | has BF16/Q8_0/Q4_0 |

Other QAT variants not used here: `…-qat-mobile-ct` (wNa8o8 mobile schema), `…-qat-mobile-transformers`, and `…-assistant` drafters.

**Resolving the E2B perplexity anomaly** (QAT Q4_0 PPL 65 vs unsloth Q8_0 PPL 212):
- **Hypothesis:** the gap comes from the **QAT training run** changing the model's text distribution, not from quantization. The Q8_0 is effectively the original IT model, and IT models score poorly on raw WikiText.
- Google's card and launch blog do not describe the QAT data or recipe (**UNVERIFIED**).
- **Test, all on b12110:** convert `google/gemma-4-E2B-it-qat-q4_0-unquantized` to bf16 GGUF, then run `llama-perplexity` on four models:
  1. QAT-bf16
  2. QAT-Q4_0 (Google)
  3. IT-bf16 (`google/gemma-4-E2B-it`, or unsloth BF16)
  4. unsloth Q8_0

  Read the results as follows:
  - **(1)≈(2)** isolates the 4-bit error of the QAT model.
  - **(3)≈(4)** isolates the Q8_0 error.
  - **(1) vs (3)** is the effect of QAT training itself.
- **Also rule out tokenization:** Gemma needs BOS; check `tokenizer.ggml.add_bos_token` in both GGUFs.
- **The fair metric** for "quantization damage" is KLD of each quant vs **its own** bf16 parent, not raw PPL across different parents.

---

## 5. BFSI / financial-regulation calibration and held-out text

| Candidate | License | Size | Gated | Fit |
|---|---|---|---|---|
| **`Azzindani/US_Regulation_ECFR_20260101`** (sha `04148fee3c9832a2789a61ee4cdcd38063ec08d7`) | CC-BY-4.0 on the curation; the eCFR text itself is U.S. public domain | 61,482 sections, 240 MB, parquet. Title 12 "Banks and Banking" has **1,120** sections; Title 17 "Commodity and Securities Exchanges" has **757**. `section_text` median 846 chars, mean 1,915 | no | Directly on-domain (OCC/Fed/FDIC/SEC/CFTC rules). Filter on `title_number in {"12","17"}`. Rich metadata (`rag_id`, `part_name`) supports a part-level split with no leakage |
| `chenjigaram/eu-financial-regulation-aligned` (sha `d1fb306a171376c05914f52819c49e0d79d5f2d0`) | CC-BY-4.0 (EUR-Lex, Decision 2011/833/EU) | 56,838 rows = 2,442 provisions × 24 languages. **English: 2,442 paragraphs**, median 456 chars. Covers MiFID II, MiFIR, MiCA, DORA, UCITS, AIFMD, PRIIPs, SFDR | no | Good **held-out, different-jurisdiction** domain set. Paragraphs are short, so concatenate them per article for PPL |
| `c3po-ai/edgar-corpus` (sha `5b6b152615afae7042778a79d25aa810091a74c7`), the parquet mirror of `eloukas/edgar-corpus` | Apache-2.0 | 10-K sections 1993–2020 (`section_1A` risk factors, `section_7` MD&A, …), "full" config with 47k train rows | no | Large, well-cited (EDGAR-CORPUS paper). Corporate disclosure rather than regulation. **The original `eloukas/edgar-corpus` is script-based and will not load with `datasets>=4`** |

Rejected:
- `alea-institute/kl3m-data-ecfr`: CC-BY-4.0, but stored **pre-tokenized** with the KL3M tokenizer.
- `wesslen/ecfr-title-12`: no license tag.
- `gtfintechlab/fomc_communication`, `TheFinAI/en-fomc`, `GadgetSoftware/federal-register`: all CC-BY-NC.

**Recommendation:**
- **Calibration:** eCFR Titles 12+17 from `Azzindani/US_Regulation_ECFR_20260101`, pinned by sha. Split by CFR **part** (e.g. a hash of `title_number:part_number`):
  - about 512 sections for calibration (truncate or pack to 2,048 tokens);
  - a disjoint held-out set of about 200–300 sections for domain PPL.
- **Second held-out set:** the English slice of the EU set. It tests whether calibrating on US rules over-fits.
- **Caveat:** the eCFR repo is a single-maintainer upload. Its section counts look like a subset of the full eCFR (e.g. Title 26 has only 271 sections), so completeness is **UNVERIFIED**. That is fine for calibration, but record the sha and row IDs.

---

## 6. vLLM 0.32.0 kernel support: L4 (sm_89) vs H100 (sm_90)

Read from `vllm/model_executor/kernels/linear/__init__.py` (priority lists), each kernel's `is_supported`/`can_implement`, and the compressed-tensors schemes.

| Checkpoint | Scheme min. capability | L4 (sm_89) | H100 (sm_90) |
|---|---|---|---|
| GPTQ W4A16 g128, sym (ct `wNa16`) | 75 | **Marlin** (Machete and CUTLASS W4A8 need sm90) | **Machete** (group sizes −1/64/128 with bf16 activations); otherwise Marlin |
| AWQ W4A16_ASYM g128 (ct `wNa16` + zero point) | 75 | **Marlin** (supports zero points) | Machete if it accepts uint4+zp for this config, else Marlin (**UNVERIFIED** which one is chosen) |
| Google QAT W4A16 **g32** (ct) | 75 | **Marlin** (supports −1/32/64/128) | **Marlin**: Machete rejects group 32 |
| W8A8 INT8 (ct) | 75 | **CUTLASS int8** (`scaled_mm_c2x_sm89_int8`) | CUTLASS int8 (c3x sm90) |
| FP8_DYNAMIC (ct, per-channel W / per-token A) | **89** | **CUTLASS FP8** (`c2x_sm89_fp8`; needs CUDA ≥12.4). FlashInfer FP8 needs sm100 | CUTLASS FP8 (c3x sm90) |
| Qwen official FP8 (native `fp8`, **block 128×128**) | | **No CUTLASS block-FP8 on sm89** (`cutlass_scaled_mm_supports_block_fp8` requires sm90+); DeepGEMM also needs sm90. Fallback order: Marlin FP8 (**weight-only W8A16**) → Triton block FP8 | DeepGEMM / CUTLASS block FP8 |

**Implications:**
- All four llm-compressor formats serve on L4.
- On L4, **our FP8_DYNAMIC runs true W8A8 on the CUTLASS path**, while **Qwen's official block-FP8 probably runs as Marlin W8A16 or Triton**. Their throughput is not directly comparable on L4. Record the kernel vLLM picks from the startup log for every run; the exact log text is **UNVERIFIED**.
- A10G (sm_86) cannot run FP8 W8A8. It falls back to Marlin FP8 weight-only.
- Gemma 4 is registered in vLLM 0.32.0 (`Gemma4ForConditionalGeneration` → `gemma4_mm`). Google states the `-qat-w4a16-ct` checkpoints are "for native, optimized inference with vLLM".

---

## Risks / unknowns
1. **compressed-tensors version skew** (0.20.1 writer vs 0.17.0 reader in vLLM 0.32.0). The schema looks compatible, but smoke-load each artifact in vLLM 0.32.0 before benchmarking (cheap L4 run).
2. **Gemma 4 + llm-compressor is untested for E-series.**
   - The upstream examples cover 31B and 26B-A4B only.
   - PLE embeddings and KV-shared layers may interact badly with the sequential pipeline's layer tracing.
   - AWQ and SmoothQuant have no Gemma 4 mappings.
   - Plan a 32-sample dry run first.
3. **Gemma 4 E4B bf16 on L4:** 16 GB of weights in 24 GB leaves little KV room. Disable the multimodal encoders (`--limit-mm-per-prompt`) and keep `max_model_len` small, or use an L40S. Fit is **UNVERIFIED**.
4. **The L4 kernel choice for Qwen's block-FP8 checkpoint is UNVERIFIED** (Marlin vs Triton). Read the vLLM log.
5. **The GGUF loglikelihood gap:** llama-server has no prompt logprobs, so MC accuracy for GGUF comes from `llama-perplexity` built-ins, not lm-eval. The numbers are not strictly comparable with lm-eval's `hellaswag`. For a like-for-like cross-engine metric, prefer KLD/PPL plus generative tasks.
6. **lm-eval in the vLLM image:** installing `lm_eval[api,ifeval]` must not upgrade torch or transformers. Check with `uv pip install --dry-run`, or use a constraints file.
7. **`--limit` subsets are not stratified,** and in MMLU-Pro they are per subject. Use fixed `--samples` index files for reportable numbers.
8. **The E2B PPL anomaly cause is a hypothesis** (QAT training shifts the distribution) until the four-way experiment in §4 runs.
9. **The BFSI eCFR dataset** is a single-maintainer upload whose completeness is unverified. Pin the sha, and consider a fallback of fetching Title 12 XML from govinfo/eCFR ourselves (public domain).
10. **The quantization-time and memory figures in §1.5 are estimates** scaled from upstream A100/8B docs, not measurements.
