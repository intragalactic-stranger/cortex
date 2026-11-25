// Display names for ids that appear in the results JSON. Unknown ids fall back to the raw id, so a new
// artifact in the data shows up without code changes.

const ARTIFACT: Record<string, string> = {
  "qwen3-4b-bf16": "Qwen3-4B bf16",
  "qwen3-4b-f16-gguf": "Qwen3-4B GGUF F16",
  "qwen3-4b-fp8-dynamic": "Qwen3-4B FP8-dynamic (ours)",
  "qwen3-4b-fp8-official": "Qwen3-4B FP8 (Qwen official)",
  "qwen3-4b-w8a8-int8": "Qwen3-4B W8A8-INT8",
  "qwen3-4b-w4a16-gptq-g128": "Qwen3-4B W4A16 GPTQ",
  "qwen3-4b-w4a16-awq-asym": "Qwen3-4B W4A16 AWQ",
  "qwen3-4b-q8_0": "Qwen3-4B GGUF Q8_0",
  "qwen3-4b-q5_k_m": "Qwen3-4B GGUF Q5_K_M",
  "qwen3-4b-q4_k_m": "Qwen3-4B GGUF Q4_K_M",
  "qwen3-4b-q4_k_m-imat-generic": "Qwen3-4B GGUF Q4_K_M + imatrix",
  "qwen3-30b-a3b-bf16": "Qwen3-30B-A3B MoE bf16",
  "qwen3.5-4b-bf16": "Qwen3.5-4B bf16 (hybrid)",
  "gemma4-e4b-bf16": "Gemma 4 E4B bf16",
  "qwen3-0.6b-q4_k_m": "Qwen3-0.6B Q4_K_M",
  "qwen3-0.6b-q8_0": "Qwen3-0.6B Q8_0",
  "qwen3-1.7b-q4_k_m": "Qwen3-1.7B Q4_K_M",
  "qwen3-1.7b-q8_0": "Qwen3-1.7B Q8_0",
  "gemma4-e2b-q4_k_m": "Gemma 4 E2B Q4_K_M",
  "gemma4-e2b-q8_0": "Gemma 4 E2B Q8_0",
  "gemma4-e2b-qat-q4_0": "Gemma 4 E2B QAT Q4_0 (Google)",
  "gemma4-e2b-ptq-q4_0": "Gemma 4 E2B PTQ Q4_0",
};

/** Short form used in chart legends, where the GPU and engine are shown separately. */
const SHORT: Record<string, string> = {
  "qwen3-4b-bf16": "bf16",
  "qwen3-4b-f16-gguf": "GGUF F16",
  "qwen3-4b-fp8-dynamic": "FP8-dynamic",
  "qwen3-4b-fp8-official": "FP8 official",
  "qwen3-4b-w8a8-int8": "INT8",
  "qwen3-4b-w4a16-gptq-g128": "W4A16 GPTQ",
  "qwen3-4b-w4a16-awq-asym": "W4A16 AWQ",
  "qwen3-4b-q8_0": "Q8_0",
  "qwen3-4b-q5_k_m": "Q5_K_M",
  "qwen3-4b-q4_k_m": "Q4_K_M",
  "qwen3-4b-q4_k_m-imat-generic": "Q4_K_M + imatrix",
  "qwen3-30b-a3b-bf16": "30B-A3B MoE",
  "qwen3.5-4b-bf16": "Qwen3.5-4B",
  "gemma4-e4b-bf16": "Gemma 4 E4B",
};

const ENGINE: Record<string, string> = { vllm: "vLLM", sglang: "SGLang", llamacpp: "llama.cpp", ollama: "Ollama" };

const GPU: Record<string, string> = { H100x2: "2×H100", CPU: "CPU" };

const QUALITY_VARIANT: Record<string, string> = {
  "qwen3-4b-bf16": "Qwen3-4B bf16",
  "qwen3-4b-bf16-kv-fp8": "Qwen3-4B bf16 + FP8 KV cache",
  "qwen3-4b-fp8-dynamic": "Qwen3-4B FP8-dynamic (ours)",
  "qwen3-4b-fp8-official": "Qwen3-4B FP8 (Qwen official)",
  "qwen3-4b-w8a8-int8-generic": "Qwen3-4B W8A8-INT8",
  "qwen3-4b-w4a16-gptq-g128-generic": "Qwen3-4B W4A16 GPTQ",
  "qwen3-4b-w4a16-gptq-g128-bfsi": "Qwen3-4B W4A16 GPTQ, BFSI calibration",
  "qwen3-4b-w4a16-awq-asym-generic": "Qwen3-4B W4A16 AWQ",
  "qwen3.5-4b-bf16": "Qwen3.5-4B bf16",
  "qwen3-30b-a3b-bf16": "Qwen3-30B-A3B MoE bf16",
  "gemma4-e4b-bf16": "Gemma 4 E4B bf16",
  "gemma4-e4b-qat-w4a16-g32": "Gemma 4 E4B QAT W4A16 (Google)",
  "gemma4-e4b-ptq-w4a16-g32": "Gemma 4 E4B GPTQ W4A16 (ours)",
  "gemma4-e4b-rtn-w4a16-g32": "Gemma 4 E4B RTN W4A16 (ours)",
};

export const artifactLabel = (id: string): string => ARTIFACT[id] ?? id;
export const artifactShort = (id: string): string => SHORT[id] ?? ARTIFACT[id] ?? id;
export const engineLabel = (id: string): string => ENGINE[id] ?? id;
export const gpuLabel = (id: string): string => GPU[id] ?? id;
export const qualityLabel = (id: string): string => QUALITY_VARIANT[id] ?? id;

export const configKey = (artifact: string, engine: string, gpu: string): string => `${artifact}|${engine}|${gpu}`;
