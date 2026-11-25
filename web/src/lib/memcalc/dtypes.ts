// Port of serve/src/cortex_serve/memcalc/dtypes.py. Keep the two files in sync; the parity
// tests (web/test/*.test.ts) fail if they drift.

/** Bytes per parameter. Weight formats include their per-group scale overhead. */
export const WEIGHT_BYTES: Readonly<Record<string, number>> = {
  fp32: 4.0,
  bf16: 2.0,
  fp16: 2.0,
  fp8: 1.0, // E4M3, per-tensor/per-channel scales are negligible
  int8: 1.0, // W8A8 per-channel scales negligible
  "w4a16-g128": 0.5 + 2 / 128, // int4 + one fp16 scale per 128 weights (GPTQ)
  "awq-g128": 0.5 + 2 / 128 + 0.5 / 128, // + int4 zero-point per group
  mxfp4: 4.25 / 8, // 4-bit + one 8-bit shared exponent per 32 values
  // GGUF (llama.cpp) average bits per weight / 8
  q8_0: 8.5 / 8,
  q6_k: 6.5625 / 8,
  q5_k_m: 5.69 / 8, // approximate mix
  q4_k_m: 4.89 / 8, // approximate mix
  q4_0: 4.5 / 8,
};

/** Bytes per KV-cache element. */
export const KV_BYTES: Readonly<Record<string, number>> = {
  fp32: 4.0,
  bf16: 2.0,
  fp16: 2.0,
  auto: 2.0, // engines default to the model dtype, 16-bit for every model in our lineup
  fp8: 1.0,
  q8_0: 8.5 / 8, // llama.cpp --cache-type-k q8_0
  q4_0: 4.5 / 8,
};

export const APPROXIMATE_WEIGHT_FORMATS: ReadonlySet<string> = new Set(["q5_k_m", "q4_k_m"]);

export const STATE_DTYPE_BYTES: Readonly<Record<string, number>> = { float32: 4.0, bfloat16: 2.0, float16: 2.0 };

export function weightBytes(fmt: string): number {
  const b = WEIGHT_BYTES[fmt.toLowerCase()];
  if (b === undefined) {
    throw new Error(`unknown weight format '${fmt}'; choose from ${Object.keys(WEIGHT_BYTES).sort().join(", ")}`);
  }
  return b;
}

export function kvBytes(fmt: string): number {
  const b = KV_BYTES[fmt.toLowerCase()];
  if (b === undefined) {
    throw new Error(`unknown KV-cache format '${fmt}'; choose from ${Object.keys(KV_BYTES).sort().join(", ")}`);
  }
  return b;
}
