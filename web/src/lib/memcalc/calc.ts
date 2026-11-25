// Port of serve/src/cortex_serve/memcalc/calc.py: weights, KV cache, concurrency, decode ceiling, FLOPs.
// The operations are written in the same order as the Python so that floating-point results match bit
// for bit (web/test/vectors.test.ts compares every field against Python output). GGUF-header inputs
// (exact GGUF weight bytes) are not ported: the site has no GGUF upload.

import { memoryBytes, type Device } from "./devices.ts";
import { APPROXIMATE_WEIGHT_FORMATS, kvBytes, weightBytes } from "./dtypes.ts";
import { producerOf } from "./families.ts";
import { activeParams, countKind, LAYER_KINDS, lmHeadParams, storedParams, type LayerKind, type LayerSpec, type ModelSpec } from "./model.ts";

export type Components = "text" | "all";

export const GIB = 2 ** 30;
export const MIB = 2 ** 20;
export const DEFAULT_RESERVE_GIB: Readonly<Record<Device["kind"], number>> = { gpu: 1.5, cpu: 0.5, unified: 0.5 };

// llm-compressor-style formats quantize the decoder's linear layers only; embeddings, lm_head, norms and
// non-text towers stay in 16-bit. GGUF quantizes embeddings too.
const LINEAR_ONLY_FORMATS = new Set(["fp8", "int8", "w4a16-g128", "awq-g128", "mxfp4"]);

// ---------------------------------------------------------------- weights

/** [params in quantizable decoder linears, params kept in 16-bit for llm-compressor formats] */
export function loadedParamGroups(
  spec: ModelSpec,
  components: Components = "text",
  pleOffload = false,
  withMtp = false,
): [number, number] {
  const p = spec.params;
  let decoder = p.decoderLayers + p.experts + p.perLayerCompute + p.unusedSharedKv;
  let kept = p.embeddings + p.lmHead + p.other + (pleOffload ? 0 : p.perLayerEmbeddings);
  if (withMtp) decoder += p.mtp;
  if (components === "all") kept += p.vision + p.audio;
  return [decoder, kept];
}

export function weightsBytes(
  spec: ModelSpec,
  fmt: string,
  components: Components = "text",
  pleOffload = false,
  withMtp = false,
): number {
  const [decoder, kept] = loadedParamGroups(spec, components, pleOffload, withMtp);
  const b = weightBytes(fmt);
  const keptB = LINEAR_ONLY_FORMATS.has(fmt.toLowerCase()) ? 2.0 : b;
  return Math.ceil(decoder * b + kept * keptB);
}

/** Bytes read from memory to decode one token of one sequence (MoE: routed experts only). */
export function activeWeightBytes(spec: ModelSpec, fmt: string): number {
  const b = weightBytes(fmt);
  const lmB = LINEAR_ONLY_FORMATS.has(fmt.toLowerCase()) ? 2.0 : b;
  return Math.ceil((activeParams(spec) - lmHeadParams(spec)) * b + lmHeadParams(spec) * lmB);
}

// ---------------------------------------------------------------- KV cache and state

export function layerSeqBytes(l: LayerSpec, ctx: number, kvB: number): number {
  if (l.kind === "full") return 2 * l.kvHeads * l.headDim * kvB * ctx;
  if (l.kind === "sliding") {
    if (l.window === null) throw new Error(`sliding layer ${l.index} has no window`);
    return 2 * l.kvHeads * l.headDim * kvB * Math.min(ctx, l.window);
  }
  if (l.kind === "linear") return l.stateElems * l.stateDtypeBytes + l.convElems * l.convDtypeBytes;
  return 0.0; // shared: reads another layer's KV
}

/** KV cache + recurrent state for one sequence of `ctx` tokens. */
export function seqBytes(spec: ModelSpec, ctx: number, kvFmt = "bf16"): number {
  const kvB = kvBytes(kvFmt);
  let total = 0; // Python sum() starts at int 0 and adds left to right
  for (const l of spec.layers) total += layerSeqBytes(l, ctx, kvB);
  return Math.ceil(total);
}

export function fixedStateBytes(spec: ModelSpec): number {
  return seqBytes(spec, 0); // only linear layers contribute at ctx=0
}

/** Marginal KV growth per token once past every sliding window (full layers only). */
export function kvBytesPerToken(spec: ModelSpec, kvFmt = "bf16"): number {
  const kvB = kvBytes(kvFmt);
  let total = 0;
  for (const x of spec.layers) if (x.kind === "full") total += 2 * x.kvHeads * x.headDim * kvB;
  return Math.ceil(total);
}

/** Per-layer breakdown for display: bytes one sequence of `ctx` tokens holds in each layer. */
export function perLayerSeqBytes(spec: ModelSpec, ctx: number, kvFmt = "bf16"): number[] {
  const kvB = kvBytes(kvFmt);
  return spec.layers.map((l) => layerSeqBytes(l, ctx, kvB));
}

// ---------------------------------------------------------------- FLOPs

/** QK^T and AV for one new token attending over its context (shared layers still compute attention). */
export function attentionFlopsPerToken(spec: ModelSpec, ctx: number): number {
  let total = 0;
  for (const l of spec.layers) {
    const src = producerOf(spec.layers, l);
    if (src.kind === "linear") continue; // O(1) per token; folded into the linear term
    const span = src.kind === "sliding" && src.window ? Math.min(ctx, src.window) : ctx;
    total += 4 * l.qHeads * l.headDim * span;
  }
  return total;
}

// ---------------------------------------------------------------- estimate

export interface Estimate {
  model: string;
  revision: string | null;
  device: string;
  weight_format: string;
  kv_format: string;
  context: number;
  util: number;
  reserve_gib: number;
  components: Components;
  layer_kinds: Partial<Record<LayerKind, number>>;
  params_stored: number;
  params_loaded: number;
  params_active: number;
  param_source: string;
  weights_gib: number;
  kv_bytes_per_token: number;
  fixed_state_mib_per_seq: number;
  seq_mib: number;
  budget_gib: number;
  free_for_kv_gib: number;
  max_concurrent_seqs: number;
  decode_ceiling_tok_s: number | null;
  decode_ceiling_at_ctx_tok_s: number | null;
  roofline_crossover_batch: number | null;
  prefill_tflop_per_1k_tokens: number;
  attention_gflop_per_token_at_ctx: number;
  notes: string[];
}

export interface EstimateOptions {
  weightFmt?: string;
  kvFmt?: string;
  ctx?: number;
  util?: number;
  reserveGib?: number | null;
  components?: Components;
  pleOffload?: boolean;
  withMtp?: boolean;
}

export function estimate(spec: ModelSpec, device: Device, opts: EstimateOptions = {}): Estimate {
  const weightFmt = opts.weightFmt ?? "bf16";
  const kvFmt = opts.kvFmt ?? "auto";
  const ctx = opts.ctx ?? 8192;
  const util = opts.util ?? 0.9;
  const components = opts.components ?? "text";
  const pleOffload = opts.pleOffload ?? false;
  const withMtp = opts.withMtp ?? false;
  if (!(ctx > 0)) throw new Error("ctx must be positive");
  if (!(util > 0 && util <= 1)) throw new Error("util must be in (0, 1]");
  const notes: string[] = [];
  const reserve = opts.reserveGib ?? DEFAULT_RESERVE_GIB[device.kind];
  const wBytes = weightsBytes(spec, weightFmt, components, pleOffload, withMtp);
  const [decoder, kept] = loadedParamGroups(spec, components, pleOffload, withMtp);
  const perSeq = seqBytes(spec, ctx, kvFmt);
  const budget = memoryBytes(device) * util;
  const free = budget - wBytes - reserve * GIB;
  const maxSeqs = perSeq ? Math.max(0, Math.floor(free / perSeq)) : 0;
  if (free <= 0) notes.push("weights + reserve exceed the memory budget: the model does not fit");
  else if (maxSeqs === 0) notes.push("weights fit, but not even one sequence of this context fits in the KV budget");

  const active = activeWeightBytes(spec, weightFmt);
  let ceiling: number | null = null;
  let ceilingCtx: number | null = null;
  let crossover: number | null = null;
  if (device.bandwidth_gbs) {
    const bw = device.bandwidth_gbs * 1e9;
    ceiling = bw / active;
    ceilingCtx = bw / (active + perSeq);
    let tflops = device.bf16_tflops;
    if (weightFmt.toLowerCase() === "fp8") {
      if (device.fp8_tflops) tflops = device.fp8_tflops;
      else if (device.kind === "gpu") {
        notes.push(
          `${device.name} has no native FP8 compute: engines fall back to weight-only FP8 kernels ` +
            "that dequantize to 16-bit (memory savings remain, compute does not speed up)",
        );
      }
    }
    if (tflops) {
      // decode FLOPs/step = 2*P*B ; bytes/step = P*b  ->  compute-bound once B > (F/BW) * b / 2
      crossover = ((tflops * 1e12) / bw) * weightBytes(weightFmt) / 2;
    } else if (device.kind === "gpu") {
      notes.push(`no 16-bit tensor-core peak listed for ${device.name}: compute crossover omitted`);
    }
  } else {
    notes.push(`no bandwidth figure for ${device.name}: decode ceilings omitted`);
  }

  if (APPROXIMATE_WEIGHT_FORMATS.has(weightFmt.toLowerCase())) {
    notes.push(`${weightFmt} is a mixed-precision GGUF format; bytes/param is an average`);
  }
  if (spec.params.perLayerEmbeddings && !pleOffload) {
    notes.push("per-layer embedding tables counted as device memory (use --ple-offload if the engine maps them)");
  }
  const shared = countKind(spec, "shared");
  if (shared) notes.push(`${shared} layers reuse another layer's KV and own none`);
  if (countKind(spec, "linear")) {
    notes.push("linear-attention layers hold a fixed per-sequence state instead of a growing KV cache");
  }
  if (spec.params.tiedDuplicate) {
    notes.push(
      `checkpoint stores a redundant lm_head (${(spec.params.tiedDuplicate / 1e6).toFixed(0)}M params) ` +
        "although the config ties it to the embeddings; it is not loaded",
    );
  }
  if (spec.params.source === "analytic") notes.push("param counts are analytic (no safetensors header available)");

  const kinds: Partial<Record<LayerKind, number>> = {};
  for (const k of LAYER_KINDS) {
    const n = countKind(spec, k);
    if (n) kinds[k] = n;
  }
  return {
    model: spec.name,
    revision: spec.revision,
    device: device.name,
    weight_format: weightFmt,
    kv_format: kvFmt,
    context: ctx,
    util,
    reserve_gib: reserve,
    components,
    layer_kinds: kinds,
    params_stored: storedParams(spec.params),
    params_loaded: decoder + kept,
    params_active: activeParams(spec),
    param_source: spec.params.source,
    weights_gib: wBytes / GIB,
    kv_bytes_per_token: kvBytesPerToken(spec, kvFmt),
    fixed_state_mib_per_seq: fixedStateBytes(spec) / MIB,
    seq_mib: perSeq / MIB,
    budget_gib: budget / GIB,
    free_for_kv_gib: free / GIB,
    max_concurrent_seqs: maxSeqs,
    decode_ceiling_tok_s: ceiling,
    decode_ceiling_at_ctx_tok_s: ceilingCtx,
    roofline_crossover_batch: crossover,
    prefill_tflop_per_1k_tokens: (2 * activeParams(spec) * 1000) / 1e12,
    attention_gflop_per_token_at_ctx: attentionFlopsPerToken(spec, ctx) / 1e9,
    notes,
  };
}
