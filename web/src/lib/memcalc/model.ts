// Port of serve/src/cortex_serve/memcalc/model.py: every architecture becomes a list of
// per-layer specs, so dense GQA, sliding-window, cross-layer KV sharing and linear-attention
// layers all go through the same arithmetic in calc.ts.

export type LayerKind = "full" | "sliding" | "shared" | "linear";
export const LAYER_KINDS: readonly LayerKind[] = ["full", "sliding", "shared", "linear"];

export interface LayerSpec {
  index: number;
  kind: LayerKind;
  qHeads: number;
  kvHeads: number;
  headDim: number;
  window: number | null;
  sharesFrom: number | null;
  // Linear-attention state (per sequence), in elements; dtype sizes are given separately.
  stateElems: number;
  stateDtypeBytes: number;
  convElems: number;
  convDtypeBytes: number;
}

export function layer(partial: Partial<LayerSpec> & Pick<LayerSpec, "index" | "kind">): LayerSpec {
  return {
    qHeads: 0,
    kvHeads: 0,
    headDim: 0,
    window: null,
    sharesFrom: null,
    stateElems: 0,
    stateDtypeBytes: 0,
    convElems: 0,
    convDtypeBytes: 0,
    ...partial,
  };
}

export interface MoESpec {
  numExperts: number;
  expertsPerToken: number;
}

export function activeFraction(moe: MoESpec): number {
  return moe.expertsPerToken / moe.numExperts;
}

/** Parameter counts by component (element counts, not bytes). */
export interface ParamBreakdown {
  /** 'safetensors-header' (exact) or 'analytic' (from config.json) */
  source: "safetensors-header" | "analytic";
  decoderLayers: number;
  experts: number;
  embeddings: number;
  lmHead: number; // 0 when tied to the embeddings
  perLayerEmbeddings: number; // Gemma PLE lookup tables (looked up, not multiplied)
  perLayerCompute: number; // Gemma PLE projections that are matmuls
  unusedSharedKv: number; // k/v weights stored in KV-shared layers but never used
  tiedDuplicate: number; // lm_head stored although the config ties it (never loaded)
  vision: number;
  audio: number;
  mtp: number; // multi-token-prediction heads
  other: number; // final norms and anything unclassified
}

export const PARAM_BUCKETS = [
  "decoderLayers",
  "experts",
  "embeddings",
  "lmHead",
  "perLayerEmbeddings",
  "perLayerCompute",
  "unusedSharedKv",
  "tiedDuplicate",
  "vision",
  "audio",
  "mtp",
  "other",
] as const;
export type ParamBucket = (typeof PARAM_BUCKETS)[number];

export function breakdown(
  source: ParamBreakdown["source"],
  counts: Partial<Record<ParamBucket, number>>,
): ParamBreakdown {
  const out = { source } as ParamBreakdown;
  for (const b of PARAM_BUCKETS) out[b] = counts[b] ?? 0;
  return out;
}

export function storedParams(p: ParamBreakdown): number {
  // Same summation order as ParamBreakdown.stored in Python.
  return (
    p.decoderLayers +
    p.experts +
    p.embeddings +
    p.lmHead +
    p.perLayerEmbeddings +
    p.perLayerCompute +
    p.unusedSharedKv +
    p.tiedDuplicate +
    p.vision +
    p.audio +
    p.mtp +
    p.other
  );
}

export interface ModelSpec {
  name: string;
  revision: string | null;
  modelType: string;
  hiddenSize: number;
  vocabSize: number;
  tiedEmbeddings: boolean;
  maxContext: number | null;
  layers: LayerSpec[];
  moe: MoESpec | null;
  params: ParamBreakdown;
}

export function countKind(spec: ModelSpec, kind: LayerKind): number {
  return spec.layers.filter((l) => l.kind === kind).length;
}

export function ownsKv(l: LayerSpec): boolean {
  return l.kind === "full" || l.kind === "sliding";
}

/** The output projection is a real matmul even when its weights are tied to the embeddings. */
export function lmHeadParams(spec: ModelSpec): number {
  return spec.params.lmHead ? spec.params.lmHead : spec.vocabSize * spec.hiddenSize;
}

/** Python's round(): round half to even. */
export function pyRound(x: number): number {
  const f = Math.floor(x);
  const d = x - f;
  if (d > 0.5) return f + 1;
  if (d < 0.5) return f;
  return f % 2 === 0 ? f : f + 1;
}

/**
 * Params whose weights are read for one decode step of one sequence. Excludes lookup-only tables
 * (input embeddings, PLE), unused weights and non-text towers; includes the lm_head matmul and,
 * for MoE, only the routed share of the experts.
 */
export function activeParams(spec: ModelSpec): number {
  const p = spec.params;
  const experts = spec.moe ? pyRound(p.experts * activeFraction(spec.moe)) : p.experts;
  return p.decoderLayers + experts + p.perLayerCompute + lmHeadParams(spec);
}
