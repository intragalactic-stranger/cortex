// Port of serve/src/cortex_serve/memcalc/params.py: parameter counting, two ways.
//  1. classifyTensors: exact counts from the safetensors header table (source of truth).
//  2. analyticParams: textbook formula from config.json for llama-style dense and MoE transformers.

import { textConfig, tiedEmbeddings } from "./families.ts";
import { breakdown, type LayerSpec, type ParamBreakdown, type ParamBucket } from "./model.ts";
import { get, int, truthy, type Config } from "./py.ts";

/** name -> [dtype, shape] */
export type TensorTable = Readonly<Record<string, readonly [string, readonly number[]]>>;

const VISION = /(^|\.)(visual|vision_tower|vision_model|embed_vision|multi_modal_projector)\./;
const AUDIO = /(^|\.)(audio_tower|embed_audio)\./;
const MTP = /(^|\.)mtp\./;
const PLE_TABLE = /embed_tokens_per_layer\./;
const PLE_COMPUTE = /per_layer_model_projection\.|per_layer_projection_norm\./;
const EMBED = /(^|\.)embed_tokens\.weight$/;
const LM_HEAD = /(^|\.)lm_head\./;
const LAYER = /(^|\.)layers\.(\d+)\./;
const EXPERT = /\.experts\./;
const KV_WEIGHTS = /\.self_attn\.(k_proj|v_proj|k_norm|v_norm)\./;

const SIMPLE_RULES: ReadonlyArray<readonly [RegExp, ParamBucket]> = [
  [VISION, "vision"],
  [AUDIO, "audio"],
  [MTP, "mtp"],
  [PLE_TABLE, "perLayerEmbeddings"],
  [PLE_COMPUTE, "perLayerCompute"],
  [EMBED, "embeddings"],
  [LM_HEAD, "lmHead"],
];

/** Return the ParamBreakdown field a tensor belongs to. */
export function classifyTensor(name: string, layers: readonly LayerSpec[], tied = false): ParamBucket {
  for (const [pattern, bucket] of SIMPLE_RULES) {
    if (pattern.test(name)) {
      // e.g. Qwen3-0.6B/1.7B store lm_head although tie_word_embeddings=true
      if (bucket === "lmHead" && tied) return "tiedDuplicate";
      return bucket;
    }
  }
  const m = LAYER.exec(name);
  if (m) {
    if (EXPERT.test(name)) return "experts";
    const idx = Number(m[2]);
    if (idx < layers.length && layers[idx]!.kind === "shared" && KV_WEIGHTS.test(name)) return "unusedSharedKv";
    return "decoderLayers";
  }
  return "other";
}

export function classifyTensors(tensors: TensorTable, layers: readonly LayerSpec[], tied = false): ParamBreakdown {
  const buckets: Partial<Record<ParamBucket, number>> = {};
  for (const [name, [, shape]] of Object.entries(tensors)) {
    const bucket = classifyTensor(name, layers, tied);
    buckets[bucket] = (buckets[bucket] ?? 0) + shape.reduce((a, b) => a * b, 1);
  }
  return breakdown("safetensors-header", buckets);
}

const LLAMA_STYLE = new Set(["llama", "mistral", "qwen2", "qwen3", "qwen3_moe", "smollm3"]);
const QK_NORM = new Set(["qwen3", "qwen3_moe"]);

/** Closed-form count for llama-style decoders; null for families it doesn't cover. */
export function analyticParams(cfg: Config): ParamBreakdown | null {
  const tc = textConfig(cfg);
  const mt = get(tc, "model_type");
  if (typeof mt !== "string" || !LLAMA_STYLE.has(mt)) return null;
  const h = int(get(tc, "hidden_size"), "hidden_size");
  const v = int(get(tc, "vocab_size"), "vocab_size");
  const nLayers = int(get(tc, "num_hidden_layers"), "num_hidden_layers");
  const nh = int(get(tc, "num_attention_heads"), "num_attention_heads");
  const kv = int(truthy(get(tc, "num_key_value_heads")) ? get(tc, "num_key_value_heads") : nh, "num_key_value_heads");
  const hd = int(truthy(get(tc, "head_dim")) ? get(tc, "head_dim") : Math.floor(h / nh), "head_dim");

  let attn = h * nh * hd + 2 * h * kv * hd + nh * hd * h; // q, k, v, o
  if (truthy(get(tc, "attention_bias"))) attn += nh * hd + 2 * kv * hd;
  if (QK_NORM.has(mt)) attn += 2 * hd; // q_norm, k_norm
  const norms = 2 * h; // input + post-attention RMSNorm

  let experts = 0;
  let mlp: number;
  const numExperts = get(tc, "num_experts");
  if (truthy(numExperts)) {
    const moeI = int(get(tc, "moe_intermediate_size"), "moe_intermediate_size");
    experts = nLayers * int(numExperts, "num_experts") * 3 * h * moeI;
    mlp = h * int(numExperts, "num_experts"); // router
  } else {
    mlp = 3 * h * int(get(tc, "intermediate_size"), "intermediate_size"); // gate, up, down (SwiGLU)
  }

  const tied = tiedEmbeddings(cfg);
  return breakdown("analytic", {
    decoderLayers: nLayers * (attn + norms + mlp),
    experts,
    embeddings: v * h,
    lmHead: tied ? 0 : v * h,
    other: h, // final norm
  });
}
