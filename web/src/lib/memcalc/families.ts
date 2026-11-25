// Port of serve/src/cortex_serve/memcalc/families.py: turn a Hugging Face config.json into per-layer specs.
//  * dense / GQA / MoE transformers: every layer FULL
//  * Gemma 4: SLIDING + FULL layers (FULL uses global_head_dim); the last num_kv_shared_layers layers reuse
//    the KV of the last non-shared layer of the same type
//  * Qwen3.5 hybrids: LINEAR (Gated DeltaNet) + FULL layers; recurrent state in mamba_ssm_dtype

import { STATE_DTYPE_BYTES } from "./dtypes.ts";
import { layer, type LayerKind, type LayerSpec, type MoESpec } from "./model.ts";
import { get, int, or, req, truthy, type Config } from "./py.ts";

export class UnsupportedArchitectureError extends Error {}

const KIND_BY_LAYER_TYPE: Readonly<Record<string, LayerKind>> = {
  full_attention: "full",
  attention: "full",
  sliding_attention: "sliding",
  linear_attention: "linear",
};

/** Multimodal checkpoints nest the language model under text_config. */
export function textConfig(cfg: Config): Config {
  const tc = get(cfg, "text_config");
  return tc !== null && tc !== undefined && typeof tc === "object" && !Array.isArray(tc) ? tc : cfg;
}

export function tiedEmbeddings(cfg: Config): boolean {
  const tc = textConfig(cfg);
  // cfg.get("tie_word_embeddings", tc.get(...)): the default applies only when the key is absent
  const value = get(cfg, "tie_word_embeddings") !== undefined ? get(cfg, "tie_word_embeddings") : get(tc, "tie_word_embeddings");
  return value !== null && value !== undefined ? truthy(value) : false;
}

export function modelDtypeBytes(cfg: Config): number {
  const tc = textConfig(cfg);
  const name =
    or(get(tc, "dtype"), get(tc, "torch_dtype"), get(cfg, "dtype"), get(cfg, "torch_dtype")) || "bfloat16";
  return STATE_DTYPE_BYTES[String(name)] ?? 2.0;
}

function layerKinds(tc: Config): LayerKind[] {
  const n = int(req(tc, "num_hidden_layers"), "num_hidden_layers");
  const types = get(tc, "layer_types");
  if (truthy(types)) {
    if (!Array.isArray(types)) throw new UnsupportedArchitectureError("layer_types must be a list");
    if (types.length !== n) {
      throw new UnsupportedArchitectureError(`layer_types has ${types.length} entries but ${n} layers`);
    }
    const unknown = [...new Set(types.map(String).filter((t) => !(t in KIND_BY_LAYER_TYPE)))].sort();
    if (unknown.length) {
      throw new UnsupportedArchitectureError(`unsupported layer types [${unknown.join(", ")}] (e.g. Mamba layers)`);
    }
    return types.map((t) => KIND_BY_LAYER_TYPE[String(t)]!);
  }
  if (truthy(get(tc, "use_sliding_window")) && truthy(get(tc, "sliding_window"))) {
    return Array<LayerKind>(n).fill("sliding");
  }
  return Array<LayerKind>(n).fill("full");
}

export function buildLayers(cfg: Config): LayerSpec[] {
  const tc = textConfig(cfg);
  const hidden = int(req(tc, "hidden_size"), "hidden_size");
  const qHeads = int(req(tc, "num_attention_heads"), "num_attention_heads");
  const kvHeads = int(or(get(tc, "num_key_value_heads"), qHeads), "num_key_value_heads");
  const headDim = int(or(get(tc, "head_dim"), Math.floor(hidden / qHeads)), "head_dim");
  const window = get(tc, "sliding_window");
  // Gemma 4 global (full-attention) layers use a larger head_dim and may use their own KV-head count
  const globalHeadDim = int(or(get(tc, "global_head_dim"), headDim), "global_head_dim");
  const globalKvHeads = int(or(get(tc, "num_global_key_value_heads"), kvHeads), "num_global_key_value_heads");
  const actBytes = modelDtypeBytes(cfg);

  const layers: LayerSpec[] = layerKinds(tc).map((kind, i) => {
    if (kind === "full") {
      return layer({ index: i, kind, qHeads, kvHeads: globalKvHeads, headDim: globalHeadDim });
    }
    if (kind === "sliding") {
      return layer({ index: i, kind, qHeads, kvHeads, headDim, window: truthy(window) ? int(window, "sliding_window") : null });
    }
    // LINEAR (Gated DeltaNet)
    const kHeads = int(req(tc, "linear_num_key_heads"), "linear_num_key_heads");
    const vHeads = int(req(tc, "linear_num_value_heads"), "linear_num_value_heads");
    const kDim = int(req(tc, "linear_key_head_dim"), "linear_key_head_dim");
    const vDim = int(req(tc, "linear_value_head_dim"), "linear_value_head_dim");
    const kernel = int(req(tc, "linear_conv_kernel_dim"), "linear_conv_kernel_dim");
    const stateDtype = String(or(get(tc, "mamba_ssm_dtype"), "float32"));
    const stateBytes = STATE_DTYPE_BYTES[stateDtype];
    if (stateBytes === undefined) throw new UnsupportedArchitectureError(`unknown mamba_ssm_dtype '${stateDtype}'`);
    return layer({
      index: i,
      kind,
      stateElems: vHeads * kDim * vDim,
      stateDtypeBytes: stateBytes,
      convElems: (2 * kHeads * kDim + vHeads * vDim) * (kernel - 1),
      convDtypeBytes: actBytes,
    });
  });
  return applyKvSharing(layers, int(or(get(tc, "num_kv_shared_layers"), 0), "num_kv_shared_layers"));
}

/** Mirror transformers' Gemma4 kv_sharing_modes: the last `numShared` layers reuse the KV of the last
 * non-shared layer of the same kind. Shared layers keep their attention shape but own no KV. */
function applyKvSharing(layers: LayerSpec[], numShared: number): LayerSpec[] {
  if (numShared === 0) return layers;
  const firstShared = layers.length - numShared;
  const producers = new Map<LayerKind, LayerSpec>();
  for (const l of layers.slice(0, firstShared)) producers.set(l.kind, l);
  const out = layers.slice(0, firstShared);
  for (const l of layers.slice(firstShared)) {
    const producer = producers.get(l.kind);
    if (!producer) throw new UnsupportedArchitectureError(`layer ${l.index} (${l.kind}) has no KV producer`);
    out.push({ ...l, kind: "shared", sharesFrom: producer.index });
  }
  return out;
}

export function producerOf(layers: readonly LayerSpec[], l: LayerSpec): LayerSpec {
  return l.sharesFrom !== null ? layers[l.sharesFrom]! : l;
}

export function buildMoe(cfg: Config): MoESpec | null {
  const tc = textConfig(cfg);
  const experts = or(get(tc, "num_experts"), get(tc, "num_local_experts"));
  const perToken = or(get(tc, "num_experts_per_tok"), get(tc, "top_k_experts"));
  if (truthy(experts) && truthy(perToken)) {
    return { numExperts: int(experts, "num_experts"), expertsPerToken: int(perToken, "num_experts_per_tok") };
  }
  return null;
}
