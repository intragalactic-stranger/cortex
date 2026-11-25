// Port of build_spec() in serve/src/cortex_serve/memcalc/loader.py (no file or Hub I/O here: the caller
// supplies the parsed config.json and, optionally, the safetensors header table).

import { buildLayers, buildMoe, textConfig, tiedEmbeddings } from "./families.ts";
import type { ModelSpec } from "./model.ts";
import { analyticParams, classifyTensors, type TensorTable } from "./params.ts";
import { get, int, type Config } from "./py.ts";

export function buildSpec(name: string, cfg: Config, tensors?: TensorTable | null, revision?: string | null): ModelSpec {
  const layers = buildLayers(cfg);
  let params;
  if (tensors) {
    params = classifyTensors(tensors, layers, tiedEmbeddings(cfg));
  } else {
    const analytic = analyticParams(cfg);
    if (analytic === null) {
      throw new Error(
        `${name}: no safetensors header available and no analytic formula for ` +
          `model_type='${String(get(textConfig(cfg), "model_type"))}'; exact counts need the checkpoint's tensor table`,
      );
    }
    params = analytic;
  }
  const tc = textConfig(cfg);
  const maxCtx = get(tc, "max_position_embeddings");
  return {
    name,
    revision: revision ?? null,
    modelType: String(get(tc, "model_type")),
    hiddenSize: int(get(tc, "hidden_size"), "hidden_size"),
    vocabSize: int(get(tc, "vocab_size"), "vocab_size"),
    tiedEmbeddings: tiedEmbeddings(cfg),
    maxContext: typeof maxCtx === "number" ? maxCtx : null,
    layers,
    moe: buildMoe(cfg),
    params,
  };
}
