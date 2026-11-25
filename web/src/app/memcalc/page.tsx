import type { Metadata } from "next";
import { Calculator } from "../../components/Calculator";
import { loadDeviceTable, loadPresets } from "../../lib/data";
import { REPO_URL } from "../../lib/markdown";

export const metadata: Metadata = {
  title: "Memory calculator",
  description: "Weights, KV cache per layer type and maximum concurrency for open models on common GPUs, computed in the browser.",
};

export default function MemcalcPage() {
  const presets = loadPresets();
  const devices = loadDeviceTable();
  return (
    <>
      <section aria-labelledby="title">
        <p className="eyebrow">memcalc · TypeScript port, checked against the Python tool in CI</p>
        <h1 id="title">Memory calculator</h1>
        <p className="lede">
          How much GPU memory a model needs and how many sequences fit at once. It counts KV memory layer by layer, so
          sliding-window, shared-KV and linear-attention layers are handled correctly. Everything runs in your browser.
        </p>
      </section>
      <section aria-label="Calculator">
        <Calculator presets={presets} devices={devices} />
      </section>
      <section aria-labelledby="accuracy">
        <h2 id="accuracy">How accurate is it?</h2>
        <ul className="prose" style={{ margin: 0, paddingLeft: "1.2rem", display: "flex", flexDirection: "column", gap: "0.35rem" }}>
          <li>
            For dense models it matched vLLM's KV capacity to within −7% to +13% across seven engine, GPU and utilisation
            settings, and exactly for Qwen3-4B at 8K and 32K.
          </li>
          <li>
            For sliding-window and hybrid models, engines allocate more than the theoretical minimum: vLLM used 6–19% more
            per sequence for Qwen3.5-4B and 30–109% more for Gemma 4 E4B.
          </li>
          <li>
            Weight sizes for the presets come from the checkpoints&apos; safetensors headers; GGUF k-quant sizes are averages.
          </li>
        </ul>
        <p className="small">
          The same calculator is a Python CLI: <code>uv run memcalc estimate -m Qwen/Qwen3-4B-Instruct-2507 -d L4</code>{" "}
          (<a href={`${REPO_URL}/tree/main/serve/src/cortex_serve/memcalc`}>source</a>).
        </p>
      </section>
    </>
  );
}
