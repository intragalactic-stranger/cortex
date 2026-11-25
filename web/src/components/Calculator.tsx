"use client";

import { useMemo, useState } from "react";
import { num, sig } from "../lib/format";
import { buildSpec, estimate, KV_BYTES, WEIGHT_BYTES, type Device, type ModelSpec } from "../lib/memcalc";
import type { Config } from "../lib/memcalc";

export interface PresetLite {
  id: string;
  label: string;
  note: string;
  spec: ModelSpec;
}

const CTX_PRESETS = [2048, 8192, 32768, 131072];
const KIND_LABEL: Record<string, string> = {
  full: "full attention",
  sliding: "sliding window",
  shared: "shared KV (no own cache)",
  linear: "linear attention (fixed state)",
};

export function Calculator({ presets, devices }: { presets: PresetLite[]; devices: Device[] }) {
  const [modelId, setModelId] = useState(presets.find((p) => p.id.includes("Qwen3-4B"))?.id ?? presets[0]?.id ?? "");
  const [deviceName, setDeviceName] = useState(devices.find((d) => d.name === "L4")?.name ?? devices[0]?.name ?? "");
  const [weightFmt, setWeightFmt] = useState("bf16");
  const [kvFmt, setKvFmt] = useState("auto");
  const [ctx, setCtx] = useState(8192);
  const [util, setUtil] = useState(0.9);
  const [pleOffload, setPleOffload] = useState(false);
  const [pasted, setPasted] = useState("");
  const [custom, setCustom] = useState<ModelSpec | null>(null);
  const [pasteError, setPasteError] = useState<string | null>(null);

  const spec = modelId === "__custom" ? custom : (presets.find((p) => p.id === modelId)?.spec ?? null);
  const device = devices.find((d) => d.name === deviceName) ?? devices[0];
  const result = useMemo(() => {
    if (!spec || !device) return { est: null, error: null };
    try {
      return { est: estimate(spec, device, { weightFmt, kvFmt, ctx, util, pleOffload }), error: null };
    } catch (e) {
      return { est: null, error: e instanceof Error ? e.message : String(e) };
    }
  }, [spec, device, weightFmt, kvFmt, ctx, util, pleOffload]);
  const est = result.est;

  const usePasted = () => {
    try {
      const cfg = JSON.parse(pasted) as Config;
      const s = buildSpec("pasted config.json", cfg, null, null);
      setCustom(s);
      setModelId("__custom");
      setPasteError(null);
    } catch (e) {
      setPasteError(`Could not read that config: ${e instanceof Error ? e.message : String(e)}`);
    }
  };

  const total = device ? device.memory_gib : 0;
  const pct = (gib: number) => `${Math.max(0, Math.min(100, (gib / total) * 100)).toFixed(2)}%`;
  const kvUsed = est ? (est.max_concurrent_seqs * est.seq_mib) / 1024 : 0;

  return (
    <div className="panel">
      <form className="controls" onSubmit={(e) => e.preventDefault()}>
        <div className="field">
          <label htmlFor="mc-model">Model</label>
          <select id="mc-model" value={modelId} onChange={(e) => setModelId(e.target.value)}>
            {presets.map((p) => (
              <option key={p.id} value={p.id}>
                {p.label}
              </option>
            ))}
            {custom && <option value="__custom">Pasted config.json</option>}
          </select>
        </div>
        <div className="field">
          <label htmlFor="mc-device">Device</label>
          <select id="mc-device" value={deviceName} onChange={(e) => setDeviceName(e.target.value)}>
            {devices.map((d) => (
              <option key={d.name} value={d.name}>
                {d.name} · {sig(d.memory_gib, 3)} GiB
              </option>
            ))}
          </select>
        </div>
        <div className="field">
          <label htmlFor="mc-wf">Weight format</label>
          <select id="mc-wf" value={weightFmt} onChange={(e) => setWeightFmt(e.target.value)}>
            {Object.keys(WEIGHT_BYTES).map((f) => (
              <option key={f}>{f}</option>
            ))}
          </select>
        </div>
        <div className="field">
          <label htmlFor="mc-kv">KV cache format</label>
          <select id="mc-kv" value={kvFmt} onChange={(e) => setKvFmt(e.target.value)}>
            {Object.keys(KV_BYTES).map((f) => (
              <option key={f}>{f}</option>
            ))}
          </select>
        </div>
        <div className="field">
          <label htmlFor="mc-ctx">Context per sequence (tokens)</label>
          <input
            id="mc-ctx"
            type="number"
            min={1}
            step={1024}
            value={ctx}
            onChange={(e) => setCtx(Math.max(1, Number(e.target.value) || 1))}
          />
        </div>
        <div className="field" role="group" aria-label="Context presets">
          <span className="small">Presets</span>
          <div className="chips">
            {CTX_PRESETS.map((c) => (
              <button key={c} type="button" className="chip" aria-pressed={ctx === c} onClick={() => setCtx(c)}>
                {c >= 1024 ? `${c / 1024}K` : c}
              </button>
            ))}
          </div>
        </div>
        <div className="field">
          <label htmlFor="mc-util">GPU memory utilisation: {Math.round(util * 100)}%</label>
          <input
            id="mc-util"
            type="range"
            min={0.3}
            max={1}
            step={0.05}
            value={util}
            onChange={(e) => setUtil(Number(e.target.value))}
          />
        </div>
        <div className="field">
          <label htmlFor="mc-ple">
            <input id="mc-ple" type="checkbox" checked={pleOffload} onChange={(e) => setPleOffload(e.target.checked)} /> Keep
            per-layer embeddings off the GPU (Gemma)
          </label>
        </div>
      </form>

      {result.error && <p className="down">{result.error}</p>}
      {est && device && (
        <>
          <div className="stats" aria-live="polite">
            <div className="stat">
              <b>{sig(est.weights_gib, 3)} GiB</b>
              <span>weights ({est.weight_format})</span>
            </div>
            <div className="stat">
              <b>{sig(est.kv_bytes_per_token / 1024, 3)} KiB</b>
              <span>KV cache per token</span>
            </div>
            <div className="stat">
              <b>{num(est.seq_mib)} MiB</b>
              <span>memory per {num(ctx)}-token sequence</span>
            </div>
            <div className="stat">
              <b>{num(est.max_concurrent_seqs)}</b>
              <span>sequences that fit at once</span>
            </div>
            <div className="stat">
              <b>{est.decode_ceiling_tok_s === null ? "—" : `${num(est.decode_ceiling_tok_s)} tok/s`}</b>
              <span>single-user decode ceiling (bandwidth bound)</span>
            </div>
          </div>
          <div>
            <div className="membar" role="img" aria-label={`Memory: weights ${sig(est.weights_gib, 3)} GiB, reserve ${est.reserve_gib} GiB, KV for ${est.max_concurrent_seqs} sequences ${sig(kvUsed, 3)} GiB, of ${sig(total, 3)} GiB`}>
              <span className="mb-w" style={{ width: pct(est.weights_gib) }} />
              <span className="mb-r" style={{ width: pct(est.reserve_gib) }} />
              <span className="mb-kv" style={{ width: pct(kvUsed) }} />
            </div>
            <ul className="legend" style={{ marginTop: "0.4rem" }}>
              <li>
                <span className="swatch mb-w" /> weights
              </li>
              <li>
                <span className="swatch mb-r" /> runtime reserve ({est.reserve_gib} GiB)
              </li>
              <li>
                <span className="swatch mb-kv" /> KV for {num(est.max_concurrent_seqs)} sequences
              </li>
              <li>unshaded: outside the {Math.round(util * 100)}% budget or unused</li>
            </ul>
          </div>
          <div className="table-wrap">
            <table>
              <tbody>
                <tr>
                  <th scope="row">Layers</th>
                  <td>
                    {Object.entries(est.layer_kinds)
                      .map(([k, n]) => `${n} ${KIND_LABEL[k] ?? k}`)
                      .join(", ")}
                  </td>
                </tr>
                <tr>
                  <th scope="row">Parameters</th>
                  <td className="num">
                    {sig(est.params_stored / 1e9, 3)} B stored · {sig(est.params_loaded / 1e9, 3)} B loaded ·{" "}
                    {sig(est.params_active / 1e9, 3)} B active per token
                  </td>
                </tr>
                <tr>
                  <th scope="row">Fixed state per sequence</th>
                  <td className="num">{sig(est.fixed_state_mib_per_seq, 3)} MiB</td>
                </tr>
                <tr>
                  <th scope="row">Free for KV</th>
                  <td className="num">
                    {sig(est.free_for_kv_gib, 3)} GiB of a {sig(est.budget_gib, 3)} GiB budget
                  </td>
                </tr>
                <tr>
                  <th scope="row">Prefill compute</th>
                  <td className="num">{sig(est.prefill_tflop_per_1k_tokens, 3)} TFLOP per 1K prompt tokens</td>
                </tr>
              </tbody>
            </table>
          </div>
          {est.notes.length > 0 && (
            <ul className="small">
              {est.notes.map((n) => (
                <li key={n}>{n}</li>
              ))}
            </ul>
          )}
        </>
      )}

      <details>
        <summary>Paste a Hugging Face config.json</summary>
        <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem", marginTop: "0.6rem" }}>
          <label htmlFor="mc-paste" className="small">
            Parameter counts come from the config (no safetensors header in the browser), so they are approximate.
          </label>
          <textarea id="mc-paste" value={pasted} onChange={(e) => setPasted(e.target.value)} placeholder='{"model_type": "qwen3", ...}' />
          <div>
            <button type="button" className="primary" onClick={usePasted} disabled={!pasted.trim()}>
              Use this config
            </button>
          </div>
          {pasteError && <p className="down">{pasteError}</p>}
        </div>
      </details>
    </div>
  );
}
