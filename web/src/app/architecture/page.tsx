import type { Metadata } from "next";
import { REPO_URL, renderMarkdown } from "../../lib/markdown";

export const metadata: Metadata = {
  title: "Architecture",
  description: "How Cortex Serve is built: the measurement lab, the Modal GPU apps and the production-style serving stack.",
};

const DIAGRAM = `
## Measurement lab

\`\`\`mermaid
flowchart LR
    MC[memcalc<br/>memory model] --> SB[servebench<br/>timing, goodput]
    SB --> MA[Modal GPU apps<br/>bench, quantize, eval]
    MA --> R[results<br/>manifests, parquet]
    R --> REP[reports<br/>guide, frontier, site]
\`\`\`

## Production-style serving (k3d)

\`\`\`mermaid
flowchart LR
    C[client] --> GW[LiteLLM gateway<br/>keys, limits, canary]
    GW --> P1[llama.cpp pool<br/>KEDA on queue depth]
    GW --> FB[second local model]
    GW -.->|fallback| GPU[Modal vLLM GPU endpoint]
    P1 --> PR[Prometheus and Grafana<br/>alerts]
\`\`\`
`;

const PARTS: { name: string; path: string; what: string }[] = [
  { name: "memcalc", path: "serve/src/cortex_serve/memcalc", what: "Weights from safetensors/GGUF headers; KV per layer type (full, sliding, shared, linear); max concurrency; decode ceilings. Ported to TypeScript for this site." },
  { name: "servebench", path: "serve/src/cortex_serve/servebench", what: "Client-side TTFT, time per token and inter-token latency; closed-loop and Poisson load; goodput; GPU telemetry; Prometheus counters; a manifest per run." },
  { name: "Modal apps", path: "serve/deploy/modal", what: "Benchmarks, llm-compressor quantization, lm-eval quality, GGUF conversion and perplexity, a GPU endpoint and a Hugging Face publisher, each with a hard timeout and a spend guard." },
  { name: "Quantization", path: "serve/src/cortex_serve/quantize", what: "FP8, INT8, GPTQ and AWQ recipes; generic and finance-regulation calibration sets; model cards built from results." },
  { name: "Serving stack", path: "serve/deploy/helm/cortex-serve", what: "Helm chart for llama.cpp pools, the LiteLLM gateway, KEDA autoscaling on queue depth, NetworkPolicies, Grafana dashboard and alerts." },
  { name: "gatewaybench", path: "serve/src/cortex_serve/gatewaybench", what: "Gateway load profiles, per-request outcome logs, autoscaling timelines and a canary promotion gate." },
  { name: "This site", path: "web", what: "Static Next.js export. All data is read from the repository at build time; nothing is fetched at runtime." },
];

export default function ArchitecturePage() {
  const { html } = renderMarkdown(DIAGRAM, "web/src/app/architecture/page.tsx");
  return (
    <>
      <section aria-labelledby="title">
        <p className="eyebrow">How it is built</p>
        <h1 id="title">Architecture</h1>
        <p className="lede">
          A measurement lab that produces every number on this site, and a production-style serving stack that applies
          what the lab found.
        </p>
      </section>
      <article className="doc" dangerouslySetInnerHTML={{ __html: html }} />
      <section aria-labelledby="parts">
        <h2 id="parts">Components</h2>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th scope="col">Component</th>
                <th scope="col">What it does</th>
              </tr>
            </thead>
            <tbody>
              {PARTS.map((p) => (
                <tr key={p.name}>
                  <td>
                    <a href={`${REPO_URL}/tree/main/${p.path}`}>{p.name}</a>
                  </td>
                  <td>{p.what}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
      <section aria-labelledby="practices">
        <h2 id="practices">Practices that kept the numbers honest</h2>
        <ul className="prose" style={{ margin: 0, paddingLeft: "1.2rem", display: "flex", flexDirection: "column", gap: "0.35rem" }}>
          <li>Every run records the git SHA, image digest, model revision and the GPU that actually ran.</li>
          <li>Engine images and model revisions are pinned; flags are checked against each engine before GPU time is spent.</li>
          <li>Paid GPU jobs refuse to start if the worst case would cross the approved budget or anything is billed beyond credits.</li>
          <li>Failed and superseded runs stay on record with a note instead of being deleted.</li>
        </ul>
      </section>
    </>
  );
}
