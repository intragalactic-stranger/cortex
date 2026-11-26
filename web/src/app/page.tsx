import Link from "next/link";
import { loadConfigs, loadFrontier, loadHfModelNames, loadSummary } from "../lib/data";
import { usd } from "../lib/format";
import { artifactLabel, configKey, engineLabel, gpuLabel } from "../lib/labels";
import { REPO_URL } from "../lib/markdown";

export default function Home() {
  const summary = loadSummary();
  const configs = loadConfigs();
  const frontier = new Set(loadFrontier());
  const cheapest = [...configs].sort((a, b) => a.usd_per_1m_slo_tokens - b.usd_per_1m_slo_tokens)[0];
  const bestL4 = configs.filter((c) => c.gpu === "L4").sort((a, b) => a.usd_per_1m_slo_tokens - b.usd_per_1m_slo_tokens)[0];
  const l4bf16 = configs.find((c) => c.gpu === "L4" && c.engine === "vllm" && c.artifact === "qwen3-4b-bf16");
  const frontierRows = configs.filter((c) => frontier.has(configKey(c.artifact, c.engine, c.gpu)));
  const hf = loadHfModelNames();

  return (
    <>
      <section className="hero" aria-labelledby="title">
        <p className="eyebrow">LLM inference lab · Modal L4 / H100 · local k3d</p>
        <h1 id="title">What does it really cost to serve an open model well?</h1>
        <blockquote>{summary.question.charAt(0).toUpperCase() + summary.question.slice(1)}</blockquote>
        <p className="lede" dangerouslySetInnerHTML={{ __html: summary.measured }} />
      </section>

      <section aria-labelledby="cost">
        <h2 id="cost">Price of a token that met the latency target</h2>
        <div className="stats">
          {cheapest && (
            <div className="stat">
              <b>{usd(cheapest.usd_per_1m_slo_tokens)}</b>
              <span>
                per 1M SLO-meeting tokens · {gpuLabel(cheapest.gpu)} · {engineLabel(cheapest.engine)} ·{" "}
                {artifactLabel(cheapest.artifact)}
              </span>
            </div>
          )}
          {bestL4 && (
            <div className="stat">
              <b>{usd(bestL4.usd_per_1m_slo_tokens)}</b>
              <span>best L4 setup · {artifactLabel(bestL4.artifact)}</span>
            </div>
          )}
          {l4bf16 && (
            <div className="stat">
              <b>{usd(l4bf16.usd_per_1m_slo_tokens)}</b>
              <span>L4 · vLLM · bf16</span>
            </div>
          )}
          <div className="stat">
            <b>{frontierRows.length}</b>
            <span>configurations on the cost–quality frontier, out of {configs.length} measured</span>
          </div>
        </div>
        <p className="small">
          Cost = hourly price ÷ the best SLO-meeting output throughput we measured (E14). SLO: first token ≤ 0.5 s
          (p95), ≤ 50 ms per output token. <Link href="/results/">Explore every configuration →</Link>
        </p>
      </section>

      <section aria-labelledby="findings">
        <h2 id="findings">Six findings</h2>
        <ol className="findings">
          {summary.findings.map((f) => (
            <li key={f.title}>
              <h3>{f.title.replace(/[:.]$/, "")}</h3>
              <p dangerouslySetInnerHTML={{ __html: f.bodyHtml }} />
            </li>
          ))}
        </ol>
      </section>

      <section aria-labelledby="next">
        <h2 id="next">Where to go next</h2>
        <div className="two-col">
          <div className="panel">
            <h3>
              <Link href="/guide/">Decision guide</Link>
            </h3>
            <p className="small">Which GPU, engine, weight format and settings to use, with the evidence for each.</p>
          </div>
          <div className="panel">
            <h3>
              <Link href="/memcalc/">Memory calculator</Link>
            </h3>
            <p className="small">
              Weights, KV cache per layer type and maximum concurrency for any model on any GPU, in your browser.
            </p>
          </div>
          <div className="panel">
            <h3>
              <Link href="/results/">Results explorer</Link>
            </h3>
            <p className="small">Throughput–latency curves, the cost frontier and quality deltas.</p>
          </div>
          <div className="panel">
            <h3>
              <a href={REPO_URL}>Source code and reports</a>
            </h3>
            <p className="small">Configs, manifests, milestone reports and every raw result on GitHub.</p>
          </div>
        </div>
        {hf.length > 0 && (
          <p className="small">
            Quantized models on Hugging Face:{" "}
            {hf.map((name, i) => (
              <span key={name}>
                {i > 0 && ", "}
                <a href={`https://huggingface.co/${name}`}>{name.split("/")[1]}</a>
              </span>
            ))}
            .
          </p>
        )}
      </section>
    </>
  );
}
