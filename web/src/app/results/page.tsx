import type { Metadata } from "next";
import { CurveExplorer } from "../../components/CurveExplorer";
import { FrontierChart } from "../../components/FrontierChart";
import { QualityTable } from "../../components/QualityTable";
import { loadConfigs, loadCurves, loadFrontier, loadQuality } from "../../lib/data";
import { num, usd } from "../../lib/format";
import { artifactLabel, configKey, engineLabel, gpuLabel } from "../../lib/labels";
import { REPO_URL } from "../../lib/markdown";

export const metadata: Metadata = {
  title: "Results",
  description: "Throughput–latency curves, the cost–quality frontier and quality deltas for every measured configuration.",
};

export default function ResultsPage() {
  const configs = loadConfigs();
  const curves = loadCurves();
  const quality = loadQuality();
  const frontier = loadFrontier();
  const base = quality["qwen3-4b-bf16"]?.["composite"] ?? 0.81;
  const fset = new Set(frontier);
  const sorted = [...configs].sort((a, b) => a.usd_per_1m_slo_tokens - b.usd_per_1m_slo_tokens);

  return (
    <>
      <section aria-labelledby="title">
        <p className="eyebrow">Results explorer · data from serve/results/web</p>
        <h1 id="title">Results</h1>
        <p className="lede">
          Every point below comes from a run directory with a manifest (git SHA, image digest, model revision, the GPU
          that actually ran). The raw tables are in{" "}
          <a href={`${REPO_URL}/blob/main/serve/RESULTS.md`}>RESULTS.md</a>.
        </p>
      </section>

      <section aria-label="Cost frontier">
        <FrontierChart rows={configs} frontier={frontier} bandCenter={Number((base * 100).toFixed(1))} bandHalf={2.4} />
      </section>

      <section aria-label="Throughput and latency curves">
        <CurveExplorer curves={curves} />
      </section>

      <section aria-labelledby="cost-table">
        <h2 id="cost-table">All configurations by cost</h2>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th scope="col">Hardware</th>
                <th scope="col">Engine</th>
                <th scope="col">Model / format</th>
                <th scope="col" className="n">
                  Best load (users)
                </th>
                <th scope="col" className="n">
                  Goodput tok/s
                </th>
                <th scope="col" className="n">
                  $ / 1M SLO tokens
                </th>
                <th scope="col" className="n">
                  Quality %
                </th>
                <th scope="col" className="n">
                  ms/token, 1 user
                </th>
              </tr>
            </thead>
            <tbody>
              {sorted.map((r) => (
                <tr key={configKey(r.artifact, r.engine, r.gpu)}>
                  <td>{gpuLabel(r.gpu)}</td>
                  <td>{engineLabel(r.engine)}</td>
                  <td>
                    {artifactLabel(r.artifact)}
                    {fset.has(configKey(r.artifact, r.engine, r.gpu)) ? " · frontier" : ""}
                  </td>
                  <td className="n">{num(r.best_level)}</td>
                  <td className="n">{num(r.goodput_tok_s)}</td>
                  <td className="n">{usd(r.usd_per_1m_slo_tokens)}</td>
                  <td className="n">{r.quality === null ? "—" : (r.quality * 100).toFixed(1)}</td>
                  <td className="n">{num(r.tpot_p50_ms_1user, 1)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="small">
          CPU rows use the CPU SLO (first token ≤ 3 s, ≤ 125 ms per token), a 0.6B model and chat 128/64, priced at the
          AWS t3a.xlarge list price. Load levels were coarse (1, 4, 16, 32, 64, 128), so a better point can lie between them.
        </p>
      </section>

      <section aria-label="Quality">
        <QualityTable quality={quality} />
      </section>
    </>
  );
}
