"use client";

import { useState } from "react";
import { usd } from "../lib/format";
import { artifactShort, configKey, engineLabel, gpuLabel } from "../lib/labels";
import type { ConfigRow } from "../lib/types";
import { gpuColor, linear, log10Scale } from "./charts";

const W = 760;
const H = 420;
const M = { top: 16, right: 20, bottom: 52, left: 56 };

interface Props {
  rows: ConfigRow[];
  frontier: string[];
  /** quality of the reference (Qwen3-4B bf16) and the CI half-width, in percent */
  bandCenter: number;
  bandHalf: number;
}

export function FrontierChart({ rows, frontier, bandCenter, bandHalf }: Props) {
  const pts = rows.filter((r) => r.quality !== null && r.usd_per_1m_slo_tokens > 0);
  const [active, setActive] = useState<string | null>(null);
  const xs = pts.map((r) => r.usd_per_1m_slo_tokens);
  const x = log10Scale(Math.min(...xs) * 0.8, Math.max(...xs) * 1.25, M.left, W - M.right);
  const y = linear(76, 88, H - M.bottom, M.top, 6);
  const fset = new Set(frontier);
  const fpts = pts
    .filter((r) => fset.has(configKey(r.artifact, r.engine, r.gpu)))
    .sort((a, b) => a.usd_per_1m_slo_tokens - b.usd_per_1m_slo_tokens);
  const topQ = Math.max(...fpts.map((r) => r.quality ?? 0));
  const gpus = [...new Set(pts.map((r) => r.gpu))];
  const sel = pts.find((r) => configKey(r.artifact, r.engine, r.gpu) === active);
  const describe = (r: ConfigRow) =>
    `${artifactShort(r.artifact)}, ${engineLabel(r.engine)}, ${gpuLabel(r.gpu)}: ${usd(r.usd_per_1m_slo_tokens)} per 1M SLO-meeting tokens at ${r.best_level} users, quality ${((r.quality ?? 0) * 100).toFixed(1)}%`;

  return (
    <figure className="panel" style={{ margin: 0 }}>
      <figcaption>
        <h2 className="panel-title">Cost frontier (E14)</h2>
        <p className="small">
          $ per 1M output tokens that met the SLO (log scale) against quality (mean of GSM8K flexible and IFEval). The
          shaded band is ±{bandHalf} points around Qwen3-4B bf16: differences inside it are within the confidence
          intervals. Focus or hover a point for details.
        </p>
      </figcaption>
      <svg className="chart" viewBox={`0 0 ${W} ${H}`} role="img" aria-labelledby="fr-title fr-desc">
        <title id="fr-title">Cost per 1M SLO-meeting tokens against quality for every measured configuration</title>
        <desc id="fr-desc">
          {`Frontier: ${fpts.map(describe).join("; ")}.`}
        </desc>
        <rect className="band" x={M.left} width={W - M.left - M.right} y={y(bandCenter + bandHalf)} height={y(bandCenter - bandHalf) - y(bandCenter + bandHalf)} />
        {x.ticks.map((t) => (
          <g key={`x${t}`}>
            <line className="grid" x1={x(t)} x2={x(t)} y1={M.top} y2={H - M.bottom} />
            <text x={x(t)} y={H - M.bottom + 16} textAnchor="middle">
              {`$${t}`}
            </text>
          </g>
        ))}
        {y.ticks.map((t) => (
          <g key={`y${t}`}>
            <line className="grid" x1={M.left} x2={W - M.right} y1={y(t)} y2={y(t)} />
            <text x={M.left - 8} y={y(t) + 4} textAnchor="end">
              {t}
            </text>
          </g>
        ))}
        <text x={(M.left + W - M.right) / 2} y={H - 10} textAnchor="middle">
          $ per 1M SLO-meeting output tokens
        </text>
        <text transform={`translate(14 ${(M.top + H - M.bottom) / 2}) rotate(-90)`} textAnchor="middle">
          quality, %
        </text>
        <polyline
          points={fpts.map((r) => `${x(r.usd_per_1m_slo_tokens)},${y((r.quality ?? 0) * 100)}`).join(" ")}
          fill="none"
          stroke="var(--good)"
          strokeWidth={1.5}
          strokeDasharray="5 4"
        />
        {pts.map((r) => {
          const k = configKey(r.artifact, r.engine, r.gpu);
          const on = k === active;
          return (
            <circle
              key={k}
              cx={x(r.usd_per_1m_slo_tokens)}
              cy={y((r.quality ?? 0) * 100)}
              r={on ? 8 : fset.has(k) ? 6.5 : 5}
              fill={gpuColor(r.gpu)}
              stroke={on || fset.has(k) ? "var(--fg)" : "var(--surface)"}
              strokeWidth={on ? 2 : 1}
              tabIndex={0}
              role="img"
              aria-label={describe(r)}
              onMouseEnter={() => setActive(k)}
              onFocus={() => setActive(k)}
              onClick={() => setActive(k)}
            >
              <title>{describe(r)}</title>
            </circle>
          );
        })}
        {fpts.map((r) => (
          <text
            key={`l${configKey(r.artifact, r.engine, r.gpu)}`}
            className="label"
            x={x(r.usd_per_1m_slo_tokens) + 10}
            y={y((r.quality ?? 0) * 100) + ((r.quality ?? 0) < topQ ? 54 : -10)}
          >
            {`${artifactShort(r.artifact)} · ${gpuLabel(r.gpu)}`}
          </text>
        ))}
      </svg>
      <ul className="legend" aria-label="GPU colours">
        {gpus.map((g) => (
          <li key={g}>
            <span className="swatch" style={{ background: gpuColor(g) }} />
            {gpuLabel(g)}
          </li>
        ))}
        <li>
          <span className="swatch" style={{ background: "var(--good)", height: 2 }} />
          Pareto frontier
        </li>
      </ul>
      <p className="readout" aria-live="polite">
        {sel ? describe(sel) : "Select a point to read its numbers."}
      </p>
    </figure>
  );
}
