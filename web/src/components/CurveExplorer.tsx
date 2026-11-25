"use client";

import { useMemo, useState } from "react";
import { num, tick } from "../lib/format";
import { artifactShort, engineLabel, gpuLabel } from "../lib/labels";
import type { Curve } from "../lib/types";
import { linear } from "./charts";

const W = 560;
const H = 320;
const M = { top: 14, right: 16, bottom: 46, left: 60 };
const PALETTE = ["#2f6fd0", "#c2410c", "#15803d", "#7c3aed", "#a16207", "#0e7490", "#be185d", "#475569", "#65a30d", "#9333ea"];

interface Props {
  curves: Curve[];
}

export function CurveExplorer({ curves }: Props) {
  const gpus = useMemo(() => [...new Set(curves.map((c) => c.gpu))], [curves]);
  const [gpu, setGpu] = useState<string>(gpus.includes("L4") ? "L4" : (gpus[0] ?? ""));
  const engines = [...new Set(curves.filter((c) => c.gpu === gpu).map((c) => c.engine))];
  const [picked, setPicked] = useState<string[]>(["vllm"]);
  const active = engines.filter((e) => picked.includes(e));
  const useEngines = active.length > 0 ? active : engines.slice(0, 1);
  const toggle = (e: string) => setPicked((p) => (p.includes(e) ? p.filter((v) => v !== e) : [...p, e]));
  const shown = curves.filter((c) => c.gpu === gpu && useEngines.includes(c.engine) && c.curve.length > 0);
  const label = (c: Curve) => `${artifactShort(c.artifact)} · ${engineLabel(c.engine)}`;
  const color = (i: number) => PALETTE[i % PALETTE.length] ?? "var(--muted)";

  const levels = shown.flatMap((c) => c.curve.map((p) => p.level));
  const thr = shown.flatMap((c) => c.curve.map((p) => p.output_throughput_tok_s));
  const tpot = shown.flatMap((c) => c.curve.map((p) => p.tpot_p50_s * 1000));
  const x1 = linear(0, Math.max(1, ...levels), M.left, W - M.right);
  const y1 = linear(0, Math.max(1, ...thr) * 1.05, H - M.bottom, M.top);
  const x2 = linear(0, Math.max(1, ...thr) * 1.05, M.left, W - M.right);
  const y2 = linear(0, Math.max(1, ...tpot) * 1.1, H - M.bottom, M.top);
  const slo = gpu === "CPU" ? 125 : 50;

  const axes = (x: typeof x1, y: typeof y1, xl: string, yl: string) => (
    <>
      {x.ticks.map((t) => (
        <g key={`x${t}`}>
          <line className="grid" x1={x(t)} x2={x(t)} y1={M.top} y2={H - M.bottom} />
          <text x={x(t)} y={H - M.bottom + 16} textAnchor="middle">
            {tick(t)}
          </text>
        </g>
      ))}
      {y.ticks.map((t) => (
        <g key={`y${t}`}>
          <line className="grid" x1={M.left} x2={W - M.right} y1={y(t)} y2={y(t)} />
          <text x={M.left - 8} y={y(t) + 4} textAnchor="end">
            {tick(t)}
          </text>
        </g>
      ))}
      <text x={(M.left + W - M.right) / 2} y={H - 8} textAnchor="middle">
        {xl}
      </text>
      <text transform={`translate(14 ${(M.top + H - M.bottom) / 2}) rotate(-90)`} textAnchor="middle">
        {yl}
      </text>
    </>
  );

  return (
    <div className="panel">
      <h2 className="panel-title">Throughput and latency by configuration</h2>
      <div className="controls">
        <div className="field" role="group" aria-labelledby="gpu-pick">
          <span id="gpu-pick" className="small">
            Hardware
          </span>
          <div className="chips">
            {gpus.map((g) => (
              <button key={g} type="button" className="chip" aria-pressed={g === gpu} onClick={() => setGpu(g)}>
                {gpuLabel(g)}
              </button>
            ))}
          </div>
        </div>
        <div className="field" role="group" aria-labelledby="eng-pick">
          <span id="eng-pick" className="small">
            Engines
          </span>
          <div className="chips">
            {engines.map((e) => (
              <button key={e} type="button" className="chip" aria-pressed={useEngines.includes(e)} onClick={() => toggle(e)}>
                {engineLabel(e)}
              </button>
            ))}
          </div>
        </div>
      </div>
      <p className="small">
        {gpu === "CPU"
          ? "CPU tier (M1): Qwen3 / Gemma GGUF files on 4 vCPUs, chat 128 in / 64 out, 1 and 4 users."
          : "Qwen3-4B-Instruct-2507 and its quantizations (plus the E15 architectures and the MoE), chat 512 in / 256 out. Each line is the best measured run for that engine and model."}
      </p>
      <div className="two-col">
        <svg className="chart" viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Output throughput against concurrent users on ${gpuLabel(gpu)}`}>
          {axes(x1, y1, "concurrent users", "output tokens/s")}
          {shown.map((c, i) => (
            <g key={label(c)}>
              <polyline
                points={c.curve.map((p) => `${x1(p.level)},${y1(p.output_throughput_tok_s)}`).join(" ")}
                fill="none"
                stroke={color(i)}
                strokeWidth={2}
              />
              {c.curve.map((p) => (
                <circle key={p.level} cx={x1(p.level)} cy={y1(p.output_throughput_tok_s)} r={3} fill={color(i)}>
                  <title>{`${label(c)}: ${num(p.output_throughput_tok_s)} tok/s at ${p.level} users`}</title>
                </circle>
              ))}
            </g>
          ))}
        </svg>
        <svg className="chart" viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Time per output token against throughput on ${gpuLabel(gpu)}`}>
          {axes(x2, y2, "output tokens/s", "ms per output token (p50)")}
          <line x1={M.left} x2={W - M.right} y1={y2(slo)} y2={y2(slo)} stroke="var(--bad)" strokeDasharray="4 4" />
          <text x={M.left + 6} y={y2(slo) - 6} textAnchor="start">{`SLO ${slo} ms`}</text>
          {shown.map((c, i) => (
            <g key={label(c)}>
              <polyline
                points={c.curve.map((p) => `${x2(p.output_throughput_tok_s)},${y2(p.tpot_p50_s * 1000)}`).join(" ")}
                fill="none"
                stroke={color(i)}
                strokeWidth={2}
              />
              {c.curve.map((p) => (
                <circle key={p.level} cx={x2(p.output_throughput_tok_s)} cy={y2(p.tpot_p50_s * 1000)} r={3} fill={color(i)}>
                  <title>{`${label(c)}: ${num(p.tpot_p50_s * 1000, 1)} ms/token at ${num(p.output_throughput_tok_s)} tok/s (${p.level} users)`}</title>
                </circle>
              ))}
            </g>
          ))}
        </svg>
      </div>
      <ul className="legend" aria-label="Lines">
        {shown.map((c, i) => (
          <li key={label(c)}>
            <span className="swatch" style={{ background: color(i) }} />
            {label(c)}
          </li>
        ))}
      </ul>
    </div>
  );
}
