"use client";

import { useState } from "react";
import { signed } from "../lib/format";
import { qualityLabel } from "../lib/labels";
import type { Quality } from "../lib/types";

const COLS: { key: string; label: string; pct: boolean; higher: boolean }[] = [
  { key: "arc_challenge", label: "ARC", pct: true, higher: true },
  { key: "hellaswag", label: "HellaSwag", pct: true, higher: true },
  { key: "leaderboard_mmlu_pro", label: "MMLU-Pro", pct: true, higher: true },
  { key: "gsm8k", label: "GSM8K strict", pct: true, higher: true },
  { key: "gsm8k_flex", label: "GSM8K flex", pct: true, higher: true },
  { key: "ifeval", label: "IFEval", pct: true, higher: true },
  { key: "arc_cot", label: "ARC reasoning-first", pct: true, higher: true },
  { key: "wikitext", label: "WikiText PPL", pct: false, higher: false },
  { key: "bfsi_ppl", label: "Finance-text PPL", pct: false, higher: false },
];

const BASE: Record<string, string> = { qwen3: "qwen3-4b-bf16", gemma: "gemma4-e4b-bf16" };
const baseOf = (name: string): string | null =>
  name.startsWith("qwen3-4b") ? BASE["qwen3"]! : name.startsWith("gemma4-e4b") ? BASE["gemma"]! : null;
// Gemma 4 IT's loglikelihood benchmarks sit at chance even for bf16 (RESULTS §M3.6): not shown.
const INVALID = (name: string, key: string) =>
  name.startsWith("gemma4") && ["arc_challenge", "hellaswag", "leaderboard_mmlu_pro", "wikitext"].includes(key);

export function QualityTable({ quality }: { quality: Quality }) {
  const names = Object.keys(quality);
  const [sort, setSort] = useState<{ key: string; desc: boolean }>({ key: "", desc: true });
  const rows = [...names].sort((a, b) => {
    if (!sort.key) return 0;
    const va = quality[a]?.[sort.key] ?? Number.NEGATIVE_INFINITY;
    const vb = quality[b]?.[sort.key] ?? Number.NEGATIVE_INFINITY;
    return sort.desc ? vb - va : va - vb;
  });
  const cell = (name: string, key: string, pct: boolean, higher: boolean) => {
    const v = quality[name]?.[key];
    if (v === undefined || INVALID(name, key)) return <td key={key} className="n">—</td>;
    const base = baseOf(name);
    const b = base && base !== name ? quality[base]?.[key] : undefined;
    const shown = pct ? (v * 100).toFixed(1) : v.toFixed(2);
    let delta = null;
    if (b !== undefined) {
      const d = pct ? (v - b) * 100 : (v / b - 1) * 100;
      const better = higher ? d > 0 : d < 0;
      delta = (
        <span className={Math.abs(d) < 0.05 ? "" : better ? "up" : "down"}>
          {" "}
          ({signed(d, 1)}
          {pct ? "" : "%"})
        </span>
      );
    }
    return (
      <td key={key} className="n">
        {shown}
        {delta}
      </td>
    );
  };
  return (
    <div className="panel">
      <h2 className="panel-title">Quality by variant</h2>
      <p className="small">
        Same items for every variant (vLLM 0.32.0 + lm-evaluation-harness 0.4.13). Deltas are against each family&apos;s bf16
        model; accuracies in %, perplexities absolute (lower is better). CIs are about ±2–3 points. Click a header to sort.
      </p>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th scope="col">Variant</th>
              {COLS.map((c) => (
                <th
                  key={c.key}
                  scope="col"
                  className="n"
                  aria-sort={sort.key === c.key ? (sort.desc ? "descending" : "ascending") : "none"}
                >
                  <button
                    type="button"
                    className="sort-btn"
                    onClick={() => setSort((s) => ({ key: c.key, desc: s.key === c.key ? !s.desc : true }))}
                  >
                    {c.label}
                  </button>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((n) => (
              <tr key={n}>
                <th scope="row" style={{ textTransform: "none", letterSpacing: 0, color: "var(--fg)", background: "transparent", fontSize: "0.9rem", whiteSpace: "normal" }}>
                  {qualityLabel(n)}
                </th>
                {COLS.map((c) => cell(n, c.key, c.pct, c.higher))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
