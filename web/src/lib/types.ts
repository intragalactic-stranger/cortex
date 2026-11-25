// Shapes of serve/results/web/*.json (written by serve/scripts/e14_frontier.py).

export interface ConfigRow {
  gpu: string;
  engine: string;
  artifact: string;
  best_level: number;
  goodput_tok_s: number;
  usd_per_1m_slo_tokens: number;
  quality: number | null;
  tpot_p50_ms_1user: number;
  run: string;
  stamp: string;
}

export interface CurvePoint {
  level: number;
  goodput_tok_s: number;
  output_throughput_tok_s: number;
  tpot_p50_s: number;
  ttft_p95_s: number;
}

export interface Curve {
  run: string;
  engine: string;
  artifact: string;
  gpu: string;
  workload: string;
  curve: CurvePoint[];
}

/** variant -> benchmark -> score (accuracies in 0..1; wikitext / bfsi_ppl are perplexities) */
export type Quality = Record<string, Record<string, number>>;

/** "artifact|engine|gpu" keys of the Pareto frontier */
export type Frontier = string[];

export interface Finding {
  title: string;
  /** inline HTML rendered from the README's markdown (repo-owned content) */
  bodyHtml: string;
}

export interface Summary {
  question: string;
  measured: string;
  findings: Finding[];
}
