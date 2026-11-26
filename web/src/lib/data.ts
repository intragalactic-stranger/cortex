// Build-time data access. Everything the site shows is read here from files committed in the repository:
//   serve/results/web/*.json          measured results (written by serve/scripts/e14_frontier.py)
//   serve/tests/fixtures/models/*     model configs + safetensors header tables (memcalc presets)
//   serve/src/.../memcalc/devices.yaml device table
//   README.md, serve/docs/*.md        headline findings and the decision guide
// Server components call these during `next build`; nothing is fetched at runtime (node:fs makes any
// accidental client import fail the build).
import { existsSync, readdirSync } from "node:fs";
import path from "node:path";
import { marked } from "marked";
import type { Device } from "./memcalc/devices.ts";
import type { ModelSpec } from "./memcalc/model.ts";
import { fixtureModelDirs, loadDevices, loadFixtureModel, readJson, readText, repoPath } from "./repo.ts";
import type { ConfigRow, Curve, Frontier, Quality, Summary } from "./types.ts";

const WEB = (f: string): string => repoPath("serve", "results", "web", f);

export const loadConfigs = (): ConfigRow[] => readJson<ConfigRow[]>(WEB("configs.json"));
export const loadCurves = (): Curve[] => readJson<Curve[]>(WEB("curves.json"));
export const loadQuality = (): Quality => readJson<Quality>(WEB("quality.json"));
export const loadFrontier = (): Frontier => readJson<Frontier>(WEB("frontier.json"));

export { loadDevices };

export interface Preset {
  id: string;
  label: string;
  note: string;
  spec: ModelSpec;
}

const PRESET_ORDER: Record<string, [string, string]> = {
  "Qwen__Qwen3-0.6B": ["Qwen3-0.6B", "dense, 28 layers"],
  "Qwen__Qwen3-1.7B": ["Qwen3-1.7B", "dense, 28 layers"],
  "Qwen__Qwen3-4B-Instruct-2507": ["Qwen3-4B-Instruct-2507", "dense GQA, the main model of the lab"],
  "Qwen__Qwen3.5-4B": ["Qwen3.5-4B", "hybrid: Gated DeltaNet + full attention"],
  "google__gemma-4-E2B-it": ["Gemma 4 E2B", "sliding window + KV sharing + per-layer embeddings"],
  "google__gemma-4-E4B-it": ["Gemma 4 E4B", "sliding window + KV sharing + per-layer embeddings"],
  "Qwen__Qwen3-30B-A3B": ["Qwen3-30B-A3B", "sparse MoE, 128 experts, 8 active"],
  "synthetic__llama-3.1-8b": ["Llama-3.1-8B (synthetic config)", "analytic parameter count; gated repo not fetched"],
};

export function loadPresets(): Preset[] {
  const dirs = fixtureModelDirs();
  const ordered = [...Object.keys(PRESET_ORDER).filter((d) => dirs.includes(d)), ...dirs.filter((d) => !(d in PRESET_ORDER))];
  return ordered.map((dir) => {
    const [label, note] = PRESET_ORDER[dir] ?? [dir.replace("__", "/"), ""];
    return { id: dir, label, note, spec: loadFixtureModel(dir) };
  });
}

export function loadDeviceTable(): Device[] {
  return loadDevices();
}

/** Headline question, the "measured on" line and the six findings from the root README. */
export function loadSummary(): Summary {
  const readme = readText(repoPath("README.md")) ?? "";
  const start = readme.indexOf("## Cortex Serve in one screen");
  if (start < 0) throw new Error("README.md: section 'Cortex Serve in one screen' not found");
  const rest = readme.slice(start).split("\n").slice(1);
  const end = rest.findIndex((l) => l.startsWith("## "));
  const lines = end < 0 ? rest : rest.slice(0, end);
  const question = lines
    .filter((l) => l.startsWith(">"))
    .map((l) => l.replace(/^>\s?/, ""))
    .join(" ")
    .replace(/\*\*Question:\*\*\s*/, "")
    .trim();
  const measured = (lines.find((l) => l.trim() && !l.startsWith(">") && !/^\d+\./.test(l)) ?? "").replace(/:\s*$/, ".");
  const findings = lines
    .filter((l) => /^\d+\.\s/.test(l))
    .map((l) => {
      const item = l.replace(/^\d+\.\s+/, "");
      const m = /^\*\*(.+?)\*\*\s*(.*)$/.exec(item);
      const title = (m ? m[1]! : item).replace(/\.$/, "");
      const body = m ? m[2]! : "";
      return { title, bodyHtml: marked.parseInline(body, { async: false }) };
    });
  if (findings.length < 6) throw new Error(`README.md: expected six findings, found ${findings.length}`);
  return { question, measured: marked.parseInline(measured, { async: false }), findings };
}

export const GUIDE_PATH = "serve/docs/DECISION_GUIDE.md";

/** The decision guide markdown, or null while it does not exist yet. */
export function loadGuideMarkdown(): string | null {
  return readText(repoPath(GUIDE_PATH));
}

/** Hugging Face model repos published from the model cards (public since 2026-11-26). */
export function loadHfModelNames(): string[] {
  const dir = repoPath("serve", "results", "hf_cards");
  if (!existsSync(dir)) return [];
  return readdirSync(dir)
    .filter((f) => f.endsWith(".md"))
    .map((f) => `intragalactic-stranger/${path.basename(f, ".md")}`)
    .sort();
}
