// Node-only helpers that read repository files at build time (never shipped to the browser).
import { existsSync, readdirSync, readFileSync } from "node:fs";
import path from "node:path";
import { gunzipSync } from "node:zlib";
import { parse as parseYaml } from "yaml";
import { buildSpec } from "./memcalc/loader.ts";
import { parseDevices, type Device } from "./memcalc/devices.ts";
import type { ModelSpec } from "./memcalc/model.ts";
import type { TensorTable } from "./memcalc/params.ts";
import type { Config } from "./memcalc/py.ts";

/** Repository root: web/ lives one level below it. Override with CORTEX_REPO_ROOT. */
export const REPO_ROOT = process.env.CORTEX_REPO_ROOT ?? path.resolve(process.cwd(), "..");
export const repoPath = (...parts: string[]): string => path.join(REPO_ROOT, ...parts);

export const MODELS_DIR = repoPath("serve", "tests", "fixtures", "models");
export const DEVICES_YAML = repoPath("serve", "src", "cortex_serve", "memcalc", "devices.yaml");

export function readJson<T>(file: string): T {
  const buf = readFileSync(file);
  return JSON.parse((file.endsWith(".gz") ? gunzipSync(buf) : buf).toString("utf8")) as T;
}

export function readText(file: string): string | null {
  return existsSync(file) ? readFileSync(file, "utf8") : null;
}

/** Mirror of loader.load_local() for a fixture directory. */
export function loadFixtureModel(dirName: string): ModelSpec {
  const dir = path.join(MODELS_DIR, dirName);
  const cfg = readJson<Config>(path.join(dir, "config.json"));
  let tensors: TensorTable | null = null;
  for (const candidate of ["tensors.json.gz", "tensors.json"]) {
    if (existsSync(path.join(dir, candidate))) {
      tensors = readJson<TensorTable>(path.join(dir, candidate));
      break;
    }
  }
  const metaFile = path.join(dir, "meta.json");
  const meta = existsSync(metaFile) ? readJson<{ repo?: string; revision?: string | null }>(metaFile) : {};
  return buildSpec(String(meta.repo ?? dirName), cfg, tensors, meta.revision ?? null);
}

export function fixtureModelDirs(): string[] {
  return readdirSync(MODELS_DIR, { withFileTypes: true })
    .filter((d) => d.isDirectory())
    .map((d) => d.name)
    .sort();
}

export function loadDevices(): Device[] {
  return parseDevices(parseYaml(readFileSync(DEVICES_YAML, "utf8")));
}
