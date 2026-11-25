// Helpers that reproduce the Python semantics the memcalc port depends on.

export type Json = null | boolean | number | string | Json[] | { [key: string]: Json };
export type Config = { [key: string]: Json };

/** Python truthiness. */
export function truthy(v: unknown): boolean {
  if (v === null || v === undefined || v === false || v === 0 || v === "") return false;
  if (Array.isArray(v)) return v.length > 0;
  if (typeof v === "object") return Object.keys(v).length > 0;
  return true;
}

/** Python `a or b`. */
export function or(...vals: unknown[]): unknown {
  for (const v of vals) if (truthy(v)) return v;
  return vals[vals.length - 1];
}

/** Python int(): truncates floats, parses numeric strings, raises otherwise. */
export function int(v: unknown, what = "value"): number {
  if (typeof v === "boolean") return v ? 1 : 0;
  const n = typeof v === "number" ? v : typeof v === "string" && v.trim() !== "" ? Number(v) : NaN;
  if (!Number.isFinite(n)) throw new Error(`expected an integer for ${what}, got ${JSON.stringify(v)}`);
  return Math.trunc(n);
}

export function get(cfg: Config, key: string): Json | undefined {
  return Object.prototype.hasOwnProperty.call(cfg, key) ? cfg[key] : undefined;
}

export function req(cfg: Config, key: string): Json {
  const v = get(cfg, key);
  if (v === undefined) throw new Error(`config.json is missing '${key}'`);
  return v;
}
