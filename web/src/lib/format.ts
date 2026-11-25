// Deterministic number formatting (identical on the build server and in the browser).

export function num(x: number, digits = 0): string {
  return x.toLocaleString("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits });
}

/** Round to `s` significant figures, grouped (e.g. 1,152 · 36.5 · 0.120). */
export function sig(x: number, s = 3): string {
  if (x === 0 || !Number.isFinite(x)) return String(x);
  const d = Math.max(0, s - 1 - Math.floor(Math.log10(Math.abs(x))));
  return num(x, Math.min(d, 6));
}

export function usd(x: number): string {
  return `$${x < 1 ? x.toFixed(3) : x.toFixed(2)}`;
}

/** Signed with a real minus sign. */
export function signed(x: number, digits = 1): string {
  const s = Math.abs(x).toFixed(digits);
  if (Number(s) === 0) return (0).toFixed(digits);
  return `${x < 0 ? "−" : "+"}${s}`;
}

/** Axis ticks: in full under 10,000, compact (10K, 12.5K) above. */
export function tick(x: number): string {
  if (Math.abs(x) >= 10000) return `${num(x / 1000, Number.isInteger(x / 1000) ? 0 : 1)}K`;
  if (Math.abs(x) >= 1 || x === 0) return num(x, Number.isInteger(x) ? 0 : 1);
  return sig(x, 2);
}
