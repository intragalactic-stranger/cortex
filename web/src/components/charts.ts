// Shared chart helpers (pure). Colours are CSS variables so charts follow the light/dark theme.

export const GPU_COLOR: Record<string, string> = {
  L4: "var(--gpu-l4)",
  H100: "var(--gpu-h100)",
  H200: "var(--gpu-h200)",
  H100x2: "var(--gpu-h100x2)",
  CPU: "var(--gpu-cpu)",
};

export const gpuColor = (gpu: string): string => GPU_COLOR[gpu] ?? "var(--muted)";

export interface Scale {
  (v: number): number;
  ticks: number[];
}

export function linear(d0: number, d1: number, r0: number, r1: number, nTicks = 5): Scale {
  const f = ((v: number) => r0 + ((v - d0) / (d1 - d0 || 1)) * (r1 - r0)) as Scale;
  const step = niceStep((d1 - d0) / nTicks);
  const ticks: number[] = [];
  for (let t = Math.ceil(d0 / step) * step; t <= d1 + 1e-9; t += step) ticks.push(Number(t.toPrecision(12)));
  f.ticks = ticks;
  return f;
}

export function log10Scale(d0: number, d1: number, r0: number, r1: number): Scale {
  const l0 = Math.log10(d0);
  const l1 = Math.log10(d1);
  const f = ((v: number) => r0 + ((Math.log10(v) - l0) / (l1 - l0 || 1)) * (r1 - r0)) as Scale;
  const ticks: number[] = [];
  for (let e = Math.floor(l0); e <= Math.ceil(l1); e++) {
    for (const m of [1, 2, 5]) {
      const t = m * 10 ** e;
      if (t >= d0 * 0.999 && t <= d1 * 1.001) ticks.push(t);
    }
  }
  f.ticks = ticks;
  return f;
}

function niceStep(raw: number): number {
  const p = 10 ** Math.floor(Math.log10(raw || 1));
  const m = raw / p;
  return (m <= 1 ? 1 : m <= 2 ? 2 : m <= 5 ? 5 : 10) * p;
}
