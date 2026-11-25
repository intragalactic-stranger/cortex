// Device type for the port of serve/src/cortex_serve/memcalc/devices.py. The table itself is read from
// devices.yaml at build time (src/lib/data.ts), so the site and the CLI share one source.

export type DeviceKind = "gpu" | "cpu" | "unified";

export interface Device {
  name: string;
  kind: DeviceKind;
  memory_gib: number;
  bandwidth_gbs: number | null;
  bf16_tflops: number | null;
  fp8_tflops: number | null;
  source_url: string;
}

export function memoryBytes(d: Device): number {
  return Math.trunc(d.memory_gib * 2 ** 30); // int(memory_gib * 2**30)
}

export function parseDevices(raw: unknown): Device[] {
  if (raw === null || typeof raw !== "object") throw new Error("devices.yaml: expected a mapping");
  return Object.entries(raw as Record<string, Record<string, unknown>>).map(([name, s]) => {
    const kind = s["kind"];
    if (kind !== "gpu" && kind !== "cpu" && kind !== "unified") throw new Error(`device ${name}: bad kind`);
    const num = (k: string): number | null => {
      const v = s[k];
      if (v === null || v === undefined) return null;
      if (typeof v !== "number") throw new Error(`device ${name}: ${k} must be a number`);
      return v;
    };
    const mem = num("memory_gib");
    if (mem === null) throw new Error(`device ${name}: memory_gib is required`);
    return {
      name,
      kind,
      memory_gib: mem,
      bandwidth_gbs: num("bandwidth_gbs"),
      bf16_tflops: num("bf16_tflops"),
      fp8_tflops: num("fp8_tflops"),
      source_url: String(s["source_url"] ?? ""),
    };
  });
}

export function getDevice(devices: readonly Device[], name: string): Device {
  const d = devices.find((x) => x.name.toLowerCase() === name.toLowerCase());
  if (!d) throw new Error(`unknown device '${name}'; choose from ${devices.map((x) => x.name).sort().join(", ")}`);
  return d;
}
