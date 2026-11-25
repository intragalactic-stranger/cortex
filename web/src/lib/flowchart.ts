// A small, build-time renderer for Mermaid `flowchart`/`graph` blocks: parse nodes and edges, lay them out
// in layers (longest-path ranking, dummy nodes on long edges, barycenter ordering) and emit a static SVG.
// It covers the subset our docs use (node shapes [], {}, (), ([]), (()), edges -->, ---, -.->, ==>, with
// labels as `-- text -->` or `-->|text|`). Anything it cannot parse returns null and the caller shows the
// source as a code block instead. No JavaScript ships to the browser for diagrams.

export type Shape = "rect" | "round" | "decision" | "stadium" | "circle";

export interface FNode {
  id: string;
  lines: string[];
  shape: Shape;
}

export interface FEdge {
  from: string;
  to: string;
  label: string;
  dotted: boolean;
}

export interface Flowchart {
  direction: "TD" | "LR";
  nodes: FNode[];
  edges: FEdge[];
}

const NODE_RE = /^([A-Za-z0-9_]+)\s*(\(\[.*?\]\)|\(\(.*?\)\)|\[\[.*?\]\]|\[\(.*?\)\]|\[.*?\]|\{.*?\}|\(.*?\))?/;
const EDGE_RES: Array<[RegExp, boolean]> = [
  [/^-->\s*(?:\|([^|]*)\|)?/, false],
  [/^---\s*(?:\|([^|]*)\|)?/, false],
  [/^==>\s*(?:\|([^|]*)\|)?/, false],
  [/^-\.->\s*(?:\|([^|]*)\|)?/, true],
  [/^--\s+(.+?)\s+-->/, false],
  [/^==\s+(.+?)\s+==>/, false],
  [/^-\.\s+(.+?)\s+\.->/, true],
];

function decodeLabel(raw: string): string[] {
  let s = raw.trim();
  if (s.startsWith('"') && s.endsWith('"')) s = s.slice(1, -1);
  return s
    .split(/<br\s*\/?>/i)
    .map((x) => x.replace(/&lt;/g, "<").replace(/&gt;/g, ">").replace(/&amp;/g, "&").trim())
    .filter(Boolean);
}

function shapeOf(token: string): [Shape, string] {
  if (token.startsWith("((")) return ["circle", token.slice(2, -2)];
  if (token.startsWith("([")) return ["stadium", token.slice(2, -2)];
  if (token.startsWith("[[") || token.startsWith("[(")) return ["rect", token.slice(2, -2)];
  if (token.startsWith("[")) return ["rect", token.slice(1, -1)];
  if (token.startsWith("{")) return ["decision", token.slice(1, -1)];
  return ["round", token.slice(1, -1)];
}

export function parseFlowchart(src: string): Flowchart | null {
  const lines = src
    .split("\n")
    .map((l) => l.replace(/%%.*$/, "").trim())
    .filter(Boolean);
  const head = /^(flowchart|graph)\s+(TD|TB|LR|BT|RL)\s*$/.exec(lines[0] ?? "");
  if (!head) return null;
  const direction = head[2] === "LR" || head[2] === "RL" ? "LR" : "TD";
  const nodes = new Map<string, FNode>();
  const edges: FEdge[] = [];
  const touch = (id: string, token?: string): void => {
    const existing = nodes.get(id);
    if (token) {
      const [shape, text] = shapeOf(token);
      nodes.set(id, { id, shape, lines: decodeLabel(text) });
    } else if (!existing) {
      nodes.set(id, { id, shape: "rect", lines: [id] });
    }
  };
  for (const line of lines.slice(1)) {
    if (/^(subgraph|end$|classDef|class |style |linkStyle|click )/.test(line)) continue;
    let rest = line.replace(/;$/, "");
    let m = NODE_RE.exec(rest);
    if (!m) return null;
    let prev = m[1]!;
    touch(prev, m[2]);
    rest = rest.slice(m[0].length).trim();
    while (rest) {
      let matched = false;
      for (const [re, dotted] of EDGE_RES) {
        const e = re.exec(rest);
        if (!e) continue;
        rest = rest.slice(e[0].length).trim();
        m = NODE_RE.exec(rest);
        if (!m) return null;
        const to = m[1]!;
        touch(to, m[2]);
        edges.push({ from: prev, to, label: (e[1] ?? "").trim(), dotted });
        prev = to;
        rest = rest.slice(m[0].length).trim();
        matched = true;
        break;
      }
      if (!matched) return null;
    }
  }
  if (nodes.size === 0) return null;
  return { direction, nodes: [...nodes.values()], edges };
}

// ------------------------------------------------------------------------------------------------ layout

const FONT = 13;
const LINE_H = 17;
const CHAR_W = 6.9; // average glyph width of the UI sans at 13px
const PAD_X = 14;
const PAD_Y = 10;
const GAP_X = 28;
const GAP_Y = 46;

interface Box {
  id: string;
  w: number;
  h: number;
  x: number; // center
  y: number; // center
  layer: number;
  dummy: boolean;
  node?: FNode;
}

const WRAP = 30; // characters per drawn line

/** The lines drawn for a node: each label segment word-wrapped; the first segment is bold if there are several. */
export function drawnLines(n: FNode): { text: string; strong: boolean }[] {
  const out: { text: string; strong: boolean }[] = [];
  n.lines.forEach((seg, si) => {
    let cur = "";
    for (const word of seg.split(/\s+/)) {
      if (cur && cur.length + 1 + word.length > WRAP) {
        out.push({ text: cur, strong: si === 0 && n.lines.length > 1 });
        cur = word;
      } else cur = cur ? `${cur} ${word}` : word;
    }
    if (cur) out.push({ text: cur, strong: si === 0 && n.lines.length > 1 });
  });
  return out;
}

function size(n: FNode): [number, number] {
  const drawn = drawnLines(n);
  const longest = Math.max(...drawn.map((l) => l.text.length));
  let w = Math.max(64, longest * CHAR_W + 2 * PAD_X);
  let h = drawn.length * LINE_H + 2 * PAD_Y;
  if (n.shape === "decision") {
    w += 28;
    h += 6;
  }
  if (n.shape === "circle") w = h = Math.max(w, h);
  return [Math.round(w), Math.round(h)];
}

export interface Layout {
  width: number;
  height: number;
  boxes: Box[];
  paths: { edge: FEdge; points: Array<[number, number]> }[];
}

export function layout(fc: Flowchart): Layout {
  const ids = fc.nodes.map((n) => n.id);
  const index = new Map(ids.map((id, i) => [id, i]));
  // Drop self-loops; reverse back edges found by DFS so the graph is acyclic.
  const edges = fc.edges.filter((e) => e.from !== e.to);
  const out = new Map<string, string[]>(ids.map((id) => [id, []]));
  for (const e of edges) out.get(e.from)!.push(e.to);
  const state = new Map<string, number>();
  const reversed = new Set<FEdge>();
  const dfs = (v: string): void => {
    state.set(v, 1);
    for (const e of edges.filter((x) => x.from === v)) {
      const s = state.get(e.to) ?? 0;
      if (s === 1) reversed.add(e);
      else if (s === 0) dfs(e.to);
    }
    state.set(v, 2);
  };
  for (const id of ids) if (!state.get(id)) dfs(id);
  const dag = edges.map((e) => (reversed.has(e) ? { u: e.to, v: e.from, e } : { u: e.from, v: e.to, e }));

  // Longest-path layering.
  const layer = new Map<string, number>(ids.map((id) => [id, 0]));
  for (let pass = 0; pass < ids.length; pass++) {
    let changed = false;
    for (const { u, v } of dag) {
      if (layer.get(v)! < layer.get(u)! + 1) {
        layer.set(v, layer.get(u)! + 1);
        changed = true;
      }
    }
    if (!changed) break;
  }

  // Boxes, with dummy nodes on edges that span more than one layer.
  const boxes = new Map<string, Box>();
  for (const n of fc.nodes) {
    // LR charts are laid out top-down with each box's width and height swapped, then transposed.
    const [w0, h0] = size(n);
    const [w, h] = fc.direction === "LR" ? [h0, w0] : [w0, h0];
    boxes.set(n.id, { id: n.id, w, h, x: 0, y: 0, layer: layer.get(n.id)!, dummy: false, node: n });
  }
  const chains: { edge: FEdge; ids: string[]; reversed: boolean }[] = [];
  let dummyN = 0;
  for (const { u, v, e } of dag) {
    const chain = [u];
    for (let l = layer.get(u)! + 1; l < layer.get(v)!; l++) {
      const id = `__d${dummyN++}`;
      boxes.set(id, { id, w: 8, h: LINE_H, x: 0, y: 0, layer: l, dummy: true });
      chain.push(id);
    }
    chain.push(v);
    chains.push({ edge: e, ids: chain, reversed: reversed.has(e) });
  }
  const nLayers = Math.max(...[...boxes.values()].map((b) => b.layer)) + 1;
  const layers: string[][] = Array.from({ length: nLayers }, () => []);
  // Initial order: declaration order, dummies after their chain's source position.
  const order = (id: string): number => index.get(id) ?? ids.length + Number(id.slice(3));
  for (const b of [...boxes.values()].sort((a, b2) => order(a.id) - order(b2.id))) layers[b.layer]!.push(b.id);

  const up = new Map<string, string[]>();
  const down = new Map<string, string[]>();
  for (const c of chains) {
    for (let i = 0; i + 1 < c.ids.length; i++) {
      const a = c.ids[i]!;
      const b = c.ids[i + 1]!;
      down.set(a, [...(down.get(a) ?? []), b]);
      up.set(b, [...(up.get(b) ?? []), a]);
    }
  }
  const pos = (): Map<string, number> => {
    const p = new Map<string, number>();
    for (const l of layers) l.forEach((id, i) => p.set(id, i));
    return p;
  };
  for (let iter = 0; iter < 12; iter++) {
    const downward = iter % 2 === 0;
    const range = downward ? [...Array(nLayers).keys()].slice(1) : [...Array(nLayers).keys()].reverse().slice(1);
    for (const l of range) {
      const p = pos();
      const nbr = downward ? up : down;
      const bary = new Map<string, number>();
      layers[l]!.forEach((id, i) => {
        const ns = nbr.get(id) ?? [];
        bary.set(id, ns.length ? ns.reduce((s, n) => s + p.get(n)!, 0) / ns.length : i);
      });
      layers[l]!.sort((a, b) => bary.get(a)! - bary.get(b)! || p.get(a)! - p.get(b)!);
    }
  }

  // Vertical placement.
  let y = 0;
  for (const l of layers) {
    const h = Math.max(...l.map((id) => boxes.get(id)!.h));
    for (const id of l) boxes.get(id)!.y = y + h / 2;
    y += h + GAP_Y;
  }
  // Horizontal placement: pack each layer, then pull nodes toward their neighbours' mean x (a few sweeps),
  // resolving overlaps left to right.
  for (const l of layers) {
    let x = 0;
    for (const id of l) {
      const b = boxes.get(id)!;
      b.x = x + b.w / 2;
      x += b.w + GAP_X;
    }
  }
  const place = (l: string[], nbr: Map<string, string[]>): void => {
    const want = l.map((id) => {
      const ns = nbr.get(id) ?? [];
      return ns.length ? ns.reduce((s, n) => s + boxes.get(n)!.x, 0) / ns.length : boxes.get(id)!.x;
    });
    // left-to-right: no overlap with the previous box
    let minX = -Infinity;
    l.forEach((id, i) => {
      const b = boxes.get(id)!;
      b.x = Math.max(want[i]!, minX + b.w / 2);
      minX = b.x + b.w / 2 + GAP_X;
    });
    // right-to-left: shift back toward the target where there is room
    let maxX = Infinity;
    for (let i = l.length - 1; i >= 0; i--) {
      const b = boxes.get(l[i]!)!;
      b.x = Math.min(b.x, maxX - b.w / 2);
      maxX = b.x - b.w / 2 - GAP_X;
    }
  };
  for (let iter = 0; iter < 8; iter++) {
    if (iter % 2 === 0) for (const l of layers.slice(1)) place(l, up);
    else for (const l of [...layers].reverse().slice(1)) place(l, down);
  }
  // Pull every node toward all of its neighbours (above and below) so leaves and branch points do not drift
  // apart; this keeps decision trees compact.
  const both = new Map<string, string[]>(
    [...boxes.keys()].map((id) => [id, [...(up.get(id) ?? []), ...(down.get(id) ?? [])]]),
  );
  for (let iter = 0; iter < 10; iter++) {
    for (const l of iter % 2 === 0 ? layers : [...layers].reverse()) place(l, both);
  }
  const all = [...boxes.values()];
  const minX = Math.min(...all.map((b) => b.x - b.w / 2));
  const margin = 8;
  for (const b of all) b.x += margin - minX;
  const width = Math.max(...all.map((b) => b.x + b.w / 2)) + margin;
  const height = y - GAP_Y + 2 * margin;
  for (const b of all) b.y += margin;

  const paths = chains.map((c) => {
    const pts = c.ids.map((id) => boxes.get(id)!);
    const points: Array<[number, number]> = pts.map((b, i) => {
      if (i === 0) return [b.x, b.y + b.h / 2];
      if (i === pts.length - 1) return [b.x, b.y - b.h / 2];
      return [b.x, b.y];
    });
    if (c.reversed) points.reverse();
    return { edge: c.edge, points };
  });
  if (fc.direction === "LR") {
    for (const b of all) [b.x, b.y, b.w, b.h] = [b.y, b.x, b.h, b.w];
    for (const p of paths) p.points = p.points.map(([px, py]) => [py, px]);
    return { width: height, height: width, boxes: all.filter((b) => !b.dummy), paths };
  }
  return { width, height, boxes: all.filter((b) => !b.dummy), paths };
}

// ------------------------------------------------------------------------------------------------ SVG

const esc = (s: string): string =>
  s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");

function shapeSvg(b: Box): string {
  const x0 = b.x - b.w / 2;
  const y0 = b.y - b.h / 2;
  const shape = b.node!.shape;
  if (shape === "decision") {
    const k = 14;
    const pts = [
      [x0, b.y],
      [x0 + k, y0],
      [x0 + b.w - k, y0],
      [x0 + b.w, b.y],
      [x0 + b.w - k, y0 + b.h],
      [x0 + k, y0 + b.h],
    ];
    return `<polygon class="fc-node fc-decision" points="${pts.map((p) => p.join(",")).join(" ")}"/>`;
  }
  if (shape === "circle") return `<circle class="fc-node" cx="${b.x}" cy="${b.y}" r="${b.w / 2}"/>`;
  const r = shape === "stadium" ? b.h / 2 : shape === "round" ? 10 : 4;
  return `<rect class="fc-node${b.layer === 0 ? " fc-start" : ""}" x="${x0}" y="${y0}" width="${b.w}" height="${b.h}" rx="${r}"/>`;
}

function pathD(points: Array<[number, number]>, lr: boolean): string {
  let d = `M${points[0]![0]},${points[0]![1]}`;
  for (let i = 1; i < points.length; i++) {
    const [x0, y0] = points[i - 1]!;
    const [x1, y1] = points[i]!;
    if (lr) {
      const mx = (x0 + x1) / 2;
      d += ` C${mx},${y0} ${mx},${y1} ${x1},${y1}`;
    } else {
      const my = (y0 + y1) / 2;
      d += ` C${x0},${my} ${x1},${my} ${x1},${y1}`;
    }
  }
  return d;
}

/** Render a Mermaid flowchart to an accessible SVG string, or null if unsupported. */
export function flowchartSvg(src: string, title: string): { svg: string; outline: string } | null {
  const fc = parseFlowchart(src);
  if (!fc) return null;
  const lay = layout(fc);
  const id = `fc-${Math.abs([...src].reduce((h, c) => (h * 31 + c.charCodeAt(0)) | 0, 7)).toString(36)}`;
  const parts: string[] = [];
  parts.push(
    `<svg class="flowchart" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${Math.ceil(lay.width)} ${Math.ceil(lay.height)}" width="${Math.ceil(lay.width)}" height="${Math.ceil(lay.height)}" role="img" aria-labelledby="${id}-t" font-size="${FONT}">`,
    `<title id="${id}-t">${esc(title)}</title>`,
    `<defs><marker id="${id}-a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fc-arrow"/></marker></defs>`,
  );
  for (const p of lay.paths) {
    parts.push(
      `<path class="fc-edge${p.edge.dotted ? " fc-dotted" : ""}" d="${pathD(p.points, fc.direction === "LR")}" marker-end="url(#${id}-a)"/>`,
    );
  }
  for (const b of lay.boxes) {
    parts.push(`<g>${shapeSvg(b)}`);
    const lines = drawnLines(b.node!);
    const top = b.y - ((lines.length - 1) * LINE_H) / 2;
    lines.forEach((l, i) => {
      parts.push(
        `<text class="fc-text${l.strong ? " fc-strong" : ""}" x="${b.x}" y="${top + i * LINE_H}" text-anchor="middle" dominant-baseline="central">${esc(l.text)}</text>`,
      );
    });
    parts.push("</g>");
  }
  // Edge labels last so they sit above the lines, near the source end.
  for (const p of lay.paths) {
    if (!p.edge.label) continue;
    const [x0, y0] = p.points[0]!;
    const [x1, y1] = p.points[1]!;
    const t = p.points.length === 2 ? 0.62 : 0.5; // direct edges: nearer the target, so fan-out labels spread
    const lx = x0 + (x1 - x0) * t;
    const ly = y0 + (y1 - y0) * t;
    const w = p.edge.label.length * CHAR_W * 0.92 + 12;
    parts.push(
      `<rect class="fc-label-bg" x="${lx - w / 2}" y="${ly - 9}" width="${w}" height="18" rx="9"/>`,
      `<text class="fc-label" x="${lx}" y="${ly}" text-anchor="middle" dominant-baseline="central">${esc(p.edge.label)}</text>`,
    );
  }
  parts.push("</svg>");

  // Text outline of the same graph, for screen readers and narrow screens.
  const label = (nid: string): string => fc.nodes.find((n) => n.id === nid)!.lines.join(" · ");
  const outline = fc.nodes
    .map((n) => {
      const outs = fc.edges.filter((e) => e.from === n.id);
      if (!outs.length) return "";
      const items = outs
        .map((e) => `<li>${e.label ? `<strong>${esc(e.label)}</strong> → ` : "→ "}${esc(label(e.to))}</li>`)
        .join("");
      return `<li><span>${esc(n.lines.join(" · "))}</span><ul>${items}</ul></li>`;
    })
    .filter(Boolean)
    .join("");
  return { svg: parts.join(""), outline: `<ol class="fc-outline">${outline}</ol>` };
}
