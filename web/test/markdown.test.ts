// The decision guide is rendered at build time; its Mermaid flowchart becomes a static SVG.
import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { flowchartSvg, layout, parseFlowchart } from "../src/lib/flowchart.ts";
import { renderMarkdown, resolveRepoLink } from "../src/lib/markdown.ts";
import { readText, repoPath } from "../src/lib/repo.ts";

const SAMPLE = `flowchart TD
    A[Start] --> B{Enough?}
    B -- no --> C[Bigger model<br/>on an H100]
    B -- yes --> D[Keep it]
    C --> E[Done]
    D -.-> E
    A -->|skip| E`;

describe("flowchart", () => {
  it("parses nodes, shapes and labelled edges", () => {
    const fc = parseFlowchart(SAMPLE);
    assert.ok(fc);
    assert.equal(fc.nodes.length, 5);
    assert.equal(fc.nodes.find((n) => n.id === "B")!.shape, "decision");
    assert.deepEqual(fc.nodes.find((n) => n.id === "C")!.lines, ["Bigger model", "on an H100"]);
    assert.deepEqual(
      fc.edges.map((e) => [e.from, e.to, e.label, e.dotted]),
      [
        ["A", "B", "", false],
        ["B", "C", "no", false],
        ["B", "D", "yes", false],
        ["C", "E", "", false],
        ["D", "E", "", true],
        ["A", "E", "skip", false],
      ],
    );
  });

  it("lays out without overlapping boxes in a layer", () => {
    const lay = layout(parseFlowchart(SAMPLE)!);
    const byY = new Map<number, typeof lay.boxes>();
    for (const b of lay.boxes) byY.set(b.y, [...(byY.get(b.y) ?? []), b]);
    for (const row of byY.values()) {
      const sorted = [...row].sort((a, b) => a.x - b.x);
      for (let i = 1; i < sorted.length; i++) {
        assert.ok(sorted[i]!.x - sorted[i]!.w / 2 >= sorted[i - 1]!.x + sorted[i - 1]!.w / 2, "overlap");
      }
    }
    assert.equal(lay.paths.length, 6);
  });

  it("returns null for unsupported input", () => {
    assert.equal(parseFlowchart("sequenceDiagram\n A->>B: hi"), null);
    assert.equal(flowchartSvg("flowchart TD\n A[x] ~~~ B[y]", "t"), null);
  });
});

describe("markdown", () => {
  it("rewrites relative links to GitHub", () => {
    assert.equal(
      resolveRepoLink("../RESULTS.md#m7", "serve/docs/DECISION_GUIDE.md"),
      "https://github.com/intragalactic-stranger/cortex/blob/main/serve/RESULTS.md#m7",
    );
    assert.equal(
      resolveRepoLink("adr/", "serve/docs/DECISION_GUIDE.md"),
      "https://github.com/intragalactic-stranger/cortex/tree/main/serve/docs/adr",
    );
    assert.equal(resolveRepoLink("https://example.org/x", "a/b.md"), "https://example.org/x");
  });

  it("renders the decision guide (if present) with every flowchart as SVG", () => {
    const src = readText(repoPath("serve", "docs", "DECISION_GUIDE.md"));
    if (src === null) return; // placeholder page until the guide exists
    const { html, headings } = renderMarkdown(src, "serve/docs/DECISION_GUIDE.md");
    const blocks = (src.match(/```mermaid/g) ?? []).length;
    assert.equal((html.match(/<svg class="flowchart"/g) ?? []).length, blocks);
    assert.ok(headings.length > 3);
    assert.ok(!/src="(?!data:)/.test(html), "no external images");
  });
});
