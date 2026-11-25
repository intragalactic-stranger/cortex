# M6 · Interactive web site: Report

> **Status:** `DONE` (built, tested, Lighthouse-checked; public deployment pending the owner's go-ahead) ·
> **Date:** 2026-11-25 · **Modal spend:** $0 · **Code:** [`web/`](../../../web/)

## 1. Summary of what was done

| Deliverable | Status | Evidence |
|---|---|---|
| Next.js App Router site, TypeScript strict, fully static export | Done | `web/`, `next.config.ts` (`output: "export"`) |
| Browser memcalc: TS port, device table, model presets, paste a `config.json` | Done | `web/src/lib/memcalc/`, `/memcalc/` |
| Parity with the Python memcalc in CI | Done | `test/golden.test.ts` (golden.json), `test/vectors.test.ts` (192 estimates + per-layer specs), CI regenerates the vectors |
| Results explorer: throughput–latency curves, quality table, cost frontier | Done | `/results/`, data from `serve/results/web/*.json` |
| Decision guide page rendered from `DECISION_GUIDE.md`, architecture page | Done | `/guide/`, `/architecture/`; Mermaid drawn as static SVG |
| Links to repo and Hugging Face models | Done | home page and footer |
| Lighthouse check | Done | [`web/LIGHTHOUSE.md`](../../../web/LIGHTHOUSE.md) |
| CI job | Done | `.github/workflows/ci.yml` job `web`: vectors up to date, `npm ci`, typecheck, tests, build |
| Deploy to Vercel | **Pending** | Needs the owner's Vercel login and an explicit yes (public action) |
| Optional live "try it" box (G-09) | Not built | Optional; would need an always-on endpoint, which conflicts with the zero-idle-cost rule |

## 2. Acceptance criteria (Phase 1 plan, M6)

| Criterion | Met? | Notes |
|---|---|---|
| Deployed URL | ⏳ | Build is host-ready (`web/out`, 1.9 MB); waiting for approval to publish |
| Golden-vector parity in CI | ✅ | 219 tests; vectors regenerated from Python in CI and diffed |
| Lighthouse ≥ 90 performance and accessibility | ✅ | Mobile performance medians 91–95, desktop 100; accessibility 100 on all pages |
| Mobile layout works | ✅ | Checked at 390 px wide: controls stack, tables and wide diagrams scroll inside their own box |

## 3. Design choices

1. **Static export, data read at build time.** Pages are server components that read the committed JSON and
   Markdown. There is no API and nothing to keep running, so hosting is free and the numbers on the site are
   always the numbers in the repository at that commit.
2. **Hand-written SVG charts** instead of a chart library: smaller bundles, full control over accessibility
   (titles, descriptions, focusable points, text readouts) and the same look in light and dark themes.
3. **A small Mermaid-subset renderer** (`src/lib/flowchart.ts`, layered layout with dummy nodes and barycentre
   ordering) instead of shipping the Mermaid library to the browser to draw three diagrams. Supports TD and LR
   flowcharts; anything else falls back to a code block.
4. **TypeScript port of memcalc rather than WASM/Pyodide.** The calculator is pure arithmetic; a port keeps the
   page light. The risk of drift is handled by the generated parity vectors, which compare every output field.

## 4. Deviations from plan

1. Site data is produced by `serve/scripts/e14_frontier.py` (M7), not a `servebench report --web` subcommand.
   Same content; one script owns both the E14 report and the site data.
2. The live demo box (G-09) was not built. Default was "no"; it would need an always-on paid endpoint.
3. Deployment waits for the owner (public action).

## 5. What went wrong

- The first flowchart layout spread the decision tree across ~1,900 px; labels were long single lines and leaves
  drifted away from their parents. Fixed with word-wrapping and a final placement pass that pulls each node
  toward all of its neighbours (now ~1,040 px). LR diagrams originally fell back to code blocks; they are now
  drawn by laying out with transposed boxes.
- First Lighthouse pass: heading-order failures (chart panels used `h3` directly under the page `h1`) and a 404
  for the missing site icon. Both fixed; accessibility and best practices are 100 on every page.
- Measured behind an uncompressed dev server, mobile performance looked like 78–99. Measuring the way a host
  serves the site (gzip) gave stable 87–95 single runs. Both are recorded in `LIGHTHOUSE.md`.

## 6. Next

- Publish the site (Vercel, or GitHub Pages with `BASE_PATH=/cortex`) once approved, then add the URL to the
  READMEs.
