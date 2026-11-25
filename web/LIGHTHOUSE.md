# Lighthouse results (M6)

Target from the M6 plan: **performance ≥ 90 and accessibility ≥ 90**, mobile layout working.

## How it was measured

- Lighthouse 13.6.0, Chrome for Testing 153.0.8010.12, headless, on a 4-vCPU cloud VM (benchmark index ≈ 950).
- Built with `npm run build` and served from `out/` by `npm start` (`serve@14.2.4`, gzip on), on localhost.
- Three runs per page and form factor; default mobile emulation (Moto G Power, 4× CPU slowdown, slow 4G)
  and `--preset=desktop`. The table shows the **median** performance score and the median LCP and TBT.

```bash
cd web && npm run build && npm start            # serves out/ on :3000
CHROME_PATH=/path/to/chrome npx lighthouse@13.6.0 http://localhost:3000/results/ \
  --only-categories=performance,accessibility,best-practices,seo --view
```

## Results (2026-11-25)

| Page | Mobile perf (3 runs) | Mobile LCP / TBT | Desktop perf | Accessibility | Best practices | SEO |
|---|---|---|---|---|---|---|
| `/` | **93** (95, 93, 87) | 0.77 s / 332 ms | 100 | 100 | 100 | 100 |
| `/results/` | **91** (92, 91, 87) | 0.92 s / 382 ms | 100 | 100 | 100 | 100 |
| `/memcalc/` | **93** (93, 93, 94) | 0.76 s / 305 ms | 100 | 100 | 100 | 100 |
| `/guide/` | **92** (92, 91, 95) | 0.91 s / 359 ms | 100 | 100 | 100 | 100 |
| `/architecture/` | **95** (95, 95, 93) | 0.76 s / 260 ms | 100 | 100 | 100 | 100 |

Cumulative layout shift was 0.000 on every run. Accessibility, best practices and SEO were 100 on all 30 runs.

## Notes

- **Mobile performance is limited by total blocking time**, which is the React/Next.js runtime parsing and
  hydrating (~160 KB of gzip-compressed JavaScript shared by all pages). Content is server-rendered HTML, so LCP is
  under 1 s. Single runs vary by ±5 points on this VM, so one run can dip below 90; the medians are all ≥ 91.
- Served **without compression** (e.g. `python -m http.server`), the same pages score 78–99 on mobile because
  ~490 KB of uncompressed JavaScript is transferred. Any real host (Vercel, GitHub Pages, Netlify) compresses.
- Fixes made because of the first Lighthouse pass: heading order (chart panels now use `h2`; rendered
  documents start at `h2`) and a missing site icon (a 404 in the console).
