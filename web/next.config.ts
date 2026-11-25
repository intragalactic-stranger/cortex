import type { NextConfig } from "next";

// Fully static site: `next build` writes plain HTML/CSS/JS to out/. No server, no runtime data fetching.
// BASE_PATH lets the same build be hosted under a sub-path (e.g. GitHub Pages: BASE_PATH=/cortex).
const basePath = process.env.BASE_PATH ?? "";

const config: NextConfig = {
  output: "export",
  trailingSlash: true,
  basePath,
  images: { unoptimized: true },
  poweredByHeader: false,
  reactStrictMode: true,
  env: { NEXT_PUBLIC_BASE_PATH: basePath },
  // The pages read ../serve/** at build time; let tracing and the dev watcher see the repo root.
  outputFileTracingRoot: new URL("..", import.meta.url).pathname,
};

export default config;
