import type { Metadata, Viewport } from "next";
import { IBM_Plex_Mono, IBM_Plex_Sans } from "next/font/google";
import type { ReactNode } from "react";
import { Nav } from "../components/Nav";
import { REPO_URL } from "../lib/markdown";
import "./globals.css";

// Fonts are downloaded at build time and served with the site (no runtime requests to Google).
const sans = IBM_Plex_Sans({ subsets: ["latin"], weight: ["400", "500", "600", "700"], variable: "--plex-sans", display: "swap" });
const mono = IBM_Plex_Mono({ subsets: ["latin"], weight: ["400", "500"], variable: "--plex-mono", display: "swap" });

export const metadata: Metadata = {
  title: { default: "Cortex Serve", template: "%s · Cortex Serve" },
  description:
    "Measured LLM serving: engines, quantization, serving features and costs for ~4B open models, with a browser memory calculator and a decision guide.",
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#f6f7f9" },
    { media: "(prefers-color-scheme: dark)", color: "#0e131a" },
  ],
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en" className={`${sans.variable} ${mono.variable}`}>
      <body>
        <a className="skip" href="#main">
          Skip to content
        </a>
        <Nav />
        <main id="main">{children}</main>
        <footer>
          <div>
            <span>
              Cortex Serve · every number comes from a committed config and run manifest ·{" "}
              <a href={REPO_URL}>source on GitHub</a>
            </span>
            <span>Apache-2.0</span>
          </div>
        </footer>
      </body>
    </html>
  );
}
