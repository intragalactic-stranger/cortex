// Build-time markdown rendering for repository documents (the decision guide). Content is repo-owned and
// rendered to static HTML: relative links become GitHub links, headings get anchors, tables scroll inside
// their own box, and Mermaid flowcharts become static SVG (see flowchart.ts).
import path from "node:path";
import { Marked, type Tokens } from "marked";
import { flowchartSvg } from "./flowchart.ts";

export const REPO_URL = "https://github.com/intragalactic-stranger/cortex";

export interface Heading {
  depth: number;
  text: string;
  id: string;
}

export function slugify(text: string): string {
  return text
    .toLowerCase()
    .replace(/<[^>]+>/g, "")
    .replace(/[^\p{L}\p{N}\s-]/gu, "")
    .trim()
    .replace(/\s+/g, "-");
}

const esc = (s: string): string =>
  s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");

/** Resolve a link found in `docPath` (repo-relative) to an absolute URL. */
export function resolveRepoLink(href: string, docPath: string): string {
  if (/^[a-z]+:/i.test(href) || href.startsWith("#") || href.startsWith("//")) return href;
  const [target, hash] = href.split("#") as [string, string | undefined];
  const resolved = path.posix.normalize(path.posix.join(path.posix.dirname(docPath), target));
  const kind = resolved.endsWith("/") || !path.posix.extname(resolved) ? "tree" : "blob";
  return `${REPO_URL}/${kind}/main/${resolved.replace(/\/$/, "")}${hash ? `#${hash}` : ""}`;
}

export function renderMarkdown(src: string, docPath: string): { html: string; headings: Heading[] } {
  const headings: Heading[] = [];
  const seen = new Map<string, number>();
  let diagrams = 0;
  const md = new Marked({ gfm: true });
  md.use({
    renderer: {
      heading(this: { parser: { parseInline(t: Tokens.Generic[]): string } }, { tokens, depth, text }: Tokens.Heading) {
        const inner = this.parser.parseInline(tokens);
        let id = slugify(text) || "section";
        const n = seen.get(id) ?? 0;
        seen.set(id, n + 1);
        if (n) id = `${id}-${n}`;
        headings.push({ depth, text: inner.replace(/<[^>]+>/g, ""), id });
        // The page supplies the h1, so document headings start at h2 (keeps heading order valid).
        const level = Math.min(Math.max(depth, 2), 6);
        return `<h${level} id="${id}"><a class="anchor" href="#${id}" aria-hidden="true" tabindex="-1">#</a>${inner}</h${level}>\n`;
      },
      link(this: { parser: { parseInline(t: Tokens.Generic[]): string } }, { href, title, tokens }: Tokens.Link) {
        const url = resolveRepoLink(href, docPath);
        const external = /^https?:/.test(url);
        return `<a href="${esc(url)}"${title ? ` title="${esc(title)}"` : ""}${external ? ' rel="noopener"' : ""}>${this.parser.parseInline(tokens)}</a>`;
      },
      image({ href, text }: Tokens.Image) {
        // No runtime fetches: images in repo docs become links to the file on GitHub.
        return `<a href="${esc(resolveRepoLink(href, docPath))}" rel="noopener">${esc(text || href)} (image)</a>`;
      },
      code({ text, lang }: Tokens.Code) {
        if (lang === "mermaid") {
          const title = `Diagram ${++diagrams}: ${headings.at(-1)?.text ?? "flowchart"}`;
          const fc = flowchartSvg(text, title);
          if (fc) {
            return (
              `<figure class="diagram"><div class="diagram-scroll" tabindex="0" role="group" aria-label="${title} (scrolls horizontally)">${fc.svg}</div>` +
              `<details class="diagram-alt"><summary>Text version</summary>${fc.outline}</details>` +
              `<details class="diagram-alt"><summary>Mermaid source</summary><pre><code>${esc(text)}</code></pre></details></figure>\n`
            );
          }
        }
        return `<pre tabindex="0"><code${lang ? ` class="language-${esc(lang)}"` : ""}>${esc(text)}</code></pre>\n`;
      },
      table(this: { parser: { parseInline(t: Tokens.Generic[]): string } }, token: Tokens.Table) {
        const cell = (c: Tokens.TableCell, tag: "th" | "td"): string =>
          `<${tag}${c.align ? ` style="text-align:${c.align}"` : ""}${tag === "th" ? ' scope="col"' : ""}>${this.parser.parseInline(c.tokens)}</${tag}>`;
        const head = `<tr>${token.header.map((c) => cell(c, "th")).join("")}</tr>`;
        const body = token.rows.map((r) => `<tr>${r.map((c) => cell(c, "td")).join("")}</tr>`).join("");
        return `<div class="table-scroll" tabindex="0" role="region" aria-label="Table"><table><thead>${head}</thead><tbody>${body}</tbody></table></div>\n`;
      },
    },
  });
  const html = md.parse(src, { async: false });
  return { html, headings };
}
