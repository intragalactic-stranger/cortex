import type { Metadata } from "next";
import { GUIDE_PATH, loadGuideMarkdown } from "../../lib/data";
import { REPO_URL, renderMarkdown } from "../../lib/markdown";

export const metadata: Metadata = {
  title: "Decision guide",
  description: "Which GPU, engine, weight format and serving settings to use for ~4B open models, with the evidence for each.",
};

export default function GuidePage() {
  const md = loadGuideMarkdown();
  if (!md) {
    return (
      <section>
        <h1>Decision guide</h1>
        <p className="lede">The guide is being written; it will appear here on the next build.</p>
      </section>
    );
  }
  // The document's own H1 becomes the page title; the renderer demotes the rest by one level.
  const body = md.replace(/^# .*\n/, "");
  const title = /^# (.*)$/m.exec(md)?.[1] ?? "Decision guide";
  const { html, headings } = renderMarkdown(body, GUIDE_PATH);
  const toc = headings.filter((h) => h.depth === 2);
  return (
    <>
      <section aria-labelledby="title">
        <p className="eyebrow">Rendered from serve/docs/DECISION_GUIDE.md</p>
        <h1 id="title">{title}</h1>
        {toc.length > 0 && (
          <nav aria-label="On this page" className="small">
            <ul style={{ display: "flex", flexWrap: "wrap", gap: "0.3rem 1rem", listStyle: "none", padding: 0, margin: 0 }}>
              {toc.map((h) => (
                <li key={h.id}>
                  <a href={`#${h.id}`}>{h.text}</a>
                </li>
              ))}
            </ul>
          </nav>
        )}
      </section>
      <article className="doc" dangerouslySetInnerHTML={{ __html: html }} />
      <p className="small">
        Source: <a href={`${REPO_URL}/blob/main/${GUIDE_PATH}`}>{GUIDE_PATH}</a>
      </p>
    </>
  );
}
