"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const LINKS = [
  { href: "/", label: "Overview" },
  { href: "/results/", label: "Results" },
  { href: "/memcalc/", label: "Memory calculator" },
  { href: "/guide/", label: "Decision guide" },
  { href: "/architecture/", label: "Architecture" },
];

export function Nav() {
  const path = usePathname() ?? "/";
  const norm = (p: string) => (p.endsWith("/") ? p : `${p}/`);
  return (
    <header className="nav">
      <nav className="nav-inner" aria-label="Main">
        <Link className="brand" href="/">
          Cortex Serve <small>v1.0</small>
        </Link>
        <ul>
          {LINKS.map((l) => (
            <li key={l.href}>
              <Link href={l.href} aria-current={norm(path) === l.href ? "page" : undefined}>
                {l.label}
              </Link>
            </li>
          ))}
        </ul>
      </nav>
    </header>
  );
}
