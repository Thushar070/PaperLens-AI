"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import type { ReactNode } from "react";

const navigation = [
  { href: "/", label: "Home", icon: "⌂" },
  { href: "/papers", label: "Browse Papers", icon: "▱" },
  { href: "/classify", label: "Classification", icon: "✣" },
  { href: "/search", label: "Semantic Search", icon: "⌕" },
  { href: "/experiments", label: "Experiments", icon: "▥" },
];

export function AppShell({ children }: { children: ReactNode }) {
  const pathname = usePathname();
  return (
    <div className="app-layout">
      <a className="skip-link" href="#main-content">Skip to content</a>
      <aside className="sidebar" aria-label="Primary navigation">
        <Link className="brand" href="/" aria-label="PaperLens home">
          <span className="brand-mark" aria-hidden="true"><span /></span>
          <span>Paper<span className="brand-accent">Lens</span></span>
        </Link>
        <p className="sidebar-caption">RESEARCH WORKSPACE</p>
        <nav className="nav-list">
          {navigation.map((item) => {
            const active = item.href === "/" ? pathname === "/" : pathname === item.href || pathname.startsWith(`${item.href}/`);
            return (
              <Link key={item.href} href={item.href} className={`nav-link${active ? " nav-link-active" : ""}`} aria-current={active ? "page" : undefined}>
                <span className="nav-icon" aria-hidden="true">{item.icon}</span><span>{item.label}</span>
              </Link>
            );
          })}
        </nav>
        <div className="sidebar-bottom">
          <div className="sidebar-note"><span className="status-dot" />Four research classes</div>
          <span className="sidebar-version">PaperLens · Phase 9</span>
        </div>
      </aside>
      <div className="content-column">
        <div className="topbar">
          <span className="topbar-context">Research paper discovery</span>
          <Link className="topbar-action" href="/papers">Explore collection <span aria-hidden="true">→</span></Link>
        </div>
        <main id="main-content" className="main-content">{children}</main>
        <footer className="site-footer"><span>PaperLens</span><span>Built from the verified Phase 1–8 artifacts</span></footer>
      </div>
    </div>
  );
}
