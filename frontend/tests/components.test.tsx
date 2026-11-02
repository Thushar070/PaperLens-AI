import { render, screen } from "@testing-library/react";
import type { AnchorHTMLAttributes } from "react";
import { describe, expect, it, vi } from "vitest";

vi.mock("next/navigation", () => ({ usePathname: () => "/papers" }));
vi.mock("next/link", () => ({
  default: ({ href, ...props }: AnchorHTMLAttributes<HTMLAnchorElement> & { href: string }) => <a href={href} {...props}>{props.children}</a>,
}));

import { AppShell } from "../app/components/app-shell";
import { CategoryBadge, EmptyState, ErrorState, LoadingState, MetricCard } from "../app/components/status";

describe("PaperLens shared interface components", () => {
  it("exposes all primary destinations and identifies the active page", () => {
    render(<AppShell><p>Page content</p></AppShell>);
    expect(screen.getByRole("link", { name: "Home" })).toHaveAttribute("href", "/");
    expect(screen.getByRole("link", { name: "Browse Papers" })).toHaveAttribute("aria-current", "page");
    expect(screen.getByRole("link", { name: "Classification" })).toHaveAttribute("href", "/classify");
    expect(screen.getByRole("link", { name: "Semantic Search" })).toHaveAttribute("href", "/search");
    expect(screen.getByRole("link", { name: "Experiments" })).toHaveAttribute("href", "/experiments");
    expect(screen.getByText("Page content")).toBeInTheDocument();
  });

  it("maps numeric labels to the original visible category names", () => {
    const { rerender } = render(<CategoryBadge label={0} />);
    expect(screen.getByText("Algorithms")).toBeInTheDocument();
    rerender(<CategoryBadge label={1} />);
    expect(screen.getByText("NLP/AI")).toBeInTheDocument();
    rerender(<CategoryBadge label={2} />);
    expect(screen.getByText("Systems / Infra")).toBeInTheDocument();
    rerender(<CategoryBadge label={3} />);
    expect(screen.getByText("Vision")).toBeInTheDocument();
  });

  it("renders loading, empty, and actionable error states accessibly", () => {
    const { rerender } = render(<LoadingState label="Loading papers" />);
    expect(screen.getByRole("status")).toHaveTextContent("Loading papers");
    rerender(<EmptyState title="No papers" description="Adjust the filters." />);
    expect(screen.getByRole("heading", { name: "No papers" })).toBeInTheDocument();
    rerender(<ErrorState message="API offline" />);
    expect(screen.getByRole("alert")).toHaveTextContent("API offline");
  });

  it("offers an explicit retry action for recoverable loading failures", () => {
    const retry = vi.fn();
    render(<ErrorState message="API offline" onRetry={retry} />);
    screen.getByRole("button", { name: "Try again" }).click();
    expect(retry).toHaveBeenCalledOnce();
  });

  it("shows metric values and context without changing the supplied count", () => {
    render(<MetricCard label="Research papers" value={19999} note="Verified Phase 1 records" />);
    expect(screen.getByText("19,999")).toBeInTheDocument();
    expect(screen.getByText("Verified Phase 1 records")).toBeInTheDocument();
  });
});
