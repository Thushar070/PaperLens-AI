import type { ReactNode } from "react";

export function PageHeader({ eyebrow, title, description, action }: {
  eyebrow: string;
  title: string;
  description: string;
  action?: ReactNode;
}) {
  return (
    <header className="page-header">
      <div>
        <p className="eyebrow">{eyebrow}</p>
        <h1>{title}</h1>
        <p className="page-description">{description}</p>
      </div>
      {action ? <div className="page-header-action">{action}</div> : null}
    </header>
  );
}

export function LoadingState({ label = "Loading saved PaperLens data…" }: { label?: string }) {
  return <div className="notice" role="status"><span className="spinner" aria-hidden="true" />{label}</div>;
}

export function ErrorState({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return <div className="notice notice-error" role="alert"><strong>Could not load this view.</strong><span>{message}</span><span>Check that the PaperLens API is available.</span>{onRetry ? <button className="button button-secondary button-small" type="button" onClick={onRetry}>Try again</button> : null}</div>;
}

export function EmptyState({ title, description }: { title: string; description: string }) {
  return <div className="empty-state"><span className="empty-icon" aria-hidden="true">⌕</span><h2>{title}</h2><p>{description}</p></div>;
}

export function MetricCard({ label, value, note, tone = "rose" }: {
  label: string;
  value: string | number;
  note: string;
  tone?: "rose" | "violet" | "emerald" | "amber";
}) {
  const displayValue = typeof value === "number" ? value.toLocaleString() : value;
  return <article className={`metric-card tone-${tone}`}><span className="metric-label">{label}</span><strong>{displayValue}</strong><span className="metric-note">{note}</span></article>;
}

export function CategoryBadge({ label }: { label: number }) {
  const names = ["Algorithms", "NLP/AI", "Systems / Infra", "Vision"];
  const classes = ["category-algorithms", "category-nlp", "category-systems", "category-vision"];
  const name = names[label] ?? "Unknown";
  return <span className={`category-badge ${classes[label] ?? ""}`}>{name}</span>;
}
