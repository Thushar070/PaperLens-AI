"use client";

import Link from "next/link";
import { Suspense, useState, type FormEvent } from "react";
import { useSearchParams } from "next/navigation";
import { CategoryBadge, EmptyState, ErrorState, LoadingState, PageHeader } from "../components/status";
import { useApi } from "../lib/use-api";
import type { PaperList } from "../lib/api";

const PAGE_SIZE = 20;

export default function PapersPage() {
  return <Suspense fallback={<LoadingState label="Opening the paper collection…" />}><PapersPageClient /></Suspense>;
}

function PapersPageClient() {
  const searchParams = useSearchParams();
  const requestedLabel = searchParams.get("label") ?? "";
  const initialCategory = /^[0-3]$/.test(requestedLabel) ? requestedLabel : "";
  return <PapersPageContent key={initialCategory} initialCategory={initialCategory} />;
}

function PapersPageContent({ initialCategory }: { initialCategory: string }) {
  const [query, setQuery] = useState("");
  const [category, setCategory] = useState(initialCategory);
  const [offset, setOffset] = useState(0);
  const [activeQuery, setActiveQuery] = useState("");
  const [activeCategory, setActiveCategory] = useState(initialCategory);
  const params = new URLSearchParams({ offset: String(offset), limit: String(PAGE_SIZE) });
  if (activeQuery) params.set("query", activeQuery);
  if (activeCategory) params.set("label", activeCategory);
  const { data, error, loading, retry } = useApi<PaperList>(`/papers?${params.toString()}`);

  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setOffset(0);
    setActiveQuery(query.trim());
    setActiveCategory(category);
  }

  return <>
    <PageHeader eyebrow="PAPER LIBRARY" title="Browse papers" description="Explore the indexed collection and filter by its original four-class labels." />
    <section className="surface-card browse-card">
      <form className="filter-bar" onSubmit={submit}>
        <label className="search-field"><span className="visually-hidden">Search titles and abstracts</span><span className="search-field-icon" aria-hidden="true">⌕</span><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search titles and abstracts…" maxLength={300} /><button className="button button-primary" type="submit">Search</button></label>
        <label className="select-field"><span className="visually-hidden">Filter by research category</span><select value={category} onChange={(event) => setCategory(event.target.value)}><option value="">All categories</option><option value="0">Algorithms</option><option value="1">NLP / AI</option><option value="2">Systems / Infra</option><option value="3">Vision</option></select></label>
      </form>
      <div className="results-toolbar"><span>{data ? `${data.total.toLocaleString()} papers found` : "Paper collection"}</span><span>Sorted by arXiv ID</span></div>
      {loading ? <LoadingState label="Loading papers…" /> : null}
      {error ? <ErrorState message={error} onRetry={retry} /> : null}
      {data && data.items.length === 0 ? <EmptyState title="No papers match these filters" description="Try another title, abstract phrase, or category." /> : null}
      {data && data.items.length > 0 ? <>
        <div className="table-scroll"><table className="data-table paper-table"><thead><tr><th scope="col">Paper</th><th scope="col">Primary category</th><th scope="col">Research class</th><th scope="col">Split</th><th scope="col"><span className="visually-hidden">Open paper</span></th></tr></thead><tbody>
          {data.items.map((paper) => <tr key={paper.arxiv_id}><td><Link className="paper-title-link" href={`/papers/${encodeURIComponent(paper.arxiv_id)}`}>{paper.title}</Link><span className="paper-id">arXiv:{paper.arxiv_id}</span></td><td><span className="mono-text">{paper.primary_cat}</span></td><td><CategoryBadge label={paper.label} /></td><td><span className="split-badge">{paper.split}</span></td><td><Link className="icon-link" aria-label={`View ${paper.title}`} href={`/papers/${encodeURIComponent(paper.arxiv_id)}`}>↗</Link></td></tr>)}
        </tbody></table></div>
        <div className="pagination"><span>Showing {offset + 1}–{Math.min(offset + data.items.length, data.total)} of {data.total.toLocaleString()}</span><div><button className="button button-secondary button-small" disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - PAGE_SIZE))}>Previous</button><button className="button button-secondary button-small" disabled={offset + PAGE_SIZE >= data.total} onClick={() => setOffset(offset + PAGE_SIZE)}>Next</button></div></div>
      </> : null}
    </section>
    <p className="source-note">Source fields are retained as supplied: arXiv ID, title, abstract, primary category, all categories, and numeric label.</p>
  </>;
}
