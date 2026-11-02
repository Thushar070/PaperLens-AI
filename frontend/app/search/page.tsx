"use client";

import Link from "next/link";
import { FormEvent, useState } from "react";
import { CategoryBadge, EmptyState, ErrorState, LoadingState, PageHeader } from "../components/status";
import { postJson, type SearchResults } from "../lib/api";

export default function SearchPage() {
  const [query, setQuery] = useState("");
  const [method, setMethod] = useState<"tfidf" | "minilm">("tfidf");
  const [results, setResults] = useState<SearchResults | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const normalizedQuery = query.trim();
    if (!normalizedQuery) {
      setError("Enter a research query before searching.");
      setResults(null);
      return;
    }
    setLoading(true);
    setError(null);
    setResults(null);
    try {
      setResults(await postJson<SearchResults>("/search", { query: normalizedQuery, method, k: 10 }));
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Search request failed.");
    } finally {
      setLoading(false);
    }
  }

  return <>
    <PageHeader eyebrow="RETRIEVAL WORKSPACE" title="Search research papers" description="Find relevant papers with the saved TF-IDF representation or the persisted Phase 7 MiniLM vectors." />
    <section className="surface-card search-panel">
      <form onSubmit={submit}>
        <label className="form-label" htmlFor="research-query">What are you researching?</label>
        <div className="search-submit-row"><span className="large-search-icon" aria-hidden="true">⌕</span><input id="research-query" className="text-input" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="e.g. graph neural networks for node classification" maxLength={4000} required /><button className="button button-primary" type="submit" disabled={loading || !query.trim()}>{loading ? "Searching…" : "Search"}<span aria-hidden="true">→</span></button></div>
        <fieldset className="method-selector"><legend>Retrieval method</legend>
          <label className={`method-option${method === "tfidf" ? " method-selected" : ""}`}><input type="radio" name="method" value="tfidf" checked={method === "tfidf"} onChange={() => setMethod("tfidf")} /><span><strong>TF-IDF</strong><small>Recommended · saved Phase 4 index</small></span><span className="method-check" aria-hidden="true">✓</span></label>
          <label className={`method-option${method === "minilm" ? " method-selected" : ""}`}><input type="radio" name="method" value="minilm" checked={method === "minilm"} onChange={() => setMethod("minilm")} /><span><strong>MiniLM</strong><small>Phase 7 encoder · pgvector cosine search</small></span><span className="method-check" aria-hidden="true">✓</span></label>
        </fieldset>
      </form>
    </section>

    {loading ? <LoadingState label="Ranking saved paper representations…" /> : null}
    {error ? <ErrorState message={error} /> : null}
    {results ? <section className="search-results" aria-live="polite">
      <div className="section-heading"><div><p className="eyebrow">{results.method === "tfidf" ? "PHASE 4 TF-IDF" : "PHASE 7 MINILM"}</p><h2>{results.items.length} matching papers</h2></div><span className="score-note">Scores are ranking signals, not probabilities</span></div>
      <p className="source-note">{results.score_description}</p>
      {results.items.length === 0 ? <EmptyState title="No matching papers found" description="Try a broader query or switch retrieval methods." /> : null}
      {results.items.map((paper, index) => <article className="surface-card result-card" key={paper.arxiv_id}>
        <div className="result-rank">{String(index + 1).padStart(2, "0")}</div>
        <div className="result-content"><div className="result-tags"><CategoryBadge label={paper.label} /><span className="mono-text">{paper.primary_cat}</span></div><h3><Link href={`/papers/${encodeURIComponent(paper.arxiv_id)}`}>{paper.title}</Link></h3><p>{paper.abstract}</p><span className="paper-id">arXiv:{paper.arxiv_id}</span></div>
        <div className="result-score"><span>Score</span><strong>{paper.score.toFixed(4)}</strong></div>
      </article>)}
      <p className="evaluation-caveat">Phase 8 retrieval evaluation used three queries and 27 assistant-assessed document judgments; those sparse judgments were not human-reviewed.</p>
    </section> : null}
  </>;
}
