"use client";

import Link from "next/link";
import { ErrorState, LoadingState, MetricCard } from "./components/status";
import { useApi } from "./lib/use-api";
import type { Experiments, Overview } from "./lib/api";

const categoryTone: Record<string, string> = {
  Algorithms: "tone-rose",
  "NLP/AI": "tone-violet",
  "Systems/Infra": "tone-emerald",
  Vision: "tone-amber",
};
const categoryLabel: Record<string, number> = { Algorithms: 0, "NLP/AI": 1, "Systems/Infra": 2, Vision: 3 };

export default function HomePage() {
  const overview = useApi<Overview>("/overview");
  const experiments = useApi<Experiments>("/experiments");
  return (
    <>
      <section className="hero-panel">
        <div className="hero-copy">
          <p className="eyebrow">YOUR RESEARCH, IN FOCUS</p>
          <h1>Discover, classify and explore <span>research papers.</span></h1>
          <p>Search the verified arXiv collection, inspect saved model results, and classify a paper with the selected Phase 3 model.</p>
          <div className="hero-actions">
            <Link className="button button-primary" href="/papers">Browse papers <span aria-hidden="true">→</span></Link>
            <Link className="button button-secondary" href="/search">Search the collection</Link>
          </div>
        </div>
        <div className="hero-art" aria-hidden="true">
          <div className="art-orbit orbit-one" /><div className="art-orbit orbit-two" />
          <div className="paper-illustration"><span /><span /><span /><i /><i /><i /></div>
          <span className="art-spark spark-one">✦</span><span className="art-spark spark-two">✧</span>
        </div>
      </section>

      {overview.loading ? <LoadingState label="Loading the indexed collection…" /> : null}
      {overview.error ? <ErrorState message={overview.error} onRetry={overview.retry} /> : null}
      {overview.data ? (
        <section className="metric-grid" aria-label="Collection overview">
          <MetricCard label="Research papers" value={overview.data.paper_count.toLocaleString()} note="Verified Phase 1 records" />
          <MetricCard label="Research classes" value={overview.data.category_count} note="Original four-class mapping" tone="violet" />
          <MetricCard label="Classifiers evaluated" value={overview.data.classifier_count} note="Saved Phase 3–7 results" tone="emerald" />
          <MetricCard label="Retrieval methods" value={overview.data.retrieval_method_count} note="Saved Phase 4 and 7 results" tone="amber" />
        </section>
      ) : null}

      <div className="dashboard-grid">
        <section className="surface-card category-panel">
          <div className="section-heading"><div><p className="eyebrow">COLLECTION</p><h2>Explore by category</h2></div><Link className="text-link" href="/papers">View papers <span aria-hidden="true">→</span></Link></div>
          {overview.data ? (
            <div className="category-list">
              {Object.entries(overview.data.label_counts).map(([name, count]) => (
                <Link className={`category-row ${categoryTone[name] ?? "tone-violet"}`} href={`/papers?label=${categoryLabel[name]}`} key={name}>
                  <span className="category-row-mark" aria-hidden="true" />
                  <span className="category-row-name">{name}</span>
                  <span className="category-row-count">{count.toLocaleString()}</span>
                  <span className="category-row-arrow" aria-hidden="true">↗</span>
                </Link>
              ))}
            </div>
          ) : <p className="muted-copy">Category counts appear when the paper database is ready.</p>}
        </section>

        <section className="surface-card selection-panel">
          <div className="section-heading"><div><p className="eyebrow">SAVED EVALUATION</p><h2>Selected classifier</h2></div><span className="selection-icon" aria-hidden="true">✣</span></div>
          {experiments.loading ? <LoadingState label="Loading saved experiment results…" /> : null}
          {experiments.error ? <ErrorState message={experiments.error} onRetry={experiments.retry} /> : null}
          {experiments.data ? (() => {
            const chosen = experiments.data.classifiers.find((row) => row.model_id === "P3/tfidf_svm");
            return chosen ? <>
              <p className="selection-model">{chosen.short_name}</p>
              <p className="muted-copy">Selected on validation macro F1. The test result is shown as a retrospective comparison only.</p>
              <div className="selection-score"><span>Validation macro F1</span><strong>{chosen.validation_macro_f1?.toFixed(4) ?? "—"}</strong></div>
              <Link className="text-link" href="/experiments">Compare all results <span aria-hidden="true">→</span></Link>
            </> : <p className="muted-copy">The selected classifier is not present in the result artifact.</p>;
          })() : null}
        </section>
      </div>

      <section className="feature-strip" aria-label="PaperLens tools">
        <Link href="/classify" className="feature-card feature-pink"><span className="feature-symbol" aria-hidden="true">✣</span><span><strong>Classify a paper</strong><small>Use the saved TF-IDF SVM</small></span><span className="feature-arrow" aria-hidden="true">↗</span></Link>
        <Link href="/search" className="feature-card feature-purple"><span className="feature-symbol" aria-hidden="true">⌕</span><span><strong>Semantic search</strong><small>Compare TF-IDF and MiniLM</small></span><span className="feature-arrow" aria-hidden="true">↗</span></Link>
        <Link href="/experiments" className="feature-card feature-mint"><span className="feature-symbol" aria-hidden="true">▥</span><span><strong>Review experiments</strong><small>Explore saved Phase 8 results</small></span><span className="feature-arrow" aria-hidden="true">↗</span></Link>
      </section>
    </>
  );
}
