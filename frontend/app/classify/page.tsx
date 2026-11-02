"use client";

import { Suspense, useState, type FormEvent } from "react";
import { useSearchParams } from "next/navigation";
import { CategoryBadge, ErrorState, LoadingState, PageHeader } from "../components/status";
import { postJson, type Classification, type Paper } from "../lib/api";
import { useApi } from "../lib/use-api";

export default function ClassificationPage() {
  return <Suspense fallback={<LoadingState label="Preparing the classifier form…" />}><ClassificationForm /></Suspense>;
}

function ClassificationForm() {
  const searchParams = useSearchParams();
  const paperId = searchParams.get("paper");
  const paper = useApi<Paper>(paperId ? `/papers/${encodeURIComponent(paperId)}` : null);
  const [draft, setDraft] = useState({ paperId, title: "", abstract: "", edited: false });
  const draftMatchesPaper = draft.paperId === paperId;
  const title = draftMatchesPaper && draft.edited ? draft.title : paper.data?.title ?? (draftMatchesPaper ? draft.title : "");
  const abstract = draftMatchesPaper && draft.edited ? draft.abstract : paper.data?.abstract ?? (draftMatchesPaper ? draft.abstract : "");
  const [result, setResult] = useState<Classification | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError(null);
    setResult(null);
    const normalizedTitle = title.trim();
    const normalizedAbstract = abstract.trim();
    if (!normalizedTitle || !normalizedAbstract) {
      setError("Enter a nonblank title and abstract before classifying.");
      return;
    }
    setSubmitting(true);
    try {
      setResult(await postJson<Classification>("/classifications", { title: normalizedTitle, abstract: normalizedAbstract }));
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Classification request failed.");
    } finally {
      setSubmitting(false);
    }
  }

  return <>
    <PageHeader eyebrow="SAVED MODEL INFERENCE" title="Classify a paper" description="Submit a title and abstract to the existing validation-selected Phase 3 TF-IDF Linear SVM." />
    {paper.loading ? <LoadingState label="Loading the selected paper record…" /> : null}
    {paper.error ? <ErrorState message={paper.error} onRetry={paper.retry} /> : null}
    <div className="classification-layout">
      <form className="surface-card form-card" onSubmit={submit}>
        <div className="form-card-heading"><span className="step-number">01</span><div><h2>Paper details</h2><p>Text is used only for this prediction.</p></div></div>
        <label className="form-label" htmlFor="paper-title">Title</label>
        <input id="paper-title" className="text-input" value={title} onChange={(event) => setDraft({ paperId, title: event.target.value, abstract, edited: true })} maxLength={1000} required aria-describedby="title-count" />
        <span id="title-count" className="field-hint">{title.length} / 1,000 characters</span>
        <label className="form-label" htmlFor="paper-abstract">Abstract</label>
        <textarea id="paper-abstract" className="text-input textarea" value={abstract} onChange={(event) => setDraft({ paperId, title, abstract: event.target.value, edited: true })} maxLength={50000} rows={12} required aria-describedby="abstract-count" />
        <span id="abstract-count" className="field-hint">{abstract.length.toLocaleString()} / 50,000 characters</span>
        {error ? <div className="inline-error" role="alert">{error}</div> : null}
        <button className="button button-primary button-wide" type="submit" disabled={submitting || !title.trim() || !abstract.trim()}>
          {submitting ? "Classifying…" : "Classify paper"}<span aria-hidden="true">→</span>
        </button>
      </form>

      <section className="surface-card prediction-card" aria-live="polite" aria-label="Prediction result">
        <div className="form-card-heading"><span className="step-number step-violet">02</span><div><h2>Prediction result</h2><p>Output from the saved classifier artifact.</p></div></div>
        {result ? <>
          <div className="prediction-category"><span className="prediction-spark" aria-hidden="true">✣</span><div><span className="eyebrow">PREDICTED CLASS</span><h3>{result.category}</h3><CategoryBadge label={result.label} /></div></div>
          <div className="prediction-meta"><span>Model</span><strong>{result.model_id}</strong></div>
          <h4>Decision scores</h4>
          <p className="muted-copy">Linear SVM margins indicate relative class scores. They are not calibrated probabilities.</p>
          <div className="score-list">{Object.entries(result.decision_scores).map(([name, score]) => {
            const max = Math.max(...Object.values(result.decision_scores).map((value) => Math.abs(value)), 0.0001);
            return <div className="score-row" key={name}><span>{name}</span><div className="score-track"><span className={score >= 0 ? "score-fill" : "score-fill score-negative"} style={{ width: `${Math.max(2, Math.abs(score) / max * 100)}%` }} /></div><strong>{score.toFixed(3)}</strong></div>;
          })}</div>
        </> : <div className="prediction-empty"><span className="prediction-placeholder" aria-hidden="true">✦</span><h3>Your prediction will appear here</h3><p>Enter a paper title and abstract, then run the saved model.</p></div>}
      </section>
    </div>
    <p className="source-note">The classifier reuses the Phase 3 TF-IDF vectorizer and Linear SVM model. No model is retrained by this app.</p>
  </>;
}
