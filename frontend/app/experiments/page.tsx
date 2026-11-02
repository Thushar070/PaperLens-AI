"use client";

import { CSSProperties } from "react";
import { ErrorState, LoadingState, PageHeader } from "../components/status";
import { useApi } from "../lib/use-api";
import type { Experiments } from "../lib/api";

export default function ExperimentsPage() {
  const { data, error, loading, retry } = useApi<Experiments>("/experiments");
  return <>
    <PageHeader eyebrow="PHASE 8 EVALUATION" title="Experiment comparison" description="Compare metrics read from the saved Phase 8 evaluation artifacts. Test scores are retrospective comparisons, not model-selection criteria." />
    {loading ? <LoadingState label="Loading saved experiment results…" /> : null}
    {error ? <ErrorState message={error} onRetry={retry} /> : null}
    {data ? <>
      <section className="surface-card experiment-chart-card">
        <div className="section-heading"><div><p className="eyebrow">CLASSIFICATION</p><h2>Test macro F1 by model</h2></div><span className="chart-unit">Macro F1 · 0 to 1</span></div>
        <div className="bar-chart" role="list" aria-label="Classifier test macro F1 scores">
          {data.classifiers.map((model) => {
            const value = model.test_macro_f1;
            const style = value === null ? undefined : { "--bar-height": `${Math.max(value * 100, 1)}%` } as CSSProperties;
            const accessibleLabel = value === null
              ? `${model.short_name}: test macro F1 unavailable`
              : `${model.short_name}: ${value.toFixed(4)} test macro F1`;
            return <div className={`bar-item${model.model_id === "P3/tfidf_svm" ? " bar-item-selected" : ""}`} role="listitem" key={model.model_id} aria-label={accessibleLabel}>
              <span className="bar-value">{value === null ? "—" : value.toFixed(3)}</span><div className={`bar-track${value === null ? " bar-track-empty" : ""}`}>{value === null ? null : <span className="bar-fill" style={style} />}</div><span className="bar-label">{model.short_name.replace(/^P[3-7] /, "")}</span>
            </div>;
          })}
        </div>
        <p className="chart-footnote">The highlighted TF-IDF SVM was selected by validation macro F1. The chart shows test results for retrospective comparison only.</p>
      </section>

      <section className="surface-card table-card">
        <div className="section-heading"><div><p className="eyebrow">{data.classifiers.length} SAVED CLASSIFIERS</p><h2>Classification results</h2></div><span className="score-note">Source: Phase 8 machine-readable results</span></div>
        <div className="table-scroll"><table className="data-table"><thead><tr><th scope="col">Model</th><th scope="col">Phase</th><th scope="col">Family</th><th scope="col">Validation macro F1</th><th scope="col">Test accuracy</th><th scope="col">Test macro F1</th><th scope="col">Selection</th></tr></thead><tbody>
          {data.classifiers.map((model) => <tr key={model.model_id}><td><strong>{model.short_name}</strong><span className="paper-id">{model.model_id}</span></td><td>P{model.phase}</td><td>{model.family}</td><td>{model.validation_macro_f1?.toFixed(4) ?? "—"}</td><td>{model.test_accuracy?.toFixed(4) ?? "—"}</td><td>{model.test_macro_f1?.toFixed(4) ?? "—"}</td><td>{model.selected_by_validation ? <span className="selection-badge">Validation-selected</span> : <span className="muted-copy">Compared</span>}</td></tr>)}
        </tbody></table></div>
      </section>

      <section className="retrieval-section"><div className="section-heading"><div><p className="eyebrow">SAVED RETRIEVAL EVALUATION</p><h2>Retrieval methods</h2></div><span className="chart-unit">Macro averages · k = 5</span></div><div className="retrieval-grid">
        {data.retrieval.map((method, index) => <article className={`surface-card retrieval-card retrieval-tone-${index}`} key={method.method_id}><span className="retrieval-index">0{index + 1}</span><h3>{method.display_name}</h3><span className="paper-id">{method.method_id}</span><div className="retrieval-metrics"><div><span>Precision</span><strong>{method.precision.toFixed(4)}</strong></div><div><span>Recall</span><strong>{method.recall.toFixed(4)}</strong></div><div><span>MRR</span><strong>{method.mrr.toFixed(4)}</strong></div><div><span>nDCG</span><strong>{method.ndcg.toFixed(4)}</strong></div></div>{method.unjudged_in_top_k > 0 ? <p className="unjudged-note">{method.unjudged_in_top_k.toFixed(2)} unjudged results in top five on average</p> : null}</article>)}
      </div></section>
      <aside className="evaluation-note"><span className="note-icon" aria-hidden="true">i</span><p><strong>Evaluation context</strong>{data.retrieval_judgment_note} Phase 8 reports are sparse-query diagnostic evidence and should not be read as a human relevance benchmark.</p></aside>
    </> : null}
  </>;
}
