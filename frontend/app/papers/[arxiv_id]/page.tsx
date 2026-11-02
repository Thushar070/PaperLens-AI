"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { CategoryBadge, ErrorState, LoadingState, PageHeader } from "../../components/status";
import { useApi } from "../../lib/use-api";
import type { Paper } from "../../lib/api";

export default function PaperDetailPage() {
  const params = useParams<{ arxiv_id: string }>();
  const arxivId = Array.isArray(params.arxiv_id) ? params.arxiv_id[0] : params.arxiv_id;
  const { data, error, loading, retry } = useApi<Paper>(arxivId ? `/papers/${encodeURIComponent(arxivId)}` : null);
  return <>
    <PageHeader eyebrow="PAPER DETAILS" title={data?.title ?? "Paper record"} description={data ? `arXiv:${data.arxiv_id} · Original dataset metadata` : "Inspect a record from the verified research collection."} action={<Link className="button button-secondary" href="/papers">← Back to papers</Link>} />
    {loading ? <LoadingState label="Loading paper record…" /> : null}
    {error ? <ErrorState message={error} onRetry={retry} /> : null}
    {data ? <article className="surface-card paper-detail-card">
      <div className="detail-tags"><CategoryBadge label={data.label} /><span className="split-badge">{data.split} split</span><span className="mono-text">{data.primary_cat}</span></div>
      <h2>{data.title || "No title is available for this record."}</h2>
      <div className="abstract-block"><h3>Abstract</h3><p>{data.abstract || "No abstract is available for this record."}</p></div>
      <dl className="metadata-grid">
        <div><dt>arXiv ID</dt><dd className="mono-text">{data.arxiv_id}</dd></div>
        <div><dt>Primary category</dt><dd className="mono-text">{data.primary_cat || "Not available"}</dd></div>
        <div><dt>All categories</dt><dd className="mono-text">{data.all_cats || "Not available"}</dd></div>
        <div><dt>Original numeric label</dt><dd>{data.label} · <CategoryBadge label={data.label} /></dd></div>
      </dl>
      <Link className="button button-primary" href={`/classify?paper=${encodeURIComponent(data.arxiv_id)}`}>Classify a new paper</Link>
    </article> : null}
  </>;
}
