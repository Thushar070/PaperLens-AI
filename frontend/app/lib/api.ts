export type Paper = {
  arxiv_id: string;
  title: string;
  abstract: string;
  primary_cat: string;
  all_cats: string;
  label: number;
  split: "train" | "validation" | "test";
};

export type Overview = {
  paper_count: number;
  category_count: number;
  classifier_count: number;
  retrieval_method_count: number;
  label_counts: Record<string, number>;
};

export type PaperList = { items: Paper[]; total: number; offset: number; limit: number };
export type Classification = {
  model_id: string;
  label: number;
  category: string;
  score_kind: "decision_function_margin";
  decision_scores: Record<string, number>;
};
export type SearchHit = Paper & { score: number };
export type SearchResults = {
  method: "tfidf" | "minilm";
  score_description: string;
  items: SearchHit[];
};
export type ClassifierExperiment = {
  model_id: string;
  short_name: string;
  phase: number;
  family: string;
  validation_macro_f1: number | null;
  test_accuracy: number | null;
  test_macro_f1: number | null;
  selected_by_validation: boolean;
};
export type RetrievalExperiment = {
  method_id: string;
  display_name: string;
  precision: number;
  recall: number;
  mrr: number;
  ndcg: number;
  unjudged_in_top_k: number;
};
export type Experiments = {
  classifiers: ClassifierExperiment[];
  retrieval: RetrievalExperiment[];
  classifier_selection: Record<string, unknown>;
  retrieval_judgment_note: string;
};

const API_BASE = (process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000/api/v1").replace(/\/$/, "");

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    cache: "no-store",
    headers: { Accept: "application/json", ...(init?.body ? { "Content-Type": "application/json" } : {}), ...init?.headers },
  });
  if (!response.ok) {
    let message = `Request failed (${response.status})`;
    try {
      const payload = await response.json();
      if (typeof payload.detail === "string") message = payload.detail;
    } catch {
      // Keep the status-based message when the service did not return JSON.
    }
    throw new Error(message);
  }
  return (await response.json()) as T;
}

export const getJson = <T>(path: string, init?: RequestInit) => request<T>(path, init);
export const postJson = <T>(path: string, body: unknown) =>
  request<T>(path, { method: "POST", body: JSON.stringify(body) });

export function labelName(label: number): string {
  return ["Algorithms", "NLP/AI", "Systems / Infra", "Vision"][label] ?? "Unknown";
}
