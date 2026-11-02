import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import type { Page, Route } from "@playwright/test";
import type { Overview, Paper, PaperList, SearchResults } from "../../app/lib/api";

type Phase1Metadata = {
  selected_records: number;
  label_mapping: Record<string, string>;
  selected_class_distribution: Record<string, number>;
};

type ClassificationArtifact = {
  selection_summary: Record<string, unknown>;
  models: Array<{
    model_id: string;
    short_name: string;
    phase: number;
    family: string;
    selected_by_validation: boolean;
    validation_metrics?: { macro_f1?: number | null };
    test_metrics?: { accuracy?: number | null; macro_f1?: number | null };
  }>;
};

type RetrievalRank = { arxiv_id: string; score: number };
type RetrievalArtifact = {
  methods: Record<string, {
    display_name: string;
    macro_average: {
      precision: number;
      recall: number;
      mrr: number;
      ndcg: number;
      unjudged_in_top_k: number;
    };
    ranked_results: Record<string, RetrievalRank[]>;
  }>;
  queries: Record<string, string>;
  query_ids: string[];
  judgments: { interpretation: string; total_query_document_grades: number };
};

const dataRoot = resolve(process.cwd(), "../artifacts/data");
const readJson = <T,>(path: string): T => JSON.parse(readFileSync(path, "utf8")) as T;
const phase1 = readJson<Phase1Metadata>(resolve(dataRoot, "phase1/metadata.json"));
const classification = readJson<ClassificationArtifact>(resolve(dataRoot, "phase8/classification_results.json"));
const retrieval = readJson<RetrievalArtifact>(resolve(dataRoot, "phase8/retrieval_results.json"));

function loadPaperCorpus(): Paper[] {
  return (["train", "validation", "test"] as const).flatMap((split) => {
    const path = resolve(dataRoot, `phase1/${split}.jsonl`);
    return readFileSync(path, "utf8")
      .split(/\r?\n/)
      .filter(Boolean)
      .map((line) => ({ ...JSON.parse(line) as Omit<Paper, "split">, split }));
  });
}

const corpus = loadPaperCorpus();
const papersById = new Map(corpus.map((paper) => [paper.arxiv_id, paper]));

function rankedPapers(methodId: string): Array<Paper & { score: number }> {
  return retrieval.methods[methodId].ranked_results.q1.map((rank) => {
    const paper = papersById.get(rank.arxiv_id);
    if (!paper) throw new Error(`Saved Phase 8 result ${rank.arxiv_id} is absent from the Phase 1 splits`);
    return { ...paper, score: rank.score };
  });
}

export const samplePaper = corpus[0];
export const savedSearchQuery = retrieval.queries.q1;

export const overviewFixture: Overview = {
  paper_count: phase1.selected_records,
  category_count: Object.keys(phase1.label_mapping).length,
  classifier_count: classification.models.length,
  retrieval_method_count: Object.keys(retrieval.methods).length,
  label_counts: phase1.selected_class_distribution,
};

export const experimentsFixture = {
  classifiers: classification.models.map((model) => ({
    model_id: model.model_id,
    short_name: model.short_name,
    phase: model.phase,
    family: model.family,
    validation_macro_f1: model.validation_metrics?.macro_f1 ?? null,
    test_accuracy: model.test_metrics?.accuracy ?? null,
    test_macro_f1: model.test_metrics?.macro_f1 ?? null,
    selected_by_validation: model.selected_by_validation,
  })),
  retrieval: Object.entries(retrieval.methods).map(([method_id, method]) => ({
    method_id,
    display_name: method.display_name,
    ...method.macro_average,
  })),
  classifier_selection: classification.selection_summary,
  retrieval_judgment_note: `${retrieval.judgments.interpretation} Metrics use ${retrieval.query_ids.length} queries and ${retrieval.judgments.total_query_document_grades} document judgments.`,
};

export type DatabaseMockOptions = {
  paperListFailures?: number;
  searchFailures?: number;
  emptySearch?: boolean;
};

async function fulfillJson(route: Route, status: number, body: unknown) {
  await route.fulfill({
    status,
    contentType: "application/json",
    headers: {
      "Access-Control-Allow-Origin": "http://127.0.0.1:3109",
      "Access-Control-Allow-Methods": "GET, POST",
      "Access-Control-Allow-Headers": "Accept, Content-Type",
    },
    body: JSON.stringify(body),
  });
}

export async function mockDatabaseApi(page: Page, options: DatabaseMockOptions = {}) {
  const requests = { paperList: 0, search: 0 };
  await page.route("**/api/v1/**", async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    const path = url.pathname.replace(/^\/api\/v1/, "");
    if (request.method() === "OPTIONS") {
      await route.fulfill({
        status: 204,
        headers: {
          "Access-Control-Allow-Origin": "http://127.0.0.1:3109",
          "Access-Control-Allow-Methods": "GET, POST",
          "Access-Control-Allow-Headers": "Accept, Content-Type",
        },
      });
      return;
    }

    if (path === "/overview") {
      await fulfillJson(route, 200, overviewFixture);
      return;
    }

    if (path === "/experiments") {
      await fulfillJson(route, 200, experimentsFixture);
      return;
    }

    if (path === "/papers" && request.method() === "GET") {
      requests.paperList += 1;
      if (requests.paperList <= (options.paperListFailures ?? 0)) {
        await fulfillJson(route, 503, { detail: "Paper database is unavailable" });
        return;
      }

      const query = (url.searchParams.get("query") ?? "").trim().toLocaleLowerCase();
      const labelParam = url.searchParams.get("label");
      const label = labelParam === null ? null : Number(labelParam);
      const offset = Number(url.searchParams.get("offset") ?? 0);
      const limit = Number(url.searchParams.get("limit") ?? 20);
      const matching = corpus.filter((paper) =>
        (label === null || paper.label === label)
        && (!query || `${paper.title}\n${paper.abstract}`.toLocaleLowerCase().includes(query)),
      );
      const response: PaperList = {
        items: matching.slice(offset, offset + limit),
        total: matching.length,
        offset,
        limit,
      };
      await fulfillJson(route, 200, response);
      return;
    }

    if (path.startsWith("/papers/") && request.method() === "GET") {
      const arxivId = decodeURIComponent(path.slice("/papers/".length));
      const paper = papersById.get(arxivId);
      await fulfillJson(route, paper ? 200 : 404, paper ?? { detail: "Paper not found" });
      return;
    }

    if (path === "/classifications" && request.method() === "POST") {
      // The route is intercepted in these UI journeys. Keep the response aligned
      // with the live saved-model prediction observed for samplePaper (2603.08709).
      await fulfillJson(route, 200, {
        model_id: "P3/tfidf_svm",
        label: samplePaper.label,
        category: ["Algorithms", "NLP/AI", "Systems/Infra", "Vision"][samplePaper.label],
        score_kind: "decision_function_margin",
        decision_scores: {
          Algorithms: -1.196,
          "NLP/AI": -1.151,
          "Systems/Infra": -0.948,
          Vision: 0.951,
        },
      });
      return;
    }

    if (path === "/search" && request.method() === "POST") {
      requests.search += 1;
      if (requests.search <= (options.searchFailures ?? 0)) {
        await fulfillJson(route, 503, { detail: "Search could not complete" });
        return;
      }
      const payload = request.postDataJSON() as { method: "tfidf" | "minilm" };
      const methodId = payload.method === "minilm" ? "phase7_minilm" : "phase4_tfidf";
      const results: SearchResults = {
        method: payload.method,
        score_description: payload.method === "minilm"
          ? "Cosine similarity from saved Phase 7 MiniLM embeddings using pgvector; ranking score, not probability."
          : "Cosine similarity from saved TF-IDF features; ranking score, not probability.",
        items: options.emptySearch ? [] : rankedPapers(methodId),
      };
      await fulfillJson(route, 200, results);
      return;
    }

    await route.continue();
  });

  return requests;
}
