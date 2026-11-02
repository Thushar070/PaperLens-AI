CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS schema_migrations (
    version TEXT PRIMARY KEY,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS papers (
    arxiv_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    abstract TEXT NOT NULL,
    primary_cat TEXT NOT NULL,
    all_cats TEXT NOT NULL,
    label SMALLINT NOT NULL CHECK (label BETWEEN 0 AND 3),
    split TEXT NOT NULL CHECK (split IN ('train', 'validation', 'test'))
);
CREATE INDEX IF NOT EXISTS papers_label_idx ON papers (label);
CREATE INDEX IF NOT EXISTS papers_primary_cat_idx ON papers (primary_cat);
CREATE TABLE IF NOT EXISTS paper_embeddings (
    arxiv_id TEXT PRIMARY KEY REFERENCES papers(arxiv_id) ON DELETE CASCADE,
    embedding VECTOR(384) NOT NULL
);
CREATE INDEX IF NOT EXISTS paper_embeddings_hnsw_idx
    ON paper_embeddings USING hnsw (embedding vector_cosine_ops);
CREATE TABLE IF NOT EXISTS experiments (
    experiment_id TEXT PRIMARY KEY,
    kind TEXT NOT NULL CHECK (kind IN ('classifier', 'retrieval')),
    phase SMALLINT NOT NULL,
    display_name TEXT NOT NULL,
    family TEXT NOT NULL,
    validation_metrics JSONB NOT NULL DEFAULT '{}'::jsonb,
    test_metrics JSONB NOT NULL DEFAULT '{}'::jsonb,
    source_artifact TEXT NOT NULL
);
