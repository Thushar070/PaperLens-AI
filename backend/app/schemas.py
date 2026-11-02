from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

LABEL_NAMES = {0: "Algorithms", 1: "NLP/AI", 2: "Systems/Infra", 3: "Vision"}


class PaperRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    arxiv_id: str = Field(min_length=1, max_length=64)
    title: str = Field(max_length=1000)
    abstract: str = Field(max_length=50000)
    primary_cat: str = Field(max_length=128)
    all_cats: str = Field(max_length=1000)
    label: int = Field(ge=0, le=3)

    @field_validator("arxiv_id")
    @classmethod
    def nonblank_id(cls, value: str) -> str:
        if not value.strip() or any(ord(char) < 32 for char in value):
            raise ValueError("arxiv_id must contain printable, non-whitespace text")
        return value


class PaperOut(PaperRecord):
    split: Literal["train", "validation", "test"]


class PaperList(BaseModel):
    items: list[PaperOut]
    total: int = Field(ge=0)
    offset: int = Field(ge=0)
    limit: int = Field(ge=1)


class ClassificationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    title: str = Field(max_length=1000)
    abstract: str = Field(max_length=50000)


class ClassificationResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_id: str
    label: int = Field(ge=0, le=3)
    category: str
    score_kind: Literal["decision_function_margin"]
    decision_scores: dict[str, float]


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    query: str = Field(min_length=1, max_length=4000)
    method: Literal["tfidf", "minilm"] = "tfidf"
    k: int = Field(default=10, ge=1, le=50)
    label: int | None = Field(default=None, ge=0, le=3)
    primary_cat: str | None = Field(default=None, max_length=128)

    @field_validator("query")
    @classmethod
    def nonblank_query(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must not be blank")
        return value


class SearchHit(PaperOut):
    score: float


class SearchResponse(BaseModel):
    method: Literal["tfidf", "minilm"]
    score_description: str
    items: list[SearchHit]


class ClassifierExperiment(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_id: str
    short_name: str
    phase: int
    family: str
    validation_macro_f1: float | None
    test_accuracy: float | None
    test_macro_f1: float | None
    selected_by_validation: bool


class RetrievalExperiment(BaseModel):
    method_id: str
    display_name: str
    precision: float
    recall: float
    mrr: float
    ndcg: float
    unjudged_in_top_k: float


class ExperimentResults(BaseModel):
    classifiers: list[ClassifierExperiment]
    retrieval: list[RetrievalExperiment]
    classifier_selection: dict[str, object]
    retrieval_judgment_note: str


class Overview(BaseModel):
    paper_count: int = Field(ge=0)
    category_count: int = Field(ge=0)
    classifier_count: int = Field(ge=0)
    retrieval_method_count: int = Field(ge=0)
    label_counts: dict[str, int]


class Health(BaseModel):
    status: Literal["ok", "not_ready"]
    database: Literal["connected", "unavailable"] | None = None
