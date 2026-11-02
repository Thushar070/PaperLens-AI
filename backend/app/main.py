from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any

from ml.models.pretrained.runtime import expose_existing_base_site_packages

expose_existing_base_site_packages()

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import Settings
from app.db.repository import DatabaseRepository
from app.schemas import (
    ClassificationRequest,
    ClassificationResponse,
    ExperimentResults,
    Health,
    LABEL_NAMES,
    Overview,
    PaperList,
    PaperOut,
    SearchRequest,
    SearchResponse,
)
from app.services.classification import ClassificationService
from app.services.experiments import ExperimentCatalog
from app.services.search import SearchService


logger = logging.getLogger("paperlens.api")


class JsonLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname.lower(),
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key in ("event", "method", "path", "status_code", "duration_ms"):
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonLogFormatter())
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


@dataclass
class AppServices:
    repository: Any
    classifier: Any
    search: Any
    experiments: Any


def build_services(settings: Settings) -> AppServices:
    repository = DatabaseRepository(settings.database_url)
    return AppServices(
        repository=repository,
        classifier=ClassificationService(settings.artifact_root, settings.model_id),
        search=SearchService(repository, settings.artifact_root),
        experiments=ExperimentCatalog(settings.artifact_root),
    )


def create_app(settings: Settings | None = None, services: AppServices | None = None) -> FastAPI:
    config = settings or Settings.from_environment()
    application = FastAPI(
        title="PaperLens API",
        description="Paper catalog, saved-model classification, semantic retrieval, and experiment results.",
        version="1.0.0",
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=list(config.cors_origins),
        allow_methods=["GET", "POST"],
        allow_headers=["Accept", "Content-Type"],
    )
    application.state.settings = config
    application.state.services = services or build_services(config)

    @application.middleware("http")
    async def request_log(request: Request, call_next):
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            logger.exception("request failed", extra={"event": "request_failed", "method": request.method, "path": request.url.path})
            raise
        logger.info(
            "request completed",
            extra={
                "event": "request_completed",
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": round((time.perf_counter() - started) * 1000, 3),
            },
        )
        return response

    @application.exception_handler(Exception)
    async def unhandled_error(_request: Request, error: Exception):
        logger.exception("unhandled API error", exc_info=error, extra={"event": "unhandled_error"})
        return JSONResponse(status_code=500, content={"detail": "An internal server error occurred"})

    prefix = config.api_prefix

    @application.get(f"{prefix}/health", response_model=Health, tags=["operations"])
    def health() -> Health:
        return Health(status="ok")

    @application.get(f"{prefix}/ready", response_model=Health, tags=["operations"])
    def readiness(request: Request) -> Health:
        if not request.app.state.services.repository.ready():
            raise HTTPException(status_code=503, detail="Paper database is unavailable")
        return Health(status="ok", database="connected")

    @application.get(f"{prefix}/overview", response_model=Overview, tags=["papers"])
    def overview(request: Request) -> Overview:
        services: AppServices = request.app.state.services
        try:
            experiment_results = services.experiments.response()
            return Overview(
                paper_count=services.repository.paper_count(),
                category_count=len(LABEL_NAMES),
                classifier_count=len(experiment_results["classifiers"]),
                retrieval_method_count=len(experiment_results["retrieval"]),
                label_counts=services.repository.label_counts(),
            )
        except Exception as error:
            logger.warning("overview unavailable", exc_info=error, extra={"event": "overview_unavailable"})
            raise HTTPException(status_code=503, detail="Paper database is unavailable") from error

    @application.get(f"{prefix}/papers", response_model=PaperList, tags=["papers"])
    def papers(
        request: Request,
        query: str | None = Query(default=None, max_length=300),
        label: int | None = Query(default=None, ge=0, le=3),
        primary_cat: str | None = Query(default=None, max_length=128),
        offset: int = Query(default=0, ge=0),
        limit: int = Query(default=25, ge=1, le=config.max_page_size),
    ) -> PaperList:
        try:
            rows, total = request.app.state.services.repository.list_papers(
                query=query, label=label, primary_cat=primary_cat, offset=offset, limit=limit
            )
            return PaperList(items=rows, total=total, offset=offset, limit=limit)
        except Exception as error:
            logger.warning("paper list unavailable", exc_info=error, extra={"event": "paper_list_unavailable"})
            raise HTTPException(status_code=503, detail="Paper database is unavailable") from error

    @application.get(f"{prefix}/papers/{{arxiv_id}}", response_model=PaperOut, tags=["papers"])
    def paper_detail(arxiv_id: str, request: Request) -> PaperOut:
        try:
            row = request.app.state.services.repository.get_paper(arxiv_id)
        except Exception as error:
            logger.warning("paper detail unavailable", exc_info=error, extra={"event": "paper_detail_unavailable"})
            raise HTTPException(status_code=503, detail="Paper database is unavailable") from error
        if row is None:
            raise HTTPException(status_code=404, detail="Paper not found")
        return PaperOut(**row)

    @application.post(f"{prefix}/classifications", response_model=ClassificationResponse, tags=["classification"])
    def classify(payload: ClassificationRequest, request: Request) -> ClassificationResponse:
        try:
            return ClassificationResponse(**request.app.state.services.classifier.predict(payload.title, payload.abstract))
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except Exception as error:
            logger.exception("saved classifier unavailable", exc_info=error, extra={"event": "classifier_unavailable"})
            raise HTTPException(status_code=503, detail="Saved classifier could not complete the request") from error

    @application.post(f"{prefix}/search", response_model=SearchResponse, tags=["search"])
    def search(payload: SearchRequest, request: Request) -> SearchResponse:
        try:
            rows = request.app.state.services.search.search(payload.query, payload.method, payload.k)
        except FileNotFoundError as error:
            logger.exception("search artifacts unavailable", exc_info=error, extra={"event": "search_artifacts_unavailable"})
            raise HTTPException(status_code=503, detail="Required saved search artifacts are unavailable") from error
        except Exception as error:
            logger.exception("search failed", exc_info=error, extra={"event": "search_unavailable"})
            raise HTTPException(status_code=503, detail="Search could not complete; verify database and saved artifacts") from error
        description = (
            "Cosine similarity from saved TF-IDF features; ranking score, not probability."
            if payload.method == "tfidf"
            else "Cosine similarity from saved Phase 7 MiniLM embeddings using pgvector; ranking score, not probability."
        )
        return SearchResponse(method=payload.method, score_description=description, items=rows)

    @application.get(f"{prefix}/experiments", response_model=ExperimentResults, tags=["experiments"])
    def experiments(request: Request) -> ExperimentResults:
        return ExperimentResults(**request.app.state.services.experiments.response())

    return application


app = create_app()
