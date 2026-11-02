from __future__ import annotations

import math
from typing import Any, Sequence

from sqlalchemy import bindparam, create_engine, text
from sqlalchemy.engine import Engine


PAPER_COLUMNS = "arxiv_id, title, abstract, primary_cat, all_cats, label, split"


class DatabaseRepository:
    """Small SQLAlchemy repository; engine creation is lazy for health reporting."""

    def __init__(self, database_url: str, engine: Engine | None = None) -> None:
        self.database_url = database_url
        self._engine = engine

    @property
    def engine(self) -> Engine:
        if self._engine is None:
            self._engine = create_engine(self.database_url, pool_pre_ping=True, future=True)
        return self._engine

    def ready(self) -> bool:
        try:
            with self.engine.connect() as connection:
                connection.execute(text("SELECT 1"))
            return True
        except Exception:
            return False

    def paper_count(self) -> int:
        with self.engine.connect() as connection:
            return int(connection.execute(text("SELECT count(*) FROM papers")).scalar_one())

    def label_counts(self) -> dict[str, int]:
        with self.engine.connect() as connection:
            rows = connection.execute(text("SELECT label, count(*) AS n FROM papers GROUP BY label")).all()
        names = ("Algorithms", "NLP/AI", "Systems/Infra", "Vision")
        return {names[int(label)]: int(count) for label, count in rows}

    def list_papers(
        self,
        *,
        query: str | None = None,
        label: int | None = None,
        primary_cat: str | None = None,
        offset: int = 0,
        limit: int = 25,
    ) -> tuple[list[dict[str, Any]], int]:
        clauses: list[str] = []
        parameters: dict[str, Any] = {"offset": offset, "limit": limit}
        if query:
            clauses.append("(lower(title) LIKE lower(:pattern) OR lower(abstract) LIKE lower(:pattern))")
            parameters["pattern"] = f"%{query}%"
        if label is not None:
            clauses.append("label = :label")
            parameters["label"] = label
        if primary_cat:
            clauses.append("primary_cat = :primary_cat")
            parameters["primary_cat"] = primary_cat
        where_sql = " WHERE " + " AND ".join(clauses) if clauses else ""
        with self.engine.connect() as connection:
            total = int(connection.execute(text("SELECT count(*) FROM papers" + where_sql), parameters).scalar_one())
            rows = connection.execute(
                text(f"SELECT {PAPER_COLUMNS} FROM papers{where_sql} ORDER BY arxiv_id DESC LIMIT :limit OFFSET :offset"),
                parameters,
            ).mappings().all()
        return [dict(row) for row in rows], total

    def get_paper(self, arxiv_id: str) -> dict[str, Any] | None:
        with self.engine.connect() as connection:
            row = connection.execute(
                text(f"SELECT {PAPER_COLUMNS} FROM papers WHERE arxiv_id = :arxiv_id"),
                {"arxiv_id": arxiv_id},
            ).mappings().first()
        return dict(row) if row is not None else None

    def get_papers_by_ids(self, arxiv_ids: Sequence[str]) -> dict[str, dict[str, Any]]:
        if not arxiv_ids:
            return {}
        statement = text(f"SELECT {PAPER_COLUMNS} FROM papers WHERE arxiv_id IN :ids").bindparams(
            bindparam("ids", expanding=True)
        )
        with self.engine.connect() as connection:
            rows = connection.execute(statement, {"ids": list(arxiv_ids)}).mappings().all()
        return {str(row["arxiv_id"]): dict(row) for row in rows}

    def vector_search(self, vector: Sequence[float], *, k: int) -> list[dict[str, Any]]:
        if len(vector) != 384 or any(not math.isfinite(float(value)) for value in vector):
            raise ValueError("pgvector search requires exactly 384 finite values")
        encoded = "[" + ",".join(format(float(value), ".8g") for value in vector) + "]"
        statement = text(
            "SELECT p.arxiv_id, p.title, p.abstract, p.primary_cat, p.all_cats, p.label, p.split, "
            "1 - (e.embedding <=> CAST(:vector AS vector)) AS score "
            "FROM paper_embeddings AS e JOIN papers AS p USING (arxiv_id) "
            "ORDER BY e.embedding <=> CAST(:vector AS vector), p.arxiv_id LIMIT :limit"
        )
        with self.engine.connect() as connection:
            rows = connection.execute(statement, {"vector": encoded, "limit": k}).mappings().all()
        return [dict(row) for row in rows]
