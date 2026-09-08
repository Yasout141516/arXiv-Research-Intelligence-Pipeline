"""FastAPI service exposing the cleaned tables and the RAG endpoint."""
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from . import config, rag
from .db import connect, scalar

app = FastAPI(
    title="arXiv Research Intelligence API",
    description=(
        f"Query {config.SAMPLE_SIZE:,} cleaned arXiv CS papers, "
        "or ask questions answered by RAG."
    ),
    version="1.0.0",
)

PAPER_FIELDS = (
    "arxiv_id, title, primary_category, submitted_year, pub_status, first_author"
)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Natural-language question")
    n_results: int = Field(config.DEFAULT_N_RESULTS, ge=1, le=20)
    category_filter: str | None = Field(None, description="e.g. cs.AI")
    year_filter: int | None = Field(None, description="e.g. 2023")


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[dict]
    model_used: str
    category_filter: str | None
    year_filter: int | None


def paper_filters(category: str | None, year: int | None) -> tuple[str, tuple]:
    """Build the shared WHERE fragment used by both the page and count queries."""
    clauses, params = [], []
    if category:
        clauses.append("primary_category = ?")
        params.append(category)
    if year:
        clauses.append("submitted_year = ?")
        params.append(year)
    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    return where, tuple(params)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "message": "arXiv RAG server is running"}


@app.get("/papers")
def get_papers(
    category: str | None = Query(None, description="Filter by primary_category e.g. cs.AI"),
    year: int | None = Query(None, description="Filter by submitted_year e.g. 2023"),
    limit: int = Query(20, ge=1, le=200, description="Max results to return"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
) -> dict:
    where, params = paper_filters(category, year)

    with connect(row_factory=True) as conn:
        rows = conn.execute(
            f"SELECT {PAPER_FIELDS} FROM papers{where} LIMIT ? OFFSET ?",
            [*params, limit, offset],
        ).fetchall()
        total = scalar(conn, f"SELECT COUNT(*) FROM papers{where}", params)

    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "papers": [dict(row) for row in rows],
    }


@app.post("/query", response_model=QueryResponse)
def query_endpoint(req: QueryRequest) -> QueryResponse:
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    return QueryResponse(**rag.answer_question(
        question=req.question,
        n_results=req.n_results,
        category_filter=req.category_filter,
        year_filter=req.year_filter,
    ))
