"""Stage 5 — retrieval-augmented generation over the cleaned abstracts.

Abstracts are chunked with overlap, embedded locally with a sentence-transformer,
and stored in a persistent Chroma collection. `answer_question` retrieves the
nearest chunks and asks Groq to answer strictly from them.

Everything expensive is deferred: the torch, Chroma and Groq imports happen
inside the cached getters below rather than at module scope, so
`import arxiv_pipeline.rag` costs nothing. That keeps the plain SQL API
endpoints fast to start and lets this module be imported in tests without
pulling in a deep-learning stack.
"""
from functools import lru_cache
from typing import TYPE_CHECKING

from . import config
from .db import connect

if TYPE_CHECKING:  # for type checkers only — never imported at runtime
    import chromadb

PROMPT_TEMPLATE = """You are a research assistant helping answer questions about arXiv academic papers.

Use the following retrieved paper excerpts to answer the question.
Be concise, factual, and cite arxiv_ids where relevant.

Retrieved context:
{context}

Question: {question}

Answer:"""

PAPERS_FOR_EMBEDDING = """
    SELECT arxiv_id, title, abstract, primary_category,
           submitted_year, pub_status, first_author
    FROM papers
    WHERE abstract IS NOT NULL
    ORDER BY RANDOM()
    LIMIT ?
"""


@lru_cache(maxsize=1)
def get_embedder():
    """The sentence-transformer, loaded once on first use."""
    from sentence_transformers import SentenceTransformer

    print(f"Loading embedding model {config.EMBED_MODEL}...")
    return SentenceTransformer(config.EMBED_MODEL)


@lru_cache(maxsize=1)
def get_groq_client():
    from groq import Groq

    api_key = config.groq_api_key()
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return Groq(api_key=api_key)


@lru_cache(maxsize=1)
def get_collection() -> "chromadb.Collection":
    """The persistent Chroma collection, created on first use."""
    import chromadb

    client = chromadb.PersistentClient(path=str(config.VECTOR_STORE_DIR))
    return client.get_or_create_collection(
        name=config.COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )


def chunk_text(
    text: str,
    chunk_size: int = config.CHUNK_SIZE,
    overlap: int = config.CHUNK_OVERLAP,
) -> list[str]:
    """Split into word windows of `chunk_size` that overlap by `overlap` words."""
    words = text.split()
    chunks = []
    for start in range(0, len(words), chunk_size - overlap):
        chunks.append(" ".join(words[start:start + chunk_size]))
        if start + chunk_size >= len(words):
            break  # this window already reached the end of the text
    return chunks


def build_vector_store(rebuild: bool = False) -> "chromadb.Collection":
    """Embed a sample of abstracts into the collection. No-op if already built."""
    collection = get_collection()

    existing = collection.count()
    if existing > 0 and not rebuild:
        print(f"Vector store ready ({existing:,} chunks). Skipping rebuild.")
        return collection

    print("Building vector store...")
    with connect(row_factory=True) as conn:
        rows = conn.execute(PAPERS_FOR_EMBEDDING, (config.RAG_SAMPLE_SIZE,)).fetchall()
    print(f"  Loaded {len(rows):,} papers")

    ids, documents, metadatas = [], [], []
    for row in rows:
        for i, chunk in enumerate(chunk_text(row["abstract"])):
            ids.append(f"{row['arxiv_id']}_chunk{i}")
            documents.append(chunk)
            metadatas.append({
                "arxiv_id": str(row["arxiv_id"]),
                "title": str(row["title"] or ""),
                "category": str(row["primary_category"] or ""),
                "year": int(row["submitted_year"] or 0),
                "pub_status": str(row["pub_status"] or ""),
                "first_author": str(row["first_author"] or ""),
            })

    print(f"  Generated {len(ids):,} chunks, embedding now...")
    embedder = get_embedder()
    batch = config.EMBED_BATCH_SIZE
    # Embed and insert a batch at a time, so peak memory is one batch of
    # vectors rather than the whole matrix converted to Python floats.
    for start in range(0, len(ids), batch):
        stop = start + batch
        collection.add(
            ids=ids[start:stop],
            embeddings=embedder.encode(documents[start:stop]).tolist(),
            documents=documents[start:stop],
            metadatas=metadatas[start:stop],
        )
        print(f"  Inserted batch {start // batch + 1} of {-(-len(ids) // batch)}")

    print(f"Vector store built with {collection.count():,} chunks.")
    return collection


def build_where(category: str | None, year: int | None) -> dict | None:
    """Translate optional filters into a Chroma metadata filter."""
    clauses = []
    if category:
        clauses.append({"category": {"$eq": category}})
    if year:
        clauses.append({"year": {"$eq": int(year)}})

    if not clauses:
        return None
    # Chroma requires $and only when combining more than one clause.
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def retrieve(
    query: str,
    n_results: int = config.DEFAULT_N_RESULTS,
    category_filter: str | None = None,
    year_filter: int | None = None,
) -> list[dict]:
    """Return the nearest abstract chunks, closest first."""
    results = get_collection().query(
        query_embeddings=get_embedder().encode([query]).tolist(),
        n_results=n_results,
        where=build_where(category_filter, year_filter),
        include=["documents", "metadatas", "distances"],
    )

    if not results["ids"] or not results["ids"][0]:
        return []

    return [
        {
            "chunk_text": document,
            "arxiv_id": meta.get("arxiv_id", ""),
            "title": meta.get("title", ""),
            "category": meta.get("category", ""),
            "year": meta.get("year", 0),
            "pub_status": meta.get("pub_status", ""),
            "first_author": meta.get("first_author", ""),
            "distance": round(distance, 6),
        }
        for document, meta, distance in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        )
    ]


def generate_answer(question: str, chunks: list[dict]) -> str:
    """Answer `question` from `chunks` only; refuse when retrieval came back empty."""
    if not chunks:
        return "No relevant papers found for this query."

    context = "\n\n".join(
        f"Paper: {c['title']} ({c['arxiv_id']})\n"
        f"Category: {c['category']} | Year: {c['year']} | Status: {c['pub_status']}\n"
        f"Abstract excerpt: {c['chunk_text']}"
        for c in chunks
    )

    response = get_groq_client().chat.completions.create(
        model=config.GROQ_MODEL,
        messages=[{
            "role": "user",
            "content": PROMPT_TEMPLATE.format(context=context, question=question),
        }],
        temperature=0.2,
        max_tokens=config.MAX_ANSWER_TOKENS,
    )

    choice = response.choices[0]
    answer = (choice.message.content or "").strip()
    if not answer:
        # A reasoning model that exhausts its budget mid-thought returns an
        # empty content field. Say so rather than recording a blank answer.
        raise RuntimeError(
            f"Model returned no answer (finish_reason={choice.finish_reason}); "
            f"MAX_ANSWER_TOKENS={config.MAX_ANSWER_TOKENS} may be too low."
        )
    return answer


def answer_question(
    question: str,
    n_results: int = config.DEFAULT_N_RESULTS,
    category_filter: str | None = None,
    year_filter: int | None = None,
) -> dict:
    """Retrieve then generate — the full RAG round trip."""
    chunks = retrieve(question, n_results, category_filter, year_filter)
    return {
        "question": question,
        "answer": generate_answer(question, chunks),
        "sources": chunks,
        "model_used": config.MODEL_LABEL,
        "category_filter": category_filter,
        "year_filter": year_filter,
    }


if __name__ == "__main__":
    build_vector_store()
