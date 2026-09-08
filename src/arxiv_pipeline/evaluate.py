"""Stage 6 â€” run the question set against a live API and grade the answers.

Each question in `evaluation/questions.json` carries a `grading` block; the
keyword_match grader counts how many expected terms appear in the generated
answer and passes when the minimum is met. Results, sources and per-question
grades are written to `evaluation/answers.json`.

Transport failures are recorded in an `error` field rather than smuggled into
the answer text, so a timeout is distinguishable from a genuinely poor answer.
"""
import argparse
import json
import time

import requests

from . import config

DEFAULT_SERVER = "http://127.0.0.1:8000"
REQUEST_TIMEOUT = 30
# The Groq free tier rate-limits, so pace the requests.
PAUSE_SECONDS = 4


def grade(answer: str, grading: dict | None) -> dict | None:
    """Score one answer against its grading spec."""
    if not grading or grading.get("type") != "keyword_match":
        return None

    expected = grading.get("must_contain_any", [])
    minimum = grading.get("min_keyword_hits", 1)
    lowered = answer.lower()
    hits = [term for term in expected if term.lower() in lowered]

    return {
        "type": "keyword_match",
        "keyword_hits": len(hits),
        "min_keyword_hits": minimum,
        "matched": hits,
        "passed": len(hits) >= minimum,
    }


def record(question: dict, payload: dict, error: str | None = None) -> dict:
    """One result row, shaped the same whether the call succeeded or failed.

    `payload` is the server's own response, so `model_used` reflects what
    actually answered rather than what this machine happens to have configured.
    """
    answer = payload.get("answer", "")
    return {
        "question_id": str(question.get("id", "unknown")),
        "question": question["question"],
        "answer": answer,
        "sources": payload.get("sources", []),
        "model_used": payload.get("model_used", config.MODEL_LABEL),
        "category_filter": question.get("category_filter"),
        "year_filter": question.get("year_filter"),
        "error": error,
        "grade": None if error else grade(answer, question.get("grading")),
    }


def ask(session: requests.Session, server: str, question: dict) -> dict:
    """POST one question, returning a row even when the call fails."""
    year = question.get("year_filter")
    body = {
        "question": question["question"],
        "n_results": config.DEFAULT_N_RESULTS,
        "category_filter": question.get("category_filter"),
        "year_filter": int(year) if year else None,
    }

    try:
        response = session.post(f"{server}/query", json=body, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
    except requests.exceptions.Timeout:
        return record(question, {}, error=f"timed out after {REQUEST_TIMEOUT}s")
    except Exception as exc:
        return record(question, {}, error=str(exc))

    return record(question, response.json())


def describe(row: dict) -> str:
    """The one-line outcome for the console."""
    if row["error"]:
        return f"ERROR â€” {row['error']}"

    graded = row["grade"]
    if not graded:
        return f"OK â€” {len(row['sources'])} sources"

    verdict = "PASS" if graded["passed"] else "FAIL"
    return (
        f"OK â€” {len(row['sources'])} sources, grade {verdict} "
        f"({graded['keyword_hits']}/{graded['min_keyword_hits']} hits)"
    )


def run(server: str = DEFAULT_SERVER) -> list[dict]:
    questions = json.loads(config.QUESTIONS_PATH.read_text(encoding="utf-8"))
    print(f"Running {len(questions)} questions against {server}/query\n")

    results = []
    with requests.Session() as session:
        for index, question in enumerate(questions):
            print(f"  [q{question.get('id', '?')}] {question['question']}")
            started = time.monotonic()
            row = ask(session, server, question)
            print(f"         {describe(row)}")
            results.append(row)

            # Pace against elapsed time, and never sleep after the last
            # question — nothing follows it to rate-limit against.
            if index < len(questions) - 1:
                time.sleep(max(0.0, PAUSE_SECONDS - (time.monotonic() - started)))

    config.ANSWERS_PATH.write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n{len(results)} answers saved to {config.ANSWERS_PATH.name}")

    errors = [row for row in results if row["error"]]
    graded = [row["grade"] for row in results if row["grade"]]
    if graded:
        passed = sum(g["passed"] for g in graded)
        print(f"Grading: {passed}/{len(graded)} passed ({passed / len(graded):.0%})")
    if errors:
        print(f"{len(errors)} question(s) failed to reach the server.")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", default=DEFAULT_SERVER, help="API base URL")
    run(parser.parse_args().server)
