"""SQLite trace logger for every RAG pipeline run."""

import json
import sqlite3
import logging
from datetime import datetime, timezone
from pathlib import Path
from contextlib import contextmanager
from config import cfg

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS traces (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    query_id            TEXT    NOT NULL,
    cycle_number        INTEGER NOT NULL DEFAULT 0,
    timestamp           TEXT    NOT NULL,
    original_query      TEXT    NOT NULL,
    ground_truth        TEXT,
    collections_searched TEXT,
    search_mode         TEXT,
    ecfr_filters        TEXT,
    guidance_filters    TEXT,
    rewritten_query     TEXT,
    planner_version     INTEGER DEFAULT 1,
    child_chunk_ids     TEXT,
    parent_chunk_ids    TEXT,
    num_children        INTEGER DEFAULT 0,
    num_parents         INTEGER DEFAULT 0,
    generated_answer    TEXT,
    generator_version   INTEGER DEFAULT 1,
    faithfulness        REAL,
    context_recall      REAL,
    context_precision   REAL,
    answer_relevancy    REAL,
    answer_correctness  REAL,
    diagnosis           TEXT,
    passed              INTEGER
);
CREATE INDEX IF NOT EXISTS idx_traces_query_id   ON traces(query_id);
CREATE INDEX IF NOT EXISTS idx_traces_cycle      ON traces(cycle_number);
CREATE INDEX IF NOT EXISTS idx_traces_diagnosis  ON traces(diagnosis);
CREATE INDEX IF NOT EXISTS idx_traces_passed     ON traces(passed);
"""


def _get_db_path() -> Path:
    path = cfg.storage.SQLITE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


@contextmanager
def _connection():
    conn = sqlite3.connect(str(_get_db_path()))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    with _connection() as conn:
        conn.executescript(SCHEMA)


def save_trace(state: dict) -> int:
    plan   = state.get("retrieval_plan")
    chunks = state.get("child_chunks", [])
    parents = state.get("parent_chunks", [])
    child_ids  = [c.chunk_id for c in chunks]
    parent_ids = [p.chunk_id for p in parents]

    row = {
        "query_id":            state.get("query_id", ""),
        "cycle_number":        state.get("cycle_number", 0),
        "timestamp":           datetime.now(timezone.utc).isoformat(),
        "original_query":      state.get("original_query", ""),
        "ground_truth":        state.get("ground_truth"),
        "collections_searched": json.dumps(plan.collections)    if plan else None,
        "search_mode":          plan.search_mode                 if plan else None,
        "ecfr_filters":         json.dumps(plan.ecfr_filters)   if plan else None,
        "guidance_filters":     json.dumps(plan.guidance_filters) if plan else None,
        "rewritten_query":      plan.rewritten_query              if plan else None,
        "planner_version":      state.get("planner_version", 1),
        "child_chunk_ids":      json.dumps(child_ids),
        "parent_chunk_ids":     json.dumps(parent_ids),
        "num_children":         len(child_ids),
        "num_parents":          len(parent_ids),
        "generated_answer":     state.get("generated_answer", ""),
        "generator_version":    state.get("generator_version", 1),
    }

    sql = """
        INSERT INTO traces (
            query_id, cycle_number, timestamp, original_query, ground_truth,
            collections_searched, search_mode, ecfr_filters, guidance_filters,
            rewritten_query, planner_version,
            child_chunk_ids, parent_chunk_ids, num_children, num_parents,
            generated_answer, generator_version
        ) VALUES (
            :query_id, :cycle_number, :timestamp, :original_query, :ground_truth,
            :collections_searched, :search_mode, :ecfr_filters, :guidance_filters,
            :rewritten_query, :planner_version,
            :child_chunk_ids, :parent_chunk_ids, :num_children, :num_parents,
            :generated_answer, :generator_version
        )
    """
    with _connection() as conn:
        cursor = conn.execute(sql, row)
        row_id = cursor.lastrowid
    return row_id


def update_eval_scores(row_id: int, scores: dict, diagnosis: str, passed: bool):
    sql = """
        UPDATE traces SET
            faithfulness       = :faithfulness,
            context_recall     = :context_recall,
            context_precision  = :context_precision,
            answer_relevancy   = :answer_relevancy,
            answer_correctness = :answer_correctness,
            diagnosis          = :diagnosis,
            passed             = :passed
        WHERE id = :row_id
    """
    with _connection() as conn:
        conn.execute(sql, {
            "faithfulness":       scores.get("faithfulness"),
            "context_recall":     scores.get("context_recall"),
            "context_precision":  scores.get("context_precision"),
            "answer_relevancy":   scores.get("answer_relevancy"),
            "answer_correctness": scores.get("answer_correctness"),
            "diagnosis":          diagnosis,
            "passed":             1 if passed else 0,
            "row_id":             row_id,
        })


def get_recent_failures(cycle_number: int, min_count: int = 10) -> list[dict]:
    sql = """
        SELECT * FROM traces
        WHERE cycle_number = ? AND passed = 0 AND diagnosis IS NOT NULL
        ORDER BY timestamp DESC
    """
    with _connection() as conn:
        rows = conn.execute(sql, (cycle_number,)).fetchall()
    return [dict(r) for r in rows]


def get_dominant_failure(cycle_number: int) -> str | None:
    sql = """
        SELECT diagnosis, COUNT(*) as cnt
        FROM traces
        WHERE cycle_number = ? AND passed = 0 AND diagnosis IS NOT NULL
        GROUP BY diagnosis ORDER BY cnt DESC LIMIT 1
    """
    with _connection() as conn:
        row = conn.execute(sql, (cycle_number,)).fetchone()
    return row["diagnosis"] if row else None


def get_passing_traces(cycle_number: int, limit: int = 20) -> list[dict]:
    sql = """
        SELECT * FROM traces
        WHERE cycle_number = ? AND passed = 1
        ORDER BY RANDOM() LIMIT ?
    """
    with _connection() as conn:
        rows = conn.execute(sql, (cycle_number, limit)).fetchall()
    return [dict(r) for r in rows]


def get_cycle_summary(cycle_number: int) -> dict:
    sql = """
        SELECT
            COUNT(*)                                    AS total,
            SUM(CASE WHEN passed = 1 THEN 1 ELSE 0 END) AS passed_count,
            AVG(faithfulness)                           AS avg_faithfulness,
            AVG(context_recall)                        AS avg_context_recall,
            AVG(context_precision)                     AS avg_context_precision,
            AVG(answer_relevancy)                      AS avg_answer_relevancy
        FROM traces
        WHERE cycle_number = ? AND diagnosis IS NOT NULL
    """
    with _connection() as conn:
        row = conn.execute(sql, (cycle_number,)).fetchone()
    if not row or row["total"] == 0:
        return {}
    total = row["total"]
    return {
        "cycle":             cycle_number,
        "total":             total,
        "passed":            row["passed_count"],
        "accuracy":          round(row["passed_count"] / total, 4),
        "avg_faithfulness":  round(row["avg_faithfulness"] or 0, 4),
        "avg_context_recall": round(row["avg_context_recall"] or 0, 4),
        "avg_context_precision": round(row["avg_context_precision"] or 0, 4),
        "avg_answer_relevancy": round(row["avg_answer_relevancy"] or 0, 4),
    }


init_db()
