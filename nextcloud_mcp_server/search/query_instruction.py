"""Query-side instruction prefix for instruction-aware embedding models.

Models such as Qwen3-Embedding and E5-instruct are trained asymmetrically: a
search *query* is embedded with a one-line task instruction in front of it,
while *documents* are embedded as plain text. Qwen reports a 1-5% retrieval drop
when the query instruction is omitted.

``EMBEDDING_QUERY_INSTRUCTION`` opts into that format. It is applied only where a
search query is turned into a dense vector, never on the indexing path, so the
stored document vectors (and the collection) are unaffected and no reindex is
needed to turn it on, change it, or turn it off. The BM25 sparse side always
sees the raw query: keyword matching on the instruction text would be noise.
"""

from __future__ import annotations

from typing import Any


def format_query_for_embedding(query: str, instruction: str | None) -> str:
    """Return the text to embed for ``query``.

    With an empty/unset ``instruction`` the query is returned unchanged (the
    default, and the pre-existing behaviour). Otherwise it is wrapped in the
    Qwen3-Embedding / E5-instruct query format::

        Instruct: {instruction}
        Query: {query}
    """
    if not isinstance(instruction, str):
        return query
    instruction = instruction.strip()
    if not instruction:
        return query
    return f"Instruct: {instruction}\nQuery: {query}"


def query_embedding_text(query: str, settings: Any) -> str:
    """``format_query_for_embedding`` using the configured instruction."""
    return format_query_for_embedding(
        query, getattr(settings, "embedding_query_instruction", None)
    )
