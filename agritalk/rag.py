"""Lightweight retrieval layer for Agri Talk.

This is a small TF-IDF RAG prototype. It keeps the knowledge base local,
retrieves relevant English passages, and returns their source metadata. It is
intentionally lightweight so it can run on a student laptop and on Streamlit
Community Cloud without downloading a large embedding model.
"""

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_KB_DIR = PROJECT_ROOT / "knowledge_base"


def _read_kb_file(path: Path) -> List[Dict]:
    with Path(path).open("r", encoding="utf-8") as file:
        documents = json.load(file)

    if not isinstance(documents, list) or not documents:
        raise ValueError(f"Knowledge-base file {path.name} must be a non-empty JSON list.")

    required = {"id", "title", "content", "topics", "source_title", "source_url"}
    for index, document in enumerate(documents):
        missing = required - set(document)
        if missing:
            raise ValueError(
                f"Knowledge-base item {index} in {path.name} is missing fields: {sorted(missing)}"
            )
    return documents


def _tier_for(path: Path) -> str:
    """Imported dataset files are background material; everything else is curated."""
    return "background" if path.stem.startswith("hf_") else "curated"


def load_documents(path: Optional[Path] = None) -> List[Dict]:
    """Load the curated knowledge base, or a single JSON file when a path is given.

    Every ``*.json`` file in ``knowledge_base/`` is loaded, so curated entries and
    imported datasets can live side by side. Files named ``hf_*.json`` are tagged
    as background material and only fill retrieval slots left by curated entries.
    """
    if path is not None:
        documents = _read_kb_file(Path(path))
        for document in documents:
            document.setdefault("tier", "curated")
        return documents

    documents: List[Dict] = []
    for kb_file in sorted(DEFAULT_KB_DIR.glob("*.json")):
        for document in _read_kb_file(kb_file):
            document["tier"] = _tier_for(kb_file)
            documents.append(document)
    if not documents:
        raise ValueError(f"No knowledge-base JSON files found in {DEFAULT_KB_DIR}")

    ids = [document["id"] for document in documents]
    duplicates = {item for item in ids if ids.count(item) > 1}
    if duplicates:
        raise ValueError(f"Duplicate knowledge-base ids: {sorted(duplicates)[:5]}")
    return documents


def _document_text(document: Dict) -> str:
    topics = " ".join(document.get("topics", []))
    return f"{document['title']} {topics} {document['content']}"


@lru_cache(maxsize=1)
def _build_index():
    """Build and cache two small TF-IDF indexes: word and character n-grams."""
    documents = load_documents()
    texts = [_document_text(document) for document in documents]

    word_vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    char_vectorizer = TfidfVectorizer(
        lowercase=True,
        analyzer="char_wb",
        ngram_range=(3, 5),
        sublinear_tf=True,
        min_df=1,
    )

    word_matrix = word_vectorizer.fit_transform(texts)
    char_matrix = char_vectorizer.fit_transform(texts)
    return documents, word_vectorizer, word_matrix, char_vectorizer, char_matrix


BACKGROUND_MIN_SCORE = 0.25


def retrieve_documents(
    query: str,
    top_k: int = 4,
    min_score: float = 0.075,
    background_min_score: float = BACKGROUND_MIN_SCORE,
) -> List[Dict]:
    """Return relevant documents ranked by a blended lexical similarity score.

    Non-English questions should be translated to English before calling this
    function. Word n-grams provide the main signal; character n-grams help with
    minor typos and word-form differences. This is retrieval, not a guarantee
    of semantic understanding, so the caller must handle low-evidence queries.

    Curated source-backed entries are ranked first. Imported background entries
    (``hf_*.json``) only fill the remaining slots and need a higher score, since
    their short generic answers otherwise outrank authoritative passages.
    """
    query = (query or "").strip()
    if not query:
        return []

    documents, word_vectorizer, word_matrix, char_vectorizer, char_matrix = _build_index()
    word_query = word_vectorizer.transform([query])
    char_query = char_vectorizer.transform([query])

    word_scores = cosine_similarity(word_query, word_matrix).ravel()
    char_scores = cosine_similarity(char_query, char_matrix).ravel()
    combined_scores = (0.75 * word_scores) + (0.25 * char_scores)

    curated = []
    background = []
    for index in np.argsort(combined_scores)[::-1]:
        score = float(combined_scores[index])
        document = documents[int(index)]
        if document.get("tier", "curated") == "background":
            if score >= background_min_score:
                background.append((score, document))
        elif score >= min_score:
            curated.append((score, document))

    results = []
    for score, document in [*curated[:top_k], *background]:
        if len(results) >= max(1, top_k):
            break
        item = dict(document)
        item["score"] = round(score, 4)
        results.append(item)
    return results
