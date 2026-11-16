"""Basic tests for Agri Talk's local retrieval and language helpers."""

from agritalk.language import detect_language
from agritalk.rag import load_documents, retrieve_documents


def test_knowledge_base_loads():
    docs = load_documents()
    assert len(docs) >= 8
    assert all(document.get("source_url", "").startswith("http") for document in docs)


def test_grain_storage_query_retrieves_storage_source():
    docs = retrieve_documents("How do I reduce moisture and insects when storing harvested grain?")
    assert docs
    assert any("storage" in doc["title"].lower() or "grain" in doc["title"].lower() for doc in docs)


def test_soil_test_query_retrieves_soil_health_source():
    docs = retrieve_documents("Why should I test soil before applying fertilizer?")
    assert docs
    assert any("Soil Health Card" in doc["source_title"] for doc in docs)


def test_black_soil_query_retrieves_black_soil_sources():
    docs = retrieve_documents("What can grow on black soil?")
    assert docs
    assert docs[0]["id"].startswith("black_soil")
    assert "black soil" in docs[0]["topics"]


def test_knowledge_base_loads_imported_background_tier():
    docs = load_documents()
    tiers = {doc["tier"] for doc in docs}
    assert tiers == {"curated", "background"}
    ids = [doc["id"] for doc in docs]
    assert len(ids) == len(set(ids))


def test_curated_sources_rank_before_imported_background_entries():
    docs = retrieve_documents("What is drip irrigation?")
    assert docs
    assert docs[0]["id"] == "drip_irrigation"
    first_background = next(
        (index for index, doc in enumerate(docs) if doc["tier"] == "background"), len(docs)
    )
    assert all(doc["tier"] == "curated" for doc in docs[:first_background])


def test_unrelated_query_is_not_force_matched():
    docs = retrieve_documents("Who won the football world cup in 1998?")
    assert docs == []


def test_language_detection_by_script():
    assert detect_language("मिट्टी की जाँच क्यों जरूरी है?", "auto") == "hi"
    assert detect_language("மண்ணை ஏன் பரிசோதிக்க வேண்டும்?", "auto") == "ta"
    assert detect_language("How do I store grain?", "auto") == "en"


def test_explicit_language_overrides_script_detection():
    assert detect_language("namaste farmer", "hi") == "hi"
    assert detect_language("vanakkam vivasayi", "ta") == "ta"
