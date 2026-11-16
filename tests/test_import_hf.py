"""Tests for the Hugging Face knowledge-base importer."""

from scripts.import_hf_dataset import build_entries, is_usable_row, topics_from_question


def test_row_with_agrochemical_dose_is_rejected():
    assert not is_usable_row(
        "How do I control leaf curl in chilli?",
        "Spray rogor 35 EC at 2 ml per litre of water every 15 days.",
        60,
        700,
    )


def test_generic_agronomy_row_is_accepted():
    assert is_usable_row(
        "Why is crop rotation important in farming?",
        "Crop rotation helps prevent soil erosion and depletion and can reduce pests and diseases.",
        60,
        700,
    )


def test_short_answer_and_non_ascii_rows_are_rejected():
    assert not is_usable_row("What is crop rotation?", "It is rotation.", 60, 700)
    assert not is_usable_row("गेहूं कैसे स्टोर करें?", "ठंडे और सूखे स्थान पर रखें, नमी से बचाएँ।", 20, 700)


def test_non_agricultural_question_is_rejected():
    assert not is_usable_row(
        "Who won the football world cup in 1998?",
        "France won the tournament hosted by France in that year.",
        20,
        700,
    )


def test_duplicate_questions_are_collapsed():
    rows = [
        {"question": "What is crop rotation?", "answers": "Growing different crops in sequence in the same field."},
        {"question": "what is crop rotation", "answers": "A different wording of the same text about rotation."},
    ]
    entries = build_entries(rows, "source", "https://example.org", 20, 700)
    assert len(entries) == 1


def test_colliding_slugs_get_unique_ids():
    question_a = "How can crop rotation be used to control soil borne pests in wheat?"
    question_b = "How can crop rotation be used to control soil borne pests in maize?"
    rows = [
        {"question": question_a, "answers": "Rotate cereals with pulses to break the pest cycle in the field."},
        {"question": question_b, "answers": "Rotate cereals with legumes to break the pest cycle in the field."},
    ]
    entries = build_entries(rows, "source", "https://example.org", 20, 700)
    ids = [entry["id"] for entry in entries]
    assert len(ids) == len(set(ids)) == 2


def test_topics_are_derived_from_the_question():
    topics = topics_from_question("Why is crop rotation important in farming?")
    assert "crop" in topics and "rotation" in topics
    assert "why" not in topics
